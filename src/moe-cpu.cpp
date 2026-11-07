#include "moe-cpu.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {

typedef struct ggml_threadpool * (*ggml_threadpool_new_t)(struct ggml_threadpool_params * params);
typedef void (*ggml_threadpool_free_t)(struct ggml_threadpool * threadpool);
typedef void (*ggml_backend_cpu_set_threadpool_t)(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
typedef void (*ggml_backend_set_n_threads_t)(ggml_backend_t backend, int n_threads);

struct aligned_buffer {
    void * ptr = nullptr;
    size_t size = 0;

    aligned_buffer() = default;

    aligned_buffer(size_t sz, size_t alignment = 64) {
        alloc(sz, alignment);
    }

    ~aligned_buffer() {
        free_mem();
    }

    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;

    aligned_buffer(aligned_buffer && o) noexcept : ptr(o.ptr), size(o.size) {
        o.ptr = nullptr;
        o.size = 0;
    }

    aligned_buffer & operator=(aligned_buffer && o) noexcept {
        if (this != &o) {
            free_mem();
            ptr = o.ptr;
            size = o.size;
            o.ptr = nullptr;
            o.size = 0;
        }
        return *this;
    }

    void alloc(size_t sz, size_t alignment = 64) {
        free_mem();
        if (sz == 0) {
            return;
        }
        size = sz;
#if defined(_WIN32)
        ptr = _aligned_malloc(sz, alignment);
#else
        if (posix_memalign(&ptr, alignment, sz) != 0) {
            ptr = nullptr;
        }
#endif
        GGML_ASSERT(ptr != nullptr);
    }

    void free_mem() {
        if (ptr) {
#if defined(_WIN32)
            _aligned_free(ptr);
#else
            free(ptr);
#endif
            ptr = nullptr;
            size = 0;
        }
    }

    void * data() { return ptr; }
    const void * data() const { return ptr; }
};

struct geom_key {
    ggml_type types[3];
    size_t expert_stride;
    int64_t n_embd;
    int64_t n_ff;
    int32_t n_slots;
    float clamp;

    bool operator==(const geom_key & o) const {
        return types[0] == o.types[0] &&
               types[1] == o.types[1] &&
               types[2] == o.types[2] &&
               expert_stride == o.expert_stride &&
               n_embd == o.n_embd &&
               n_ff == o.n_ff &&
               n_slots == o.n_slots &&
               clamp == o.clamp;
    }
};

struct plan_entry {
    struct ggml_context * ctx = nullptr;
    struct ggml_cgraph * gf = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;

    struct ggml_tensor * w_gate = nullptr;
    struct ggml_tensor * w_up = nullptr;
    struct ggml_tensor * w_down = nullptr;
    struct ggml_tensor * cur = nullptr;
    struct ggml_tensor * ids = nullptr;
    struct ggml_tensor * down = nullptr;
};

} // namespace

struct moe_cpu_executor::impl {
    ggml_backend_t backend_cpu = nullptr;
    struct ggml_threadpool * threadpool = nullptr;
    ggml_threadpool_free_t threadpool_free_fn = nullptr;

    std::vector<moe_cpu_layer> layers;
    int n_threads = 1;
    int max_tokens = 1;

    std::vector<geom_key> unique_geoms;
    std::vector<int> layer_geom;

    // Plans per unique geometry and job count: [geom_id][n_jobs]
    std::vector<std::vector<plan_entry>> geom_plans;

    // Persistent scratch buffers sized for max_tokens
    aligned_buffer gathered_input_buf;
    aligned_buffer buf_gate;
    aligned_buffer buf_up;
    aligned_buffer buf_act;
    aligned_buffer dummy_slots_buf;
    aligned_buffer dummy_out_buf;

    impl(ggml_backend_dev_t cpu_device, const std::vector<moe_cpu_layer> & layers_in, int n_th, int max_tok)
        : layers(layers_in),
          n_threads(n_th),
          max_tokens(max_tok) {

        GGML_ASSERT(!layers.empty());
        GGML_ASSERT(n_threads > 0 && max_tokens > 0 && max_tokens <= 64);

        if (cpu_device == nullptr) {
            cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        GGML_ASSERT(cpu_device != nullptr && "CPU backend device not found");

        backend_cpu = ggml_backend_dev_init(cpu_device, nullptr);
        GGML_ASSERT(backend_cpu != nullptr);

        auto * reg = ggml_backend_dev_backend_reg(cpu_device);
        GGML_ASSERT(reg != nullptr);
        auto threadpool_new_fn = (ggml_threadpool_new_t) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        threadpool_free_fn = (ggml_threadpool_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        auto set_threadpool_fn = (ggml_backend_cpu_set_threadpool_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        auto set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        GGML_ASSERT(threadpool_new_fn && threadpool_free_fn && set_threadpool_fn && set_n_threads_fn);
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        tpp.poll = 0;
        threadpool = threadpool_new_fn(&tpp);
        GGML_ASSERT(threadpool != nullptr);
        set_threadpool_fn(backend_cpu, threadpool);
        set_n_threads_fn(backend_cpu, n_threads);

        int64_t max_embd = 0;
        int64_t max_ff   = 0;
        for (const auto & l : layers) {
            GGML_ASSERT(l.n_embd > 0);
            GGML_ASSERT(l.n_ff > 0);
            GGML_ASSERT(l.n_slots > 0);
            GGML_ASSERT(l.expert_stride > 0);
            GGML_ASSERT(l.weights[0] && l.weights[1] && l.weights[2]);
            GGML_ASSERT(l.expert_stride <= SIZE_MAX / (size_t) l.n_slots);

            max_embd = std::max(max_embd, l.n_embd);
            max_ff   = std::max(max_ff,   l.n_ff);
        }

        if ((size_t) max_embd > SIZE_MAX / sizeof(float) || (size_t) max_ff > SIZE_MAX / sizeof(float) ||
            (size_t) max_tokens > SIZE_MAX / ((size_t) max_embd * sizeof(float)) ||
            (size_t) max_tokens > SIZE_MAX / ((size_t) max_ff * sizeof(float))) {
            GGML_ABORT("buffer allocation size overflow");
        }

        const size_t input_bytes = (size_t) max_tokens * (size_t) max_embd * sizeof(float);
        const size_t ff_bytes    = (size_t) max_tokens * (size_t) max_ff * sizeof(float);

        gathered_input_buf.alloc(input_bytes);
        buf_gate.alloc(ff_bytes);
        buf_up.alloc(ff_bytes);
        buf_act.alloc(ff_bytes);

        dummy_slots_buf.alloc((size_t) max_tokens * sizeof(int32_t));
        memset(dummy_slots_buf.data(), 0, (size_t) max_tokens * sizeof(int32_t));
        dummy_out_buf.alloc(input_bytes);

        // Group layers by geometry/quant types/clamp
        layer_geom.resize(layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto & l = layers[i];
            geom_key k = {
                { l.types[0], l.types[1], l.types[2] },
                l.expert_stride,
                l.n_embd,
                l.n_ff,
                l.n_slots,
                l.clamp
            };
            int found = -1;
            for (size_t g = 0; g < unique_geoms.size(); ++g) {
                if (unique_geoms[g] == k) {
                    found = (int) g;
                    break;
                }
            }
            if (found < 0) {
                found = (int) unique_geoms.size();
                unique_geoms.push_back(k);
            }
            layer_geom[i] = found;
        }

        geom_plans.resize(unique_geoms.size());
        for (size_t g = 0; g < unique_geoms.size(); ++g) {
            geom_plans[g].resize((size_t) max_tokens + 1);
        }

        for (size_t g = 0; g < unique_geoms.size(); ++g) {
            for (int j = 1; j <= max_tokens; ++j) {
                create_plan_locked((int) g, j);
            }
        }
    }

    ~impl() {
        for (auto & geom_vec : geom_plans) {
            for (auto & entry : geom_vec) {
                if (entry.plan && backend_cpu) {
                    ggml_backend_graph_plan_free(backend_cpu, entry.plan);
                    entry.plan = nullptr;
                }
                if (entry.ctx) {
                    ggml_free(entry.ctx);
                    entry.ctx = nullptr;
                }
            }
            geom_vec.clear();
        }
        geom_plans.clear();

        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
            backend_cpu = nullptr;
        }

        if (threadpool && threadpool_free_fn) {
            threadpool_free_fn(threadpool);
            threadpool = nullptr;
        }
    }

    void create_plan_locked(int geom_id, int n_jobs) {
        if (geom_plans[geom_id][n_jobs].ctx != nullptr) {
            return;
        }

        const auto & geom = unique_geoms[geom_id];

        int rep_layer = 0;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layer_geom[i] == geom_id) {
                rep_layer = (int) i;
                break;
            }
        }

        const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false);
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        // Compact expert weights with full-expert stride on nb[2]
        struct ggml_tensor * w_gate = ggml_new_tensor_4d(ctx, geom.types[0], geom.n_embd, geom.n_ff, geom.n_slots, 1);
        w_gate->nb[2] = geom.expert_stride;
        w_gate->nb[3] = geom.expert_stride * (size_t) geom.n_slots;
        w_gate->data = const_cast<void *>(layers[rep_layer].weights[0]);

        struct ggml_tensor * w_up = ggml_new_tensor_4d(ctx, geom.types[1], geom.n_embd, geom.n_ff, geom.n_slots, 1);
        w_up->nb[2] = geom.expert_stride;
        w_up->nb[3] = geom.expert_stride * (size_t) geom.n_slots;
        w_up->data = const_cast<void *>(layers[rep_layer].weights[1]);

        struct ggml_tensor * w_down = ggml_new_tensor_4d(ctx, geom.types[2], geom.n_ff, geom.n_embd, geom.n_slots, 1);
        w_down->nb[2] = geom.expert_stride;
        w_down->nb[3] = geom.expert_stride * (size_t) geom.n_slots;
        w_down->data = const_cast<void *>(layers[rep_layer].weights[2]);

        // Input gathered view: [n_embd, 1, n_jobs, 1]
        struct ggml_tensor * cur = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, geom.n_embd, 1, n_jobs, 1);
        cur->data = gathered_input_buf.data();

        // Selected slot indices: [1, n_jobs, 1, 1]
        struct ggml_tensor * ids = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1, n_jobs, 1, 1);
        ids->data = dummy_slots_buf.data();

        // Gate matmul
        struct ggml_tensor * gate = ggml_mul_mat_id(ctx, w_gate, cur, ids);
        gate->data = buf_gate.data();

        // Up matmul
        struct ggml_tensor * up = ggml_mul_mat_id(ctx, w_up, cur, ids);
        up->data = buf_up.data();

        // Routed activation with DeepSeek41 SwiGLU clamp reproduction
        struct ggml_tensor * act = nullptr;
        if (geom.clamp > 1e-6f) {
            act = ggml_swiglu_clamp(ctx, gate, up, geom.clamp);
        } else {
            act = ggml_swiglu_split(ctx, gate, up);
        }
        act->data = buf_act.data();

        // Down projection matmul
        struct ggml_tensor * down = ggml_mul_mat_id(ctx, w_down, act, ids);
        down->data = dummy_out_buf.data();

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(gf, down);

        ggml_backend_graph_plan_t plan = ggml_backend_graph_plan_create(backend_cpu, gf);
        GGML_ASSERT(plan != nullptr);

        plan_entry entry;
        entry.ctx = ctx;
        entry.gf = gf;
        entry.plan = plan;
        entry.w_gate = w_gate;
        entry.w_up = w_up;
        entry.w_down = w_down;
        entry.cur = cur;
        entry.ids = ids;
        entry.down = down;

        geom_plans[geom_id][n_jobs] = entry;
    }


    void compute(int layer, int n_jobs, const int32_t * slots, const int32_t * tokens, const float * input, float * output) {
        if (n_jobs <= 0) {
            return;
        }

        GGML_ASSERT(layer >= 0 && layer < (int) layers.size());
        GGML_ASSERT(n_jobs <= max_tokens);
        GGML_ASSERT(slots != nullptr);
        GGML_ASSERT(tokens != nullptr);
        GGML_ASSERT(input != nullptr);
        GGML_ASSERT(output != nullptr);

        const auto & l = layers[layer];
        const int64_t n_embd = l.n_embd;

        for (int j = 0; j < n_jobs; ++j) {
            GGML_ASSERT(tokens[j] >= 0 && tokens[j] < max_tokens);
            GGML_ASSERT(slots[j] >= 0 && slots[j] < l.n_slots);
        }

        // 1. Gather input rows for selected tokens
        float * gathered = (float *) gathered_input_buf.data();
        for (int j = 0; j < n_jobs; ++j) {
            const int32_t tok = tokens[j];
            memcpy(gathered + (size_t) j * n_embd,
                   input + (size_t) tok * n_embd,
                   (size_t) n_embd * sizeof(float));
        }

        // 2. Lookup plan for this geometry and job count
        const int gid = layer_geom[layer];
        plan_entry & p = geom_plans[gid][n_jobs];

        // 3. Bind layer weight pointers and caller buffers
        p.w_gate->data = const_cast<void *>(l.weights[0]);
        p.w_up->data   = const_cast<void *>(l.weights[1]);
        p.w_down->data = const_cast<void *>(l.weights[2]);
        p.cur->data    = gathered;
        p.ids->data    = const_cast<void *>(static_cast<const void *>(slots));
        p.down->data   = output;

        const enum ggml_status status = ggml_backend_graph_plan_compute(backend_cpu, p.plan);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
};

moe_cpu_executor::moe_cpu_executor(ggml_backend_dev_t cpu_device, const std::vector<moe_cpu_layer> & layers, int n_threads, int max_tokens)
    : pimpl(std::make_unique<impl>(cpu_device, layers, n_threads, max_tokens)) {
}

moe_cpu_executor::~moe_cpu_executor() = default;

void moe_cpu_executor::compute(int layer, int n_jobs, const int32_t * slots, const int32_t * tokens, const float * input, float * output) {
    pimpl->compute(layer, n_jobs, slots, tokens, input, output);
}
