#include "llama-inkling-marlin.h"
#include "llama-inkling-marlin-cuda.h"
#include "../ggml/src/ggml-cuda/custom-dispatch.h"

#include "ggml.h"
#include "llama-impl.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int hidden = 2304;
constexpr int intermediate = 1152;
constexpr int n_expert = inkling_experts;
constexpr int top_k = inkling_top_k;
constexpr int block_size = inkling_block_size;
constexpr int n_layers = inkling_layers;
constexpr int n_fp4_layers = 20;
constexpr int threads = 128;
constexpr int blocks = top_k * (hidden / 128);
constexpr int shared_bytes = 65536;

void cuda_check(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char * message = nullptr;
        cuGetErrorString(result, &message);
        throw std::runtime_error(message ? message : "unknown CUDA driver error");
    }
}

void runtime_check(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

struct context_guard {
    explicit context_guard(CUcontext ctx) {
        cuda_check(cuCtxPushCurrent(ctx));
    }
    ~context_guard() {
        CUcontext previous;
        cuCtxPopCurrent(&previous);
    }
};

struct device_buffer {
    CUdeviceptr data = 0;

    device_buffer() = default;
    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;

    ~device_buffer() {
        if (data) {
            cuMemFree(data);
        }
    }

    void allocate(size_t size) {
        cuda_check(cuMemAlloc(&data, size));
    }
};

struct projection {
    std::ifstream pack;
    const int k;
    const size_t weight_bytes;
    const size_t scale_bytes;
    const size_t stride;

    projection(const std::filesystem::path & path, int il, int k, int bits, int group_size) :
        pack(path, std::ios::binary), k(k), weight_bytes(size_t(hidden) * k * bits / 8),
        scale_bytes(size_t(hidden) * k / group_size), stride(weight_bytes + scale_bytes + sizeof(float)) {
        std::array<char, 8> magic;
        std::array<uint32_t, 4> dims;
        pack.read(magic.data(), magic.size());
        pack.read(reinterpret_cast<char *>(dims.data()), sizeof(dims));
        if (!pack || std::memcmp(magic.data(), "INKMAR1", magic.size()) != 0 ||
                dims != std::array<uint32_t, 4>{uint32_t(il), uint32_t(k), hidden, n_expert} ||
                std::filesystem::file_size(path) != 24 + n_expert * stride) {
            throw std::runtime_error("invalid Inkling Marlin pack: " + path.string());
        }
    }

    void read(int expert, char * destination) {
        pack.seekg(24 + size_t(expert) * stride);
        if (!pack.read(destination, stride)) {
            throw std::runtime_error("cannot read packed Inkling expert");
        }
    }
};

struct projection_bank {
    const int k;
    const size_t weight_bytes;
    const size_t scale_bytes;
    device_buffer weights, scales, globals;

    projection_bank(int k, int bits, int group) :
        k(k), weight_bytes(size_t(hidden) * k * bits / 8), scale_bytes(size_t(hidden) * k / group) {}

    size_t stride() const {
        return weight_bytes + scale_bytes + sizeof(float);
    }

    size_t host_stride() const {
        return (stride() + 15) & ~size_t(15);
    }

    void allocate(size_t slots) {
        weights.allocate(slots * weight_bytes);
        scales.allocate(slots * scale_bytes);
        globals.allocate(slots * sizeof(float));
    }

    inkling_marlin_copy_dst copy_dst() const {
        return {reinterpret_cast<uint32_t *>(weights.data), reinterpret_cast<uint32_t *>(scales.data),
                reinterpret_cast<uint32_t *>(globals.data), uint32_t(weight_bytes / sizeof(uint32_t)),
                uint32_t(scale_bytes / sizeof(uint32_t))};
    }

    void upload(int slot, const char * source, CUstream stream) {
        cuda_check(cuMemcpyHtoDAsync(weights.data + slot * weight_bytes, source, weight_bytes, stream));
        cuda_check(cuMemcpyHtoDAsync(scales.data + slot * scale_bytes, source + weight_bytes, scale_bytes, stream));
        cuda_check(cuMemcpyHtoDAsync(globals.data + slot * sizeof(float), source + weight_bytes + scale_bytes, sizeof(float), stream));
    }
};

struct expert_bank {
    const int bits;
    const int group;
    projection_bank gate_up, down;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    char * staging = nullptr;
    char * pinned = nullptr;
    CUdeviceptr mapped_staging = 0;
    CUdeviceptr mapped_pinned = 0;
    std::vector<int> pinned_keys;
    std::vector<uint64_t> pinned_ages;
    std::array<int, n_layers * n_expert> pinned_slots;
    size_t protected_host_slots = 0;
    std::vector<int> residents;
    std::vector<uint64_t> ages;
    std::array<int, n_layers * n_expert> slots;
    device_buffer slot_table, gpu_ages, gpu_clock;
    size_t protected_slots = 0;
    uint64_t clock = 0;

    explicit expert_bank(bool nvfp4) :
        bits(nvfp4 ? 4 : 8), group(nvfp4 ? 16 : 32),
        gate_up(hidden, bits, group), down(intermediate, bits, group) {
        slots.fill(-1);
        pinned_slots.fill(-1);
    }

    ~expert_bank() {
        if (staging) {
            cuMemFreeHost(staging);
        }
        if (pinned) {
            cuMemFreeHost(pinned);
        }
        if (module) {
            cuModuleUnload(module);
        }
    }

    size_t record_size() const {
        return gate_up.stride() + down.stride();
    }

    size_t host_record_size() const {
        return gate_up.host_stride() + down.host_stride();
    }

    void load(const std::filesystem::path & dir, size_t gpu_slots, size_t host_slots) {
        const std::string variant = bits == 4 ? "fe2m1f" : "fe4m3fn";
        std::ifstream symbol_file(dir / (variant + ".symbol"));
        std::array<char, 512> symbol = {};
        if (!symbol_file.getline(symbol.data(), symbol.size()) || symbol[0] == '\0') {
            throw std::runtime_error("missing Marlin kernel symbol");
        }
        cuda_check(cuModuleLoad(&module, (dir / (variant + ".cubin")).c_str()));
        cuda_check(cuModuleGetFunction(&kernel, module, symbol.data()));
        cuda_check(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes));

        gate_up.allocate(gpu_slots);
        down.allocate(gpu_slots);
        residents.assign(gpu_slots, -1);
        ages.assign(gpu_slots, 0);
        cuda_check(cuMemHostAlloc(reinterpret_cast<void **>(&staging), top_k * host_record_size(), 0));
        allocate_pinned(host_slots);
    }

    void allocate_pinned(size_t host_slots) {
        if (host_slots) {
            cuda_check(cuMemHostAlloc(reinterpret_cast<void **>(&pinned), host_slots * host_record_size(), 0));
            pinned_keys.assign(host_slots, -1);
            pinned_ages.assign(host_slots, 0);
        }
    }

    bool gpu_protected(int key) const {
        return slots[key] >= 0 && size_t(slots[key]) < protected_slots;
    }

    void assign_pinned(size_t slot, int key) {
        const int old = pinned_keys[slot];
        if (old >= 0) {
            pinned_slots[old] = -1;
        }
        pinned_keys[slot] = key;
        pinned_slots[key] = slot;
    }

    CUdeviceptr mapped_source(const char * source) const {
        const auto addr = reinterpret_cast<uintptr_t>(source);
        const auto base = reinterpret_cast<uintptr_t>(staging);
        if (addr >= base && addr - base < top_k * host_record_size()) {
            return mapped_staging + (addr - base);
        }
        return mapped_pinned + (addr - reinterpret_cast<uintptr_t>(pinned));
    }

    void assign(int slot, int key) {
        if (residents[slot] >= 0) {
            slots[residents[slot]] = -1;
        }
        residents[slot] = key;
        slots[key] = slot;
    }

    void upload(int slot, const char * source, CUstream stream) {
        gate_up.upload(slot, source, stream);
        down.upload(slot, source + gate_up.host_stride(), stream);
    }
};

// Marlin's route rows keep the original top-k index even when dispatched in two groups.
using route_batch = inkling_marlin_route;

route_batch make_routes(const std::array<int, top_k> & selection, unsigned mask) {
    route_batch routes{};
    std::fill_n(routes.sorted, top_k * block_size, top_k);
    routes.mask = mask;
    for (int i = 0; i < top_k; ++i) {
        if (mask & (1u << i)) {
            routes.slots[routes.padded / block_size] = selection[i];
            routes.sorted[routes.padded] = i;
            routes.padded += block_size;
        }
    }
    return routes;
}

} // namespace

struct llama_inkling_marlin::impl {
    struct layer {
        impl & owner;
        expert_bank & bank;
        const int physical_layer;
        projection gate_up, down;
        std::array<bool, n_expert> seen_upload{};
        uint64_t pinned_hits = 0;
        uint64_t pinned_fills = 0;
        uint64_t pinned_evictions = 0;
        uint64_t staged_uploads = 0;
        uint64_t host_read_bytes = 0;
        uint64_t hits = 0;
        uint64_t uploads = 0;
        ggml_cuda_custom_dispatch dispatch{};
        uint64_t visits = 0;
        uint64_t mixed_visits = 0;
        uint64_t timing_samples = 0;
        double shared_ms = 0;
        uint64_t dma_samples = 0;
        double dma_ms = 0;
        double exposed_ms = 0;
        double route_sync_ms = 0;
        double stage_ms = 0;
        double read_ms = 0;

        layer(impl & owner, expert_bank & bank, const std::filesystem::path & stem, int il) :
            owner(owner), bank(bank), physical_layer(il),
            gate_up(stem.string() + "-w13.marlin", il, hidden, bank.bits, bank.group),
            down(stem.string() + "-w2.marlin", il, intermediate, bank.bits, bank.group) {
            dispatch = {impl::eval, this, owner.gpu_resolve && owner.gpu_graph};
        }

        int key(int expert) const {
            return physical_layer * n_expert + expert;
        }

        void read(int expert, char * destination) {
            gate_up.read(expert, destination);
            down.read(expert, destination + bank.gate_up.host_stride());
        }

        std::array<int, top_k> plan(const int32_t * ids, unsigned & misses, int32_t * evicted = nullptr) {
            std::array<int, top_k> slots;
            std::array<int, top_k> required;
            misses = 0;
            for (int i = 0; i < top_k; ++i) {
                if (ids[i] < 0 || ids[i] >= n_expert || std::find(ids, ids + i, ids[i]) != ids + i) {
                    throw std::runtime_error("invalid Inkling expert selection");
                }
                required[i] = key(ids[i]);
            }
            for (int i = 0; i < top_k; ++i) {
                if (evicted) {
                    evicted[i] = -1;
                }
                int slot = bank.slots[required[i]];
                if (slot == -1) {
                    for (size_t j = bank.protected_slots; j < bank.residents.size(); ++j) {
                        if (std::find(required.begin(), required.end(), bank.residents[j]) == required.end() &&
                                (slot == -1 || bank.ages[j] < bank.ages[slot])) {
                            slot = j;
                        }
                    }
                    if (slot < 0) {
                        throw std::runtime_error("no free Inkling Marlin slot");
                    }
                    misses |= 1u << i;
                    if (evicted) {
                        evicted[i] = bank.residents[slot];
                    }
                    bank.assign(slot, required[i]);
                    uploads++;
                } else {
                    hits++;
                }
                bank.ages[slot] = ++bank.clock;
                slots[i] = slot;
            }
            return slots;
        }

        std::array<const char *, top_k> prepare_uploads(const int32_t * ids, unsigned misses) {
            std::array<const char *, top_k> sources{};
            char * next = bank.staging;
            std::array<int, top_k> required;
            for (int i = 0; i < top_k; ++i) {
                required[i] = key(ids[i]);
            }
            for (int i = 0; i < top_k; ++i) {
                if (!(misses & (1u << i))) {
                    continue;
                }
                const int expert = ids[i];
                const int row = bank.pinned_slots[required[i]];
                if (row >= 0) {
                    bank.pinned_ages[row] = bank.clock;
                    sources[i] = bank.pinned + row * bank.host_record_size();
                    pinned_hits++;
                    continue;
                }

                // Admit on a repeated GPU miss. Never replace a source named by this visit.
                size_t victim = bank.pinned_keys.size();
                if (seen_upload[expert]) {
                    for (size_t row = bank.protected_host_slots; row < bank.pinned_keys.size(); ++row) {
                        if (std::find(required.begin(), required.end(), bank.pinned_keys[row]) == required.end() &&
                                (victim == bank.pinned_keys.size() || bank.pinned_ages[row] < bank.pinned_ages[victim])) {
                            victim = row;
                        }
                    }
                }
                seen_upload[expert] = true;
                char * destination;
                if (victim < bank.pinned_keys.size()) {
                    destination = bank.pinned + victim * bank.host_record_size();
                    pinned_evictions += bank.pinned_keys[victim] >= 0;
                    bank.assign_pinned(victim, required[i]);
                    bank.pinned_ages[victim] = bank.clock;
                    pinned_fills++;
                } else {
                    destination = next;
                    next += bank.host_record_size();
                    staged_uploads++;
                }

                using host_clock = std::chrono::steady_clock;
                const auto start = owner.profile ? host_clock::now() : host_clock::time_point{};
                read(expert, destination);
                if (owner.profile) {
                    read_ms += std::chrono::duration<double, std::milli>(host_clock::now() - start).count();
                }
                host_read_bytes += bank.record_size();
                sources[i] = destination;
            }
            return sources;
        }

        void upload(const std::array<int, top_k> & slots, const std::array<const char *, top_k> & sources, CUstream stream) {
            for (int i = 0; i < top_k; ++i) {
                if (sources[i]) {
                    bank.upload(slots[i], sources[i], stream);
                }
            }
        }

        void execute(ggml_tensor * dst, CUstream stream) {
            std::lock_guard<std::mutex> lock(owner.mutex);
            context_guard guard(owner.context);

            if (owner.last_stream && owner.last_stream != stream) {
                cuda_check(cuStreamWaitEvent(stream, owner.done, 0));
            }
            for (int64_t token = 0; token < dst->ne[1]; ++token) {
                if (owner.gpu_resolve) {
                    execute_gpu_token(dst, token, stream);
                } else {
                    execute_token(dst, token, stream);
                }
            }
        }

        void execute_gpu_token(ggml_tensor * dst, int64_t token, CUstream stream) {
            constexpr unsigned all = (1u << top_k) - 1;
            runtime_check(llama_inkling_marlin_route(static_cast<const int32_t *>(dst->src[1]->data) + token * top_k,
                physical_layer, reinterpret_cast<int32_t *>(bank.slot_table.data),
                reinterpret_cast<uint64_t *>(bank.gpu_ages.data), bank.residents.size(),
                reinterpret_cast<uint64_t *>(bank.gpu_clock.data), owner.mailbox_device,
                reinterpret_cast<route_batch *>(owner.routes_gpu.data),
                reinterpret_cast<inkling_marlin_stats *>(owner.gpu_stats.data),
                reinterpret_cast<inkling_marlin_visit *>(owner.gpu_trace.data), owner.trace_capacity, stream));
            runtime_check(llama_inkling_marlin_copy_misses(owner.mailbox_device,
                reinterpret_cast<const route_batch *>(owner.routes_gpu.data),
                bank.gate_up.copy_dst(), bank.down.copy_dst(), stream));
            runtime_check(llama_inkling_marlin_cast(static_cast<const float *>(dst->src[0]->data) + token * hidden,
                                                   reinterpret_cast<void *>(owner.input.data), hidden, stream,
                                                   reinterpret_cast<const inkling_marlin_stats *>(owner.gpu_stats.data),
                                                   reinterpret_cast<inkling_marlin_visit *>(owner.gpu_trace.data)));
            run_group_device(owner.routes_gpu.data, all, blocks, stream);
            runtime_check(llama_inkling_marlin_reduce(reinterpret_cast<const void *>(owner.output.data),
                          static_cast<const float *>(dst->src[2]->data) + token * top_k,
                          static_cast<float *>(dst->data) + token * hidden, top_k, hidden, stream));
            if (!owner.gpu_graph || physical_layer == n_layers - 1) {
                // A captured graph records completion after its last Marlin layer.
                cuda_check(cuEventRecord(owner.done, stream));
            }
            owner.last_stream = stream;
        }

        void execute_token(ggml_tensor * dst, int64_t token, CUstream stream) {
            using host_clock = std::chrono::steady_clock;
            const auto route_start = owner.profile ? host_clock::now() : host_clock::time_point{};
            cuda_check(cuMemcpyDtoHAsync(owner.ids_host, reinterpret_cast<CUdeviceptr>(dst->src[1]->data) + token * top_k * sizeof(int32_t),
                                         top_k * sizeof(int32_t), stream));
            // Expert residency is still reconciled on the host.
            cuda_check(cuStreamSynchronize(stream));
            owner.collect_transfer();
            const auto stage_start = owner.profile ? host_clock::now() : host_clock::time_point{};
            const auto * ids = static_cast<const int32_t *>(owner.ids_host);
            unsigned misses;
            const auto slots = plan(ids, misses);
            const double reads_before = read_ms;
            constexpr unsigned all = (1u << top_k) - 1;
            const unsigned hits_mask = all ^ misses;
            const bool split = owner.overlap && hits_mask && misses;
            if (misses) {
                const auto sources = prepare_uploads(ids, misses);
                const auto transfer_stream = split ? owner.copy_stream : stream;
                if (owner.profile) {
                    cuda_check(cuEventRecord(owner.copy_begin, transfer_stream));
                    owner.timing_layer = this;
                    owner.timing_split = split;
                }
                upload(slots, sources, transfer_stream);
                if (split || owner.profile) {
                    cuda_check(cuEventRecord(owner.copied, transfer_stream));
                }
            }
            visits++;
            if (owner.profile) {
                route_sync_ms += std::chrono::duration<double, std::milli>(stage_start - route_start).count();
                const double stage = std::chrono::duration<double, std::milli>(host_clock::now() - stage_start).count();
                stage_ms += stage;
                if (owner.visit_log.is_open() && dst->ne[1] == 1) {
                    owner.visit = {physical_layer, bank.record_size() * __builtin_popcount(misses),
                        std::chrono::duration<double, std::milli>(stage_start - route_start).count(),
                        read_ms - reads_before, stage, 0.0, 0.0};
                }
            }

            runtime_check(llama_inkling_marlin_cast(static_cast<const float *>(dst->src[0]->data) + token * hidden,
                                                   reinterpret_cast<void *>(owner.input.data), hidden, stream));
            if (split) {
                mixed_visits++;
                if (owner.profile) {
                    cuda_check(cuEventRecord(owner.resident_begin, stream));
                }
                run_group(slots, hits_mask, 0, stream);
                if (owner.profile) {
                    cuda_check(cuEventRecord(owner.resident_end, stream));
                }
                cuda_check(cuStreamWaitEvent(stream, owner.copied, 0));
                if (owner.profile) {
                    cuda_check(cuEventRecord(owner.copy_wait_end, stream));
                }
                run_group(slots, misses, 1, stream);
            } else {
                run_group(slots, all, 0, stream);
            }
            runtime_check(llama_inkling_marlin_reduce(reinterpret_cast<const void *>(owner.output.data),
                          static_cast<const float *>(dst->src[2]->data) + token * top_k,
                          static_cast<float *>(dst->data) + token * hidden, top_k, hidden, stream));
            cuda_check(cuEventRecord(owner.done, stream));
            owner.last_stream = stream;
        }
        void run_group(const std::array<int, top_k> & slots, unsigned mask, int index, CUstream stream) {
            const auto routes = make_routes(slots, mask);
            auto * routes_host = owner.routes_host + index * sizeof(route_batch);
            std::memcpy(routes_host, &routes, sizeof(routes));
            const auto routes_gpu = owner.routes_gpu.data + index * sizeof(route_batch);
            cuda_check(cuMemcpyHtoDAsync(routes_gpu, routes_host, sizeof(routes), stream));
            const int grid = (routes.padded / block_size) * (hidden / 128);
            run_group_device(routes_gpu, mask, grid, stream);
        }

        void run_group_device(CUdeviceptr routes_gpu, unsigned mask, int grid, CUstream stream) {
            launch(bank.gate_up, owner.input.data, owner.gate_out.data, top_k, 1, routes_gpu, grid, stream);
            runtime_check(llama_inkling_marlin_situ(reinterpret_cast<const void *>(owner.gate_out.data),
                          reinterpret_cast<void *>(owner.inner.data), intermediate, top_k, mask,
                          owner.beta, owner.linear_beta, stream));
            launch(bank.down, owner.inner.data, owner.output.data, 1, top_k, routes_gpu, grid, stream);
        }

        void launch(projection_bank & p, CUdeviceptr input, CUdeviceptr output, int top, int m,
                    CUdeviceptr routes, int grid, CUstream stream) {
            CUdeviceptr null = 0;
            CUdeviceptr sorted_ids = routes + offsetof(route_batch, sorted);
            CUdeviceptr expert_ids = routes + offsetof(route_batch, slots);
            CUdeviceptr padded = routes + offsetof(route_batch, padded);
            int n = hidden;
            int k = p.k;
            bool no = false;
            bool fp32_reduce = true;
            void * args[] = {
                &input, &p.weights.data, &output, &owner.scratch.data,
                &null, &p.scales.data, &p.globals.data, &null, &sorted_ids,
                &expert_ids, &padded, &owner.routing.data, &top, &no, &m, &n, &k,
                &owner.locks.data, &no, &no, &fp32_reduce,
            };
            cuda_check(cuLaunchKernel(bank.kernel, grid, 1, 1, threads, 1, 1, shared_bytes, stream, args, nullptr));
        }
    };

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream last_stream = nullptr;
    CUevent done = nullptr;
    CUstream copy_stream = nullptr;
    CUevent copied = nullptr;
    CUevent copy_begin = nullptr;
    CUevent resident_begin = nullptr;
    CUevent resident_end = nullptr;
    CUevent copy_wait_end = nullptr;
    layer * timing_layer = nullptr;
    const bool gpu_resolve = [] {
        const char * value = std::getenv("LLAMA_INKLING_MARLIN_GPU_RESOLVE");
        return value && std::strcmp(value, "1") == 0;
    }();
    const bool gpu_graph = [] {
        const char * value = std::getenv("LLAMA_INKLING_MARLIN_GPU_GRAPH");
        return value && std::strcmp(value, "1") == 0;
    }();
    const char * trace_path = std::getenv("LLAMA_INKLING_MARLIN_GPU_TRACE");
    static constexpr uint32_t trace_capacity = 8192;
    inkling_marlin_mailbox * mailbox = nullptr;
    inkling_marlin_mailbox * mailbox_device = nullptr;
    device_buffer gpu_stats, gpu_trace;
    std::atomic<bool> worker_running{false};
    std::thread worker;
    bool timing_split = false;
    std::mutex mutex;
    device_buffer input, gate_out, inner, output, scratch, routes_gpu, locks, routing;
    char * routes_host = nullptr;
    const bool overlap = [] {
        const char * value = std::getenv("LLAMA_INKLING_MARLIN_OVERLAP");
        return value && std::strcmp(value, "1") == 0;
    }();
    void * ids_host = nullptr;
    const size_t pinned_slots = [] {
        const char * value = std::getenv("LLAMA_INKLING_MARLIN_PINNED_SLOTS");
        if (!value) {
            return size_t(16);
        }
        char * end = nullptr;
        const long slots = std::strtol(value, &end, 10);
        if (end == value || *end || slots < 0 || slots > 32) {
            throw std::runtime_error("Inkling Marlin pinned slots must be between 0 and 32");
        }
        return size_t(slots);
    }();
    const bool profile = std::getenv("LLAMA_INKLING_MARLIN_PROFILE") != nullptr;
    struct visit_sample {
        int layer = -1;
        size_t upload_bytes;
        double route_ms, read_ms, stage_ms, dma_ms, exposed_ms;
    } visit;
    std::ofstream visit_log;
    const float beta;
    const float linear_beta;

    impl(float beta, float linear_beta) : beta(beta), linear_beta(linear_beta) {}
    std::array<std::unique_ptr<expert_bank>, 2> banks;
    std::array<std::unique_ptr<layer>, n_layers> layers;

    size_t host_budget() const {
        const char * value = std::getenv("LLAMA_INKLING_MARLIN_HOST_MIB");
        if (!value) {
            return 0;
        }
        char * end = nullptr;
        const long mib = std::strtol(value, &end, 10);
        if (end == value || *end || mib < 1 || mib > 8192) {
            throw std::runtime_error("Inkling Marlin host budget must be between 1 and 8192 MiB");
        }
        return size_t(mib) * 1024 * 1024;
    }

    void preload_host_experts(const std::filesystem::path & placement) {
        std::ifstream ranked(placement);
        if (!ranked) {
            throw std::runtime_error("cannot open Inkling Marlin host placement");
        }
        int il, expert;
        while (ranked >> il) {
            if (!(ranked >> expert) || il < 0 || il >= n_layers || !layers[il] || expert < 0 || expert >= n_expert) {
                throw std::runtime_error("invalid Inkling Marlin host placement entry");
            }
            auto & layer = *layers[il];
            auto & bank = layer.bank;
            const int key = layer.key(expert);
            if (bank.protected_host_slots >= bank.pinned_keys.size() * 3 / 4 || bank.pinned_slots[key] >= 0 ||
                    bank.gpu_protected(key)) {
                continue;
            }
            const size_t slot = bank.protected_host_slots++;
            layer.read(expert, bank.pinned + slot * bank.host_record_size());
            bank.assign_pinned(slot, key);
        }
        if (!ranked.eof()) {
            throw std::runtime_error("malformed Inkling Marlin host placement file");
        }
    }

    void cover_host_experts(size_t byte_budget, const std::array<size_t, 2> & counts) {
        size_t bytes = 0;
        for (size_t i = 0; i < banks.size(); ++i) {
            if (banks[i]) {
                bytes += (counts[i] * n_expert - banks[i]->protected_slots) * banks[i]->host_record_size();
            }
        }
        if (bytes > byte_budget) {
            throw std::runtime_error("Inkling Marlin host budget cannot cover the experts outside the protected GPU slots");
        }
        for (size_t i = 0; i < banks.size(); ++i) {
            if (banks[i]) {
                banks[i]->allocate_pinned(counts[i] * n_expert - banks[i]->protected_slots);
            }
        }
        for (auto & layer : layers) {
            if (!layer) {
                continue;
            }
            auto & bank = layer->bank;
            for (int expert = 0; expert < n_expert; ++expert) {
                const int key = layer->key(expert);
                if (!bank.gpu_protected(key)) {
                    const size_t slot = bank.protected_host_slots++;
                    layer->read(expert, bank.pinned + slot * bank.host_record_size());
                    bank.assign_pinned(slot, key);
                }
            }
        }
        LLAMA_LOG_INFO("Inkling Marlin: covered all experts outside the protected GPU slots (%zu pinned bytes)\n", bytes);
    }

    void load_banks(const std::filesystem::path & dir) {
        std::array<size_t, 2> counts{};
        for (int il = 0; il < n_layers; ++il) {
            const auto stem = dir / ("layer-" + std::to_string(il));
            const bool gate_up = std::filesystem::exists(stem.string() + "-w13.marlin");
            const bool down = std::filesystem::exists(stem.string() + "-w2.marlin");
            if (gate_up != down) {
                throw std::runtime_error("incomplete Marlin pack for Inkling layer " + std::to_string(il));
            }
            if (gate_up) {
                const size_t index = il >= n_fp4_layers;
                if (!banks[index]) {
                    banks[index] = std::make_unique<expert_bank>(!index);
                }
                layers[il] = std::make_unique<layer>(*this, *banks[index], stem, il);
                counts[index]++;
            }
        }
        if (!counts[0] && !counts[1]) {
            throw std::runtime_error("no Inkling Marlin packs in " + dir.string());
        }

        size_t budget = 6144;
        if (const char * value = std::getenv("LLAMA_INKLING_MARLIN_GPU_MIB")) {
            char * end = nullptr;
            const long mib = std::strtol(value, &end, 10);
            if (end == value || *end || mib < 128 || mib > 16384) {
                throw std::runtime_error("Inkling Marlin GPU budget must be between 128 and 16384 MiB");
            }
            budget = mib;
        }
        size_t total = 0;
        size_t host_total = 0;
        const size_t host_bytes = host_budget();
        const char * cover = std::getenv("LLAMA_INKLING_MARLIN_HOST_COVER");
        const bool cover_host = cover && std::strcmp(cover, "1") == 0;
        for (size_t i = 0; i < banks.size(); ++i) {
            if (banks[i]) {
                total += counts[i] * n_expert * banks[i]->record_size();
                host_total += counts[i] * n_expert * banks[i]->host_record_size();
            }
        }
        const bool full = counts[0] + counts[1] == n_layers;
        budget *= 1024 * 1024;
        size_t reserve = 0;
        for (const auto & bank : banks) {
            if (bank) {
                reserve += top_k * bank->record_size();
            }
        }
        std::array<size_t, 2> split_bytes{};
        if (const char * value = std::getenv("LLAMA_INKLING_MARLIN_W8_MIB")) {
            char * end = nullptr;
            const long mib = std::strtol(value, &end, 10);
            if (end == value || *end || !banks[0] || !banks[1] || mib < 1 || mib > 16384) {
                throw std::runtime_error("invalid Inkling Marlin W8 bank budget");
            }
            const size_t bytes = size_t(mib) * 1024 * 1024;
            if (bytes < top_k * banks[1]->record_size() || bytes > budget - top_k * banks[0]->record_size()) {
                throw std::runtime_error("Inkling Marlin bank split must reserve eight slots in both banks");
            }
            split_bytes = {budget - bytes, bytes};
        }
        size_t protected_percent = 75;
        if (const char * value = std::getenv("LLAMA_INKLING_MARLIN_PROTECTED_PERCENT")) {
            char * end = nullptr;
            const long percent = std::strtol(value, &end, 10);
            if (end == value || *end || percent < 0 || percent > 100) {
                throw std::runtime_error("Inkling Marlin protected percentage must be between 0 and 100");
            }
            protected_percent = percent;
        }
        for (size_t i = 0; i < banks.size(); ++i) {
            if (auto * bank = banks[i].get()) {
                const size_t experts = counts[i] * n_expert;
                size_t slots = std::min(experts, split_bytes[i] ? split_bytes[i] / bank->record_size() :
                    top_k + (budget - reserve) * experts / total);
                if (!full && !std::getenv("LLAMA_INKLING_MARLIN_GPU_MIB")) {
                    slots = std::min(slots, counts[i] * 12);
                }
                const size_t host_slots = host_bytes ? std::min(experts, host_bytes * experts / host_total) : pinned_slots;
                bank->load(dir, slots, cover_host ? 0 : host_slots);
            }
        }

        const auto placement = dir / "placement.txt";
        if (std::filesystem::exists(placement)) {
            std::ifstream ranked(placement);
            int il, expert;
            while (ranked >> il >> expert) {
                if (il < 0 || il >= n_layers || !layers[il] || expert < 0 || expert >= n_expert) {
                    throw std::runtime_error("invalid Inkling Marlin placement entry");
                }
                auto & layer = *layers[il];
                auto & bank = layer.bank;
                const size_t cap = std::min(bank.residents.size() * protected_percent / 100, bank.residents.size() - top_k);
                const int key = layer.key(expert);
                if (bank.protected_slots < cap && bank.slots[key] < 0) {
                    const int slot = bank.protected_slots++;
                    layer.read(expert, bank.staging);
                    bank.upload(slot, bank.staging, nullptr);
                    cuda_check(cuStreamSynchronize(nullptr));
                    bank.assign(slot, key);
                }
            }
            if (!ranked.eof()) {
                throw std::runtime_error("malformed Inkling Marlin placement file");
            }
        }
        const auto host_placement = dir / "host-placement.txt";
        if (cover_host) {
            cover_host_experts(host_bytes, counts);
        } else if (host_bytes && std::filesystem::exists(host_placement)) {
            preload_host_experts(host_placement);
        }
        for (const auto & bank : banks) {
            if (bank) {
                LLAMA_LOG_INFO("Inkling Marlin W%d: %zu GPU slots (%zu bytes), %zu protected, %zu demand, %zu host slots (%zu protected) and %d staging slots (%zu bytes pinned)\n",
                    bank->bits, bank->residents.size(), bank->residents.size() * bank->record_size(), bank->protected_slots,
                    bank->residents.size() - bank->protected_slots, bank->pinned_keys.size(), bank->protected_host_slots, top_k,
                    (bank->pinned_keys.size() + top_k) * bank->host_record_size());
            }
        }
        LLAMA_LOG_INFO("%s: enabled native Marlin for %zu Inkling physical layers\n", __func__, counts[0] + counts[1]);
    }


    void service_gpu_misses() {
        try {
            while (worker_running.load(std::memory_order_relaxed)) {
                if (__atomic_load_n(&mailbox->state, __ATOMIC_ACQUIRE) != 1) {
                    std::this_thread::yield();
                    continue;
                }
                auto & layer = *layers.at(mailbox->layer);
                auto & bank = layer.bank;
                bank.clock = mailbox->clock;
                std::copy_n(mailbox->ages, bank.ages.size(), bank.ages.begin());
                unsigned misses;
                const auto slots = layer.plan(mailbox->ids, misses, mailbox->evicted);
                const auto sources = layer.prepare_uploads(mailbox->ids, misses);
                for (int i = 0; i < top_k; ++i) {
                    if (sources[i]) {
                        mailbox->source[i] = bank.mapped_source(sources[i]);
                    }
                }
                std::copy(slots.begin(), slots.end(), mailbox->slots);
                // The stream copies these sources before it can publish another miss.
                __atomic_store_n(&mailbox->state, 2, __ATOMIC_RELEASE);
            }
        } catch (const std::exception & e) {
            GGML_ABORT("Inkling Marlin GPU miss service: %s", e.what());
        }
    }

    void init_gpu_resolve() {
        cuda_check(cuMemHostAlloc(reinterpret_cast<void **>(&mailbox), sizeof(*mailbox), CU_MEMHOSTALLOC_DEVICEMAP));
        *mailbox = {};
        CUdeviceptr mapped;
        cuda_check(cuMemHostGetDevicePointer(&mapped, mailbox, 0));
        mailbox_device = reinterpret_cast<inkling_marlin_mailbox *>(mapped);
        gpu_stats.allocate(sizeof(inkling_marlin_stats));
        cuda_check(cuMemsetD8(gpu_stats.data, 0, sizeof(inkling_marlin_stats)));
        if (trace_path) {
            gpu_trace.allocate(trace_capacity * sizeof(inkling_marlin_visit));
        }
        for (auto & bank : banks) {
            if (bank) {
                cuda_check(cuMemHostGetDevicePointer(&bank->mapped_staging, bank->staging, 0));
                if (bank->pinned) {
                    cuda_check(cuMemHostGetDevicePointer(&bank->mapped_pinned, bank->pinned, 0));
                }
                bank->slot_table.allocate(sizeof(bank->slots));
                cuda_check(cuMemcpyHtoD(bank->slot_table.data, bank->slots.data(), sizeof(bank->slots)));
                const size_t age_bytes = bank->ages.size() * sizeof(uint64_t);
                bank->gpu_ages.allocate(age_bytes);
                bank->gpu_clock.allocate(sizeof(uint64_t));
                cuda_check(cuMemsetD8(bank->gpu_ages.data, 0, age_bytes));
                cuda_check(cuMemsetD8(bank->gpu_clock.data, 0, sizeof(uint64_t)));
            }
        }
        worker_running.store(true, std::memory_order_relaxed);
        worker = std::thread([this] { service_gpu_misses(); });
        LLAMA_LOG_INFO("Inkling Marlin: GPU route resolution with miss-only host service\n");
    }

    void log_gpu_resolve() {
        inkling_marlin_stats stats{};
        if (cuMemcpyDtoH(&stats, gpu_stats.data, sizeof(stats)) != CUDA_SUCCESS) {
            return;
        }
        uint64_t hits = 0, misses = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il]) {
                layers[il]->hits += stats.resident[il] * top_k;
                layers[il]->visits = stats.resident[il] + stats.misses[il];
                hits += stats.resident[il];
                misses += stats.misses[il];
            }
        }
        LLAMA_LOG_INFO("Inkling Marlin GPU routing: %llu resident visits, %llu host-service visits\n",
                (unsigned long long) hits, (unsigned long long) misses);
        if (gpu_trace.data && stats.trace_size && stats.trace_size <= trace_capacity) {
            std::vector<inkling_marlin_visit> visits(stats.trace_size);
            if (cuMemcpyDtoH(visits.data(), gpu_trace.data, visits.size() * sizeof(visits[0])) == CUDA_SUCCESS) {
                std::ofstream trace(trace_path);
                trace << "layer,start_ns,host_wait_start_ns,host_wait_end_ns,end_ns,compute_start_ns\n";
                for (const auto & visit : visits) {
                    trace << visit.layer << ',' << visit.start << ',' << visit.wait_start << ','
                          << visit.wait_end << ',' << visit.end << ',' << visit.compute_start << '\n';
                }
            }
        }
    }

    void collect_transfer() {
        collect_dma();
        if (visit.layer >= 0) {
            visit_log << visit.layer << ',' << visit.upload_bytes << ',' << visit.route_ms << ','
                      << visit.read_ms << ',' << visit.stage_ms << ',' << visit.dma_ms << ',' << visit.exposed_ms << '\n';
            visit.layer = -1;
        }
    }

    void collect_dma() {
        if (!timing_layer) {
            return;
        }
        float dma_ms;
        if (cuEventElapsedTime(&dma_ms, copy_begin, copied) == CUDA_SUCCESS) {
            float exposed = dma_ms;
            if (timing_split) {
                float wait_ms;
                if (cuEventElapsedTime(&wait_ms, resident_end, copy_wait_end) == CUDA_SUCCESS) {
                    exposed = wait_ms;
                }
                float gpu_start_ms, gpu_end_ms;
                if (cuEventElapsedTime(&gpu_start_ms, copy_begin, resident_begin) == CUDA_SUCCESS &&
                        cuEventElapsedTime(&gpu_end_ms, copy_begin, resident_end) == CUDA_SUCCESS) {
                    timing_layer->shared_ms += std::max(0.0f, std::min(dma_ms, gpu_end_ms) - std::max(0.0f, gpu_start_ms));
                    timing_layer->timing_samples++;
                }
            }
            timing_layer->dma_ms += dma_ms;
            timing_layer->exposed_ms += exposed;
            timing_layer->dma_samples++;
            if (visit.layer >= 0) {
                visit.dma_ms = dma_ms;
                visit.exposed_ms = exposed;
            }
        }
        timing_layer = nullptr;
    }

    void load(const std::filesystem::path & dir) {
        if (const char * path = std::getenv("LLAMA_INKLING_MARLIN_VISIT_LOG")) {
            if (!profile || gpu_resolve) {
                throw std::runtime_error("Inkling Marlin visit logging requires host routing with profiling enabled");
            }
            visit_log.open(path);
            if (!visit_log) {
                throw std::runtime_error("cannot open Inkling Marlin visit log");
            }
            visit_log << "layer,upload_bytes,route_ms,read_ms,stage_ms,dma_ms,exposed_ms\n";
        }
        cuda_check(cuInit(0));
        cuda_check(cuDeviceGet(&device, 0));
        int major, minor, count;
        cuda_check(cuDeviceGetCount(&count));
        cuda_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        cuda_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        if (count != 1 || major != 8 || minor != 6) {
            throw std::runtime_error("Inkling Marlin prototype requires one SM86 GPU");
        }
        cuda_check(cuDevicePrimaryCtxRetain(&context, device));
        context_guard guard(context);
        cuda_check(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING));
        if (overlap && !gpu_resolve) {
            cuda_check(cuStreamCreate(&copy_stream, CU_STREAM_NON_BLOCKING));
        }
        if (!gpu_resolve && (overlap || profile)) {
            cuda_check(cuEventCreate(&copied, profile ? CU_EVENT_DEFAULT : CU_EVENT_DISABLE_TIMING));
        }
        if (profile && !gpu_resolve) {
            cuda_check(cuEventCreate(&copy_begin, CU_EVENT_DEFAULT));
            if (overlap) {
                cuda_check(cuEventCreate(&resident_begin, CU_EVENT_DEFAULT));
                cuda_check(cuEventCreate(&resident_end, CU_EVENT_DEFAULT));
                cuda_check(cuEventCreate(&copy_wait_end, CU_EVENT_DEFAULT));
            }
        }
        input.allocate(hidden * sizeof(ggml_bf16_t));
        gate_out.allocate(top_k * hidden * sizeof(ggml_bf16_t));
        inner.allocate(top_k * intermediate * sizeof(ggml_bf16_t));
        output.allocate(top_k * hidden * sizeof(ggml_bf16_t));
        scratch.allocate(top_k * hidden * sizeof(float));
        routes_gpu.allocate(2 * sizeof(route_batch));
        cuda_check(cuMemHostAlloc(reinterpret_cast<void **>(&routes_host), 2 * sizeof(route_batch), 0));
        locks.allocate(blocks * sizeof(int32_t));
        routing.allocate(top_k * sizeof(float));
        cuda_check(cuMemHostAlloc(&ids_host, top_k * sizeof(int32_t), 0));
        std::array<float, top_k> ones;
        ones.fill(1.0f);
        cuda_check(cuMemcpyHtoD(routing.data, ones.data(), sizeof(ones)));
        cuda_check(cuMemsetD32(locks.data, 0, blocks));
        load_banks(dir);
        if (gpu_resolve) {
            init_gpu_resolve();
        }
    }

    ~impl() {
        if (!context || cuCtxPushCurrent(context) != CUDA_SUCCESS) {
            return;
        }
        if (last_stream) {
            cuEventSynchronize(done);
        }
        worker_running.store(false, std::memory_order_relaxed);
        if (worker.joinable()) {
            worker.join();
            log_gpu_resolve();
        }
        if (copy_stream) {
            cuStreamSynchronize(copy_stream);
        }
        collect_transfer();
        for (int il = 0; il < int(layers.size()); ++il) {
            if (layers[il]) {
                LLAMA_LOG_INFO("Inkling Marlin layer %d: %llu hits, %llu uploads, %llu token visits\n", il,
                    (unsigned long long) layers[il]->hits, (unsigned long long) layers[il]->uploads,
                    (unsigned long long) layers[il]->visits);
                const auto & layer = *layers[il];
                LLAMA_LOG_INFO("Inkling Marlin layer %d: %llu direct hits, %llu fills, %llu pinned evictions, %llu staged uploads, %llu host read bytes\n",
                    il, (unsigned long long) layer.pinned_hits,
                    (unsigned long long) layer.pinned_fills, (unsigned long long) layer.pinned_evictions,
                    (unsigned long long) layer.staged_uploads, (unsigned long long) layer.host_read_bytes);
                if (layer.dma_samples) {
                    LLAMA_LOG_INFO("Inkling Marlin layer %d: %.3f ms/miss visit DMA, %.3f ms/miss visit exposed GPU wait (%llu samples)\n",
                        il, layer.dma_ms / layer.dma_samples, layer.exposed_ms / layer.dma_samples, (unsigned long long) layer.dma_samples);
                }
                if (profile && layer.visits) {
                    const double visits = layers[il]->visits;
                    LLAMA_LOG_INFO("Inkling Marlin layer %d: %.3f ms/visit route wait, %.3f ms/visit RAM staging (%.3f ms buffered reads), %llu split visits\n",
                        il, layers[il]->route_sync_ms / visits, layers[il]->stage_ms / visits, layers[il]->read_ms / visits,
                        (unsigned long long) layers[il]->mixed_visits);
                }
                if (layers[il]->timing_samples) {
                    LLAMA_LOG_INFO("Inkling Marlin layer %d: %.3f ms/split visit copy and compute share GPU time (%llu samples)\n",
                        il, layers[il]->shared_ms / layers[il]->timing_samples,
                        (unsigned long long) layers[il]->timing_samples);
                }
                layers[il].reset();
            }
        }
        for (auto & bank : banks) {
            bank.reset();
        }
        for (auto * buffer : {&input, &gate_out, &inner, &output, &scratch, &routes_gpu, &locks, &routing, &gpu_stats, &gpu_trace}) {
            if (buffer->data) {
                cuMemFree(buffer->data);
                buffer->data = 0;
            }
        }
        if (mailbox) {
            cuMemFreeHost(mailbox);
        }
        if (ids_host) {
            cuMemFreeHost(ids_host);
        }
        if (routes_host) {
            cuMemFreeHost(routes_host);
        }
        if (done) {
            cuEventDestroy(done);
        }
        for (const auto event : {copied, copy_begin, resident_begin, resident_end, copy_wait_end}) {
            if (event) {
                cuEventDestroy(event);
            }
        }
        if (copy_stream) {
            cuStreamDestroy(copy_stream);
        }
        CUcontext previous;
        cuCtxPopCurrent(&previous);
        cuDevicePrimaryCtxRelease(device);
    }

    static void cpu_fallback(ggml_tensor *, int, int, void *) {
        GGML_ABORT("Inkling Marlin requires a CUDA backend");
    }

    static void eval(ggml_tensor * dst, void * stream, void * user_data) {
        try {
            static_cast<layer *>(user_data)->execute(dst, static_cast<CUstream>(stream));
        } catch (const std::exception & e) {
            GGML_ABORT("Inkling Marlin: %s", e.what());
        }
    }
};

llama_inkling_marlin::llama_inkling_marlin(const char * directory, float beta, float linear_beta) :
    state(std::make_unique<impl>(beta, linear_beta)) {
    state->load(directory);
}

llama_inkling_marlin::~llama_inkling_marlin() = default;

bool llama_inkling_marlin::has_layer(int il) const {
    return il >= 0 && il < int(state->layers.size()) && state->layers[il] != nullptr;
}

ggml_tensor * llama_inkling_marlin::build(ggml_context * ctx, int il, ggml_tensor * input, ggml_tensor * ids,
                                       ggml_tensor * weights) {
    const int64_t tokens = input->ne[1];
    if (!has_layer(il) || input->type != GGML_TYPE_F32 || input->ne[0] != hidden ||
            ggml_nelements(input) != hidden * tokens || ids->type != GGML_TYPE_I32 ||
            ggml_nelements(ids) != top_k * tokens || weights->type != GGML_TYPE_F32 ||
            ggml_nelements(weights) != top_k * tokens) {
        throw std::runtime_error("invalid Inkling Marlin graph inputs");
    }
    auto & layer = *state->layers[il];
    ggml_tensor * sources[] = {ggml_cont(ctx, input), ggml_cont(ctx, ids), ggml_cont(ctx, weights)};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, hidden, tokens, 1, 1, sources, 3, impl::cpu_fallback, 1, nullptr);
    ggml_cuda_custom_dispatch::set(result, &layer.dispatch);
    return result;
}
