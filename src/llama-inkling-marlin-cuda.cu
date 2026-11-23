#include "llama-inkling-marlin-cuda.h"

#include <cuda_bf16.h>

namespace {

constexpr int threads = 256;

__device__ uint64_t route_time() {
    uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

__global__ void cast_bf16(const float * input, __nv_bfloat16 * output, int n,
        const inkling_marlin_stats * stats, inkling_marlin_visit * trace) {
    if (trace && blockIdx.x == 0 && threadIdx.x == 0 && stats->current_trace >= 0) {
        trace[stats->current_trace].compute_start = route_time();
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = __float2bfloat16_rn(input[i]);
    }
}

__global__ void situ(const __nv_bfloat16 * input, __nv_bfloat16 * output, int cols, unsigned mask, float beta, float linear_beta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (i < cols && (mask & (1u << row))) {
        const int index = row * cols + i;
        const float gate = __bfloat162float(input[2*index]);
        const float up = __bfloat162float(input[2*index + 1]);
        const float bounded_gate = beta * tanhf(gate / beta);
        const float bounded_up = linear_beta * tanhf(up / linear_beta);
        output[index] = __float2bfloat16_rn(bounded_gate * bounded_up / (1.0f + expf(-gate)));
    }
}

__global__ void reduce(const __nv_bfloat16 * input, const float * weights, float * output, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
            sum = __fadd_rn(sum, __fmul_rn(weights[row], __bfloat162float(input[row * cols + col])));
        }
        output[col] = sum;
    }
}

} // namespace

cudaError_t llama_inkling_marlin_cast(const float * input, void * output, int n, cudaStream_t stream,
        const inkling_marlin_stats * stats, inkling_marlin_visit * trace) {
    cast_bf16<<<(n + threads - 1) / threads, threads, 0, stream>>>(input, static_cast<__nv_bfloat16 *>(output), n, stats, trace);
    return cudaGetLastError();
}

cudaError_t llama_inkling_marlin_situ(const void * input, void * output, int cols, int rows, unsigned mask,
                                     float beta, float linear_beta, cudaStream_t stream) {
    const dim3 grid((cols + threads - 1) / threads, rows);
    situ<<<grid, threads, 0, stream>>>(static_cast<const __nv_bfloat16 *>(input),
        static_cast<__nv_bfloat16 *>(output), cols, mask, beta, linear_beta);
    return cudaGetLastError();
}

cudaError_t llama_inkling_marlin_reduce(const void * input, const float * weights, float * output, int rows, int cols, cudaStream_t stream) {
    reduce<<<(cols + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(input), weights, output, rows, cols);
    return cudaGetLastError();
}

namespace {

__global__ void marlin_resolve_routes(const int32_t * ids, int layer, int32_t * table, uint64_t * ages, int n_slots,
        uint64_t * clock, inkling_marlin_mailbox * mailbox, inkling_marlin_route * route,
        inkling_marlin_stats * stats, inkling_marlin_visit * trace, uint32_t trace_capacity) {
    const int lane = threadIdx.x;
    uint64_t start = 0, wait_start = 0, wait_end = 0;
    if (lane == 0 && trace) {
        start = route_time();
    }

    int expert = -1;
    int slot = -1;
    if (lane < inkling_top_k) {
        expert = ids[lane];
        if (expert < 0 || expert >= inkling_experts) {
            asm volatile("trap;");
        }
        for (int i = 0; i < lane; ++i) {
            if (ids[i] == expert) {
                asm volatile("trap;");
            }
        }
        slot = table[layer * inkling_experts + expert];
    }
    const unsigned miss = __ballot_sync(0xffffffff, lane < inkling_top_k && slot < 0);

    if (miss) {
        if (lane < inkling_top_k) {
            mailbox->ids[lane] = expert;
        }
        for (int i = lane; i < n_slots; i += warpSize) {
            mailbox->ages[i] = ages[i];
        }
        // Publish the IDs and LRU snapshot before ringing the host.
        __threadfence_system();
        __syncwarp();
        if (lane == 0) {
            mailbox->layer = layer;
            mailbox->clock = *clock;
            wait_start = route_time();
            asm volatile("st.release.sys.global.u32 [%0], 1;" :: "l"(&mailbox->state) : "memory");
            uint32_t ready;
            do {
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(ready) : "l"(&mailbox->state) : "memory");
                if (route_time() - wait_start > 30000000000ULL) {
                    asm volatile("trap;");
                }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                __nanosleep(250);
#endif
            } while (ready != 2);
            wait_end = route_time();
            mailbox->state = 0;
        }
        __syncwarp();
        if (lane < inkling_top_k) {
            const int evicted = mailbox->evicted[lane];
            if (evicted >= 0) {
                table[evicted] = -1;
            }
            slot = mailbox->slots[lane];
            table[layer * inkling_experts + expert] = slot;
        }
    }

    const uint64_t epoch = *clock;
    if (lane < inkling_top_k) {
        route->slots[lane] = slot;
        ages[slot] = epoch + lane + 1;
    }
    for (int i = lane; i < inkling_top_k * inkling_block_size; i += warpSize) {
        route->sorted[i] = i % inkling_block_size == 0 ? i / inkling_block_size : inkling_top_k;
    }
    __syncwarp();
    if (lane == 0) {
        *clock = epoch + inkling_top_k;
        route->padded = inkling_top_k * inkling_block_size;
        route->mask = (1u << inkling_top_k) - 1;
        route->copy_mask = miss;
        if (miss) {
            stats->misses[layer]++;
        } else {
            stats->resident[layer]++;
        }
        if (trace) {
            stats->current_trace = -1;
            if (stats->trace_size < trace_capacity) {
                stats->current_trace = stats->trace_size++;
                trace[stats->current_trace] = {start, wait_start, wait_end, route_time(), 0, layer};
            }
        }
    }
}

__global__ void marlin_copy_misses(const inkling_marlin_mailbox * mailbox, const inkling_marlin_route * route,
        inkling_marlin_copy_dst gate_up, inkling_marlin_copy_dst down) {
    const unsigned row = blockIdx.y;
    if (!(route->copy_mask & (1u << row))) {
        return;
    }
    const auto * source = reinterpret_cast<const uint32_t *>(mailbox->source[row]);
    const bool is_down = blockIdx.z != 0;
    const auto dst = is_down ? down : gate_up;
    if (is_down) {
        source += (gate_up.weight_words + gate_up.scale_words + 4) & ~3u;
    }

    const unsigned slot = route->slots[row];
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    auto * weights = reinterpret_cast<uint4 *>(dst.weights + size_t(slot) * dst.weight_words);
    auto * scales = reinterpret_cast<uint4 *>(dst.scales + size_t(slot) * dst.scale_words);
    const auto * weight_source = reinterpret_cast<const uint4 *>(source);
    const auto * scale_source = reinterpret_cast<const uint4 *>(source + dst.weight_words);
    for (unsigned i = lane; i < dst.weight_words / 4; i += stride) {
        weights[i] = weight_source[i];
    }
    for (unsigned i = lane; i < dst.scale_words / 4; i += stride) {
        scales[i] = scale_source[i];
    }
    if (lane == 0) {
        dst.globals[slot] = source[dst.weight_words + dst.scale_words];
    }
}

} // namespace

cudaError_t llama_inkling_marlin_route(const int32_t * ids, int layer, int32_t * table, uint64_t * ages,
        int n_slots, uint64_t * clock, inkling_marlin_mailbox * mailbox,
        inkling_marlin_route * route, inkling_marlin_stats * stats, inkling_marlin_visit * trace,
        uint32_t trace_capacity, cudaStream_t stream) {
    marlin_resolve_routes<<<1, 32, 0, stream>>>(ids, layer, table, ages, n_slots, clock, mailbox, route,
        stats, trace, trace_capacity);
    return cudaGetLastError();
}


cudaError_t llama_inkling_marlin_copy_misses(const inkling_marlin_mailbox * mailbox, const inkling_marlin_route * route,
        inkling_marlin_copy_dst gate_up, inkling_marlin_copy_dst down, cudaStream_t stream) {
    marlin_copy_misses<<<dim3(64, inkling_top_k, 2), 128, 0, stream>>>(mailbox, route, gate_up, down);
    return cudaGetLastError();
}
