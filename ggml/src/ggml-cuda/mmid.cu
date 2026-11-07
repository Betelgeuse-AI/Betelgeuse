#include "common.cuh"
#include "mmid.cuh"


// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, const int zero_expert) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    const int n_ids = n_tokens * n_expert_used;
    int nex_prev = 0;
    for (uint32_t i = threadIdx.x; i < (uint32_t) n_ids; i += warp_size) {
        const uint32_t it = i / n_expert_used;
        const uint32_t iex = i % n_expert_used;
        const int32_t id = ids[(size_t) it * si1 + iex];
        nex_prev += (zero_expert < 0 || id != zero_expert) && (id < expert);
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    int it_compact = 0;
    if (zero_expert < 0 || expert != zero_expert) {
        for (uint32_t first = 0; first < (uint32_t) n_ids; first += warp_size) {
            const uint32_t i = first + threadIdx.x;
            const uint32_t it = i / n_expert_used;
            const uint32_t iex = i % n_expert_used;
            const bool match = i < (uint32_t) n_ids && ids[(size_t) it * si1 + iex] == expert;
            int prefix = match;
#pragma unroll
            for (int offset = 1; offset < warp_size; offset *= 2) {
                const int previous = __shfl_up_sync(0xFFFFFFFF, prefix, offset, warp_size);
                if (threadIdx.x >= (unsigned) offset) prefix += previous;
            }
            if (match) {
                const int row = nex_prev + it_compact + prefix - 1;
                ids_dst[row] = i;
                if (write_inverse) {
                    ids_src1[i] = row;
                } else {
                    ids_src1[row] = it * sis1 + iex % nchannels_y;
                }
            }
            it_compact += __shfl_sync(0xFFFFFFFF, prefix, warp_size - 1, warp_size);
        }
    }
    if (threadIdx.x == 0) {
        expert_bounds[expert] = nex_prev;
        if (expert == (int) gridDim.x - 1) expert_bounds[gridDim.x] = nex_prev + it_compact;
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, const int zero_expert, cudaStream_t stream) {
    GGML_ASSERT(n_tokens > 0 && n_expert_used_var > 0 && n_tokens <= INT_MAX / n_expert_used_var);

    if (zero_expert >= 0) {
        const int n_ids = n_tokens * n_expert_used_var;
        CUDA_CHECK(cudaMemsetAsync(ids_src1, 0xFF, (size_t) n_ids * sizeof(int32_t), stream));
        CUDA_CHECK(cudaMemsetAsync(ids_dst,  0xFF, (size_t) n_ids * sizeof(int32_t), stream));
    }

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse, zero_expert);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, const int zero_expert, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, zero_expert, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, zero_expert, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, zero_expert, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, zero_expert, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, zero_expert, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, zero_expert, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, zero_expert, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, zero_expert, stream);
            break;
    }
}
