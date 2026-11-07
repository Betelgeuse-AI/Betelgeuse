#pragma once

#include "ggml.h"

#include <cstdint>

inline int ggml_cuda_moe_zero_expert(const ggml_tensor * ids) {
    for (const ggml_tensor * t = ids; t != nullptr; t = t->view_src) {
        if (t->op == GGML_OP_MOE_SLOT_RESOLVE_KICK) {
            if (ggml_get_op_params_i32(t, 2) != 0) {
                const int32_t sentinel = ggml_get_op_params_i32(t, 1);
                return sentinel >= 0 ? sentinel : -1;
            }
            return -1;
        }
    }
    return -1;
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, int zero_expert, cudaStream_t stream);
