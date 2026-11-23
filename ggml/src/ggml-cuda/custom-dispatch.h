#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstring>

struct ggml_cuda_custom_dispatch {
    using callback = void (*)(ggml_tensor * dst, void * stream, void * data);

    static constexpr uint64_t tag = 0x47474d4c43554441ULL;
    static constexpr size_t offset = 32;

    callback compute;
    void * data;
    bool graph_safe = false;

    static ggml_cuda_custom_dispatch get(const ggml_tensor * tensor) {
        if (tensor->op != GGML_OP_CUSTOM || !tensor->extra) {
            return {};
        }
        uint64_t magic;
        std::memcpy(&magic, reinterpret_cast<const char *>(tensor->op_params) + offset, sizeof(magic));
        return magic == tag ? *static_cast<const ggml_cuda_custom_dispatch *>(tensor->extra) : ggml_cuda_custom_dispatch{};
    }

    static void set(ggml_tensor * tensor, ggml_cuda_custom_dispatch * dispatch) {
        std::memcpy(reinterpret_cast<char *>(tensor->op_params) + offset, &tag, sizeof(tag));
        // Callback addresses are process-local and never serialized with op_params.
        tensor->extra = dispatch;
    }
};

static_assert(ggml_cuda_custom_dispatch::offset + sizeof(uint64_t) <= GGML_MAX_OP_PARAMS);
