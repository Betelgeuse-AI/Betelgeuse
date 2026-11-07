#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct moe_cpu_layer {
    ggml_type types[3]; // gate, up, down
    const void * weights[3]; // borrowed host slot 0 role pointers
    size_t expert_stride; // full compact gate+up+down slot bytes
    int64_t n_embd, n_ff;
    int32_t n_slots;
    float clamp; // reproduce DeepSeek41 routed SwiGLU clamp
};

class moe_cpu_executor {
public:
    moe_cpu_executor(ggml_backend_dev_t cpu_device, const std::vector<moe_cpu_layer> & layers, int n_threads, int max_tokens);
    ~moe_cpu_executor();

    moe_cpu_executor(const moe_cpu_executor &) = delete;
    moe_cpu_executor & operator=(const moe_cpu_executor &) = delete;

    void compute(int layer, int n_jobs, const int32_t * slots, const int32_t * tokens, const float * input, float * output);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
