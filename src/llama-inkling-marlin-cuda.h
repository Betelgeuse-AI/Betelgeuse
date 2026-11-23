#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

struct inkling_marlin_stats;
struct inkling_marlin_visit;

cudaError_t llama_inkling_marlin_cast(const float * input, void * output, int n, cudaStream_t stream,
    const inkling_marlin_stats * stats = nullptr, inkling_marlin_visit * trace = nullptr);
cudaError_t llama_inkling_marlin_situ(const void * input, void * output, int cols, int rows, unsigned mask, float beta, float linear_beta, cudaStream_t stream);
cudaError_t llama_inkling_marlin_reduce(const void * input, const float * weights, float * output, int rows, int cols, cudaStream_t stream);

constexpr int inkling_layers = 24;
constexpr int inkling_experts = 128;
constexpr int inkling_top_k = 8;
constexpr int inkling_block_size = 8;

struct alignas(16) inkling_marlin_route {
    int32_t sorted[inkling_top_k * inkling_block_size];
    int32_t slots[inkling_top_k];
    int32_t padded;
    unsigned mask;
    unsigned copy_mask;
};

struct inkling_marlin_mailbox {
    uint32_t state;
    int32_t layer;
    int32_t ids[inkling_top_k];
    int32_t slots[inkling_top_k];
    int32_t evicted[inkling_top_k];
    uint64_t source[inkling_top_k];
    uint64_t clock;
    uint64_t ages[inkling_layers * inkling_experts];
};

struct inkling_marlin_visit {
    uint64_t start;
    uint64_t wait_start;
    uint64_t wait_end;
    uint64_t end;
    uint64_t compute_start;
    int32_t layer;
};

struct inkling_marlin_stats {
    uint64_t resident[inkling_layers];
    uint64_t misses[inkling_layers];
    uint32_t trace_size;
    int32_t current_trace;
};

cudaError_t llama_inkling_marlin_route(const int32_t * ids, int layer, int32_t * table,
    uint64_t * ages, int n_slots, uint64_t * clock, inkling_marlin_mailbox * mailbox,
    inkling_marlin_route * route, inkling_marlin_stats * stats, inkling_marlin_visit * trace,
    uint32_t trace_capacity, cudaStream_t stream);


struct inkling_marlin_copy_dst {
    uint32_t * weights;
    uint32_t * scales;
    uint32_t * globals;
    uint32_t weight_words;
    uint32_t scale_words;
};

cudaError_t llama_inkling_marlin_copy_misses(const inkling_marlin_mailbox * mailbox,
    const inkling_marlin_route * route, inkling_marlin_copy_dst gate_up, inkling_marlin_copy_dst down, cudaStream_t stream);
