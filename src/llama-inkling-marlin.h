#pragma once

#include <memory>

struct ggml_context;
struct ggml_tensor;

class llama_inkling_marlin {
public:
    llama_inkling_marlin(const char * directory, float beta, float linear_beta);
    ~llama_inkling_marlin();

    bool has_layer(int physical_layer) const;
    ggml_tensor * build(ggml_context * ctx, int physical_layer, ggml_tensor * input, ggml_tensor * ids,
                       ggml_tensor * weights);

private:
    struct impl;
    std::unique_ptr<impl> state;
};
