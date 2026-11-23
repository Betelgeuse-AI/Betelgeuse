# Inkling-2-Mini implementation notes

This file records source provenance and design boundaries for review and maintenance. Product documentation and command output use Betelgeuse names; these notes retain the implementation history.

## Source and license

The direct-checkpoint CUDA engine in `tools/inkling2` is based on the Inkling-2-Mini implementation from Strata PR [#6520](https://github.com/Niko1221/Strata/pull/6520), branch `baptisterajaut/Strata:feat/inkling2`, source commit `d3431b17c3b132ec85d6575aec9a8da0148b971d`. The copied and adapted upstream files are:

- `ink2_engine.cu`, `ink2_cpu.cpp`, `ink2_cpu.hpp`
- `ink2_grow_guard.cuh`, `ink2_image.cpp`, `ink2_image.hpp`
- `ink2_iq_tables.h`, `ink2_session.cuh`, `ink2_vision.cuh`, `ink2ref.cu`

`safetensors_lite.hpp` is a smaller Betelgeuse reader adapted from the PR's safetensors-loading path. The source changes made for the direct engine are recorded in `benchmarks/strata-source/source-changes.patch`. Preserve the upstream MIT notice in [LICENSE.Strata](LICENSE.Strata) when copying or distributing this code.

The IQ3_S and IQ4_NL tables in `ink2_iq_tables.h` are from llama.cpp's `ggml/src/ggml-common.h` and retain the llama.cpp repository license.

## Legacy GGUF integration

The GGUF path is implemented in `src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp`, and the surrounding `src/llama-*` architecture, graph, vocabulary, model, and KV-cache changes. It registers the model, expands the three-pass loop into independent cache/state slots, and shares physical weights across those passes. It is retained for legacy GGUF compatibility and is not the path used by the official safetensors checkpoint.

The direct safetensors engine is linked into `bel-server` when `LLAMA_BUILD_INKLING2_ENGINE` is enabled. Passing the official checkpoint directory as `-m` selects that backend; `bel-cli` uses it through its local server. The backend uses the checkpoint's packed NVFP4/MXFP8 experts and K8/V4 KV without conversion to GGUF. The standalone `bel-inkling2` executable is kept for fixed-token benchmarking and reference work.

The Marlin bridge in `src/llama-inkling-marlin.*` and `ggml/src/ggml-cuda/custom-dispatch.h` is the legacy GGUF packed-expert path. It dispatches W4A16 NVFP4 experts in physical layers 0-19 and W8A16 MXFP8 experts in layers 20-23, with bounded GPU placement, pinned host placement, and route-driven fills. It is enabled with `LLAMA_INKLING_MARLIN` and is limited to Linux CUDA builds on SM86.

The CUDA flash-attention changes in `ggml/src/ggml-cuda/fattn*.cuh`, `fattn.cu`, and `ggml-cuda.cu`, together with the relative-bias graph inputs in `src/llama-graph.*`, keep relative-bias and mask metadata compact and apply it inside attention tiles. This avoids materializing full key-by-query bias and mask arrays. Preserve the existing attention scale, sinks, causal/local masks, and recurrent loop state when changing this path.

## Direct-engine performance work

The direct engine retains the source checkpoint's expert layouts. Betelgeuse changes are listed in the source patch and benchmark records. The main kernel and memory changes include:

- expose packed weights and scales as separate views so CPU workers can consume the shared packed-expert bank without copying it into a second layout;
- bound prompt staging and release temporary prompt mappings before rebuilding expert residency;
- restore the configured MXFP8 pool before filling the remaining GPU budget with NVFP4 experts;
- fuse grouped-query decode attention so each K/V row is loaded once for its three query heads;
- keep prefill scratch elastic and cap sliding-window attention scratch at long contexts.

Keep standalone-engine, direct-server, and legacy GGUF benchmark results separate. The standalone engine's corrected RTX 3060 measurements are 50.6 pp/s / 39.7 tok/s for a short prompt, 775.8 pp/s / 35.6 tok/s at 16K with K8/V4, and 468.1 pp/s / 16.3 tok/s at 256K with K8/V4. Record server results with the exact request length and checkpoint path.

## Review notes

The loop shares weight tensors by physical layer, but each logical pass needs its own KV and convolution state. The packed host bank must cover all experts outside protected GPU slots. Any change to either invariant can produce valid-looking but incorrect output or turn a bounded placement into an out-of-memory allocation.

The Marlin bridge is opt-in and has CUDA SM86-specific kernels. It must not be enabled in builds that do not meet its documented backend and architecture requirements. The direct backend currently supports one active text request, K8/V4 KV, and a limited sampler set; keep those limits explicit in server behavior and documentation.
