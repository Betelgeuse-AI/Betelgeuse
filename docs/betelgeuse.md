# Betelgeuse build and release guide

This document covers the native Inkling 2 Mini path in Betelgeuse v0.7.0 (Alpha-Orionis). It is a
fork-specific supplement to the upstream [build instructions](build.md).

## Requirements

Validated release build:

- Linux x86-64
- CMake 3.24 or newer
- C++20 compiler (GCC 16.1.1 used for the release build)
- CUDA 13.3 / NVIDIA driver 610.43.02 used for the release build; CUDA 12 is expected to work
- NVIDIA RTX 3060 12 GB, compute capability SM86
- 16 GB system memory
- ccache (optional, recommended)
- Original Inkling 2 Mini safetensors checkpoint directory; no model weights are bundled

The binary archive is built for SM86 only. Rebuild from source for other GPUs or distributions.
System CUDA/driver and liburing are not bundled.

## Configure and build

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_INKLING2_ENGINE=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache

cmake --build build --target bel-server bel-cli llama-ink2 -j
```

Targets:

- `bel-server` - unified HTTP server with the embedded Inkling engine.
- `bel-cli` - CLI that starts `llama_server()` in a local thread for safetensors checkpoints.
- `llama-ink2` - `bel-inkling2` standalone fixed-token benchmark/reference runtime.

`LLAMA_BUILD_INKLING2_ENGINE=ON` requires `GGML_CUDA=ON`. Set it to `OFF` for an upstream-only
llama.cpp build.

## Running

```sh
build/bin/bel-server -m <checkpoint-dir> -c 16896 -t 5 \
  --host 127.0.0.1 --port 8080 --no-webui

build/bin/bel-cli -m <checkpoint-dir> -c 1536 -t 5 \
  -p "Hello, my name is" -n 8 --no-display-prompt -st
```

The server automatically selects the embedded engine for a checkpoint directory containing the
Inkling safetensors shards. Direct mode allows one active text request. Greedy sampling, support for
`ignore_eos`, `cache_prompt`, streaming and K8/V4 KV (`-ctk q8_0 -ctv q4_0`) are enabled. Grammar,
logit bias and speculative decoding are not supported in direct mode.

## Engine environment

The server forces a tuned profile to keep unrelated experiment settings out of ordinary requests:
`STRATA_NUMA_POOL=0`, `INK2_CPU_ACT=int8`, `INK2_CPU_OPT=29`, `INK2_OFF_MARGIN_MIB=64`,
`INK2_PIN_HOST_MIB=512` and `INK2_GPU_FETCH=1`. It also clears stale DSpark variables and selects a
route profile next to the checkpoint (`inkling2-routes-best512.freq` for a 16K prompt plus 512
output, `inkling2-routes-256k-decode.freq` otherwise).

Set `INK2_ALLOW_TUNING_OVERRIDE` (any value) to allow these variables to be overridden from the
environment. In particular:

- Without the override, `INK2_GPU_FETCH=0` and `INK2_PIN_HOST_MIB=N` are ignored.
- With `INK2_ALLOW_TUNING_OVERRIDE=1 INK2_GPU_FETCH=0`, the engine uses the CPU expert path and the
  bounded prefill host pin pool. The pool size can be changed with `INK2_PIN_HOST_MIB`.

The first request with GPU fetch enabled prepares a pinned host arena for evictable outer experts.
This takes about 11 seconds on the reference system and happens once per process. The prefill host
pin pool is a separate allocation and is not released by the GPU-fetch arena release path.

## Release validation

Verified on the release candidate:

- Cold request: 22 prompt tokens, 16 generated tokens, GPU-fetch arena prepared, 41.4 tok/s.
- Repeated identical request: completes normally with a 30 ms phase transition.
- Warm continuation with conversation history: completes normally.
- Streaming request: SSE chunks followed by `data: [DONE]`.
- GPU-fetch default: `INK2_GPU_FETCH=0` without override is ignored and GPU fetch still activates.
- GPU-fetch override: `INK2_ALLOW_TUNING_OVERRIDE=1 INK2_GPU_FETCH=0` keeps the host pin pool alive;
  a 4816-token prompt grows KV from 4096 to 8192 cells, recaptures the decode graph and completes.
- CUDA graph recapture reported by every prefill-to-decode transition; no changed-size reallocations
  in the short-context smoke tests.

## Package layout

The Linux binary archive uses the same structure as the previous release:

```
betelgeuse-v0.7.0-alpha-orionis-linux-x86_64-sm86/
  LICENSE
  README.txt
  run
  bin/            # bel-server, bel-cli, bel-inkling2 and required shared libraries
  docs/           # this guide and build notes
  licenses/       # upstream, Strata and CUDA notices
  validation/     # release smoke results
  SHA256SUMS
```

Run binaries through the launcher so the bundled libraries are used:

```sh
./run bel-cli --help
./run bel-server --help
./run bel-inkling2 --help
```

Verify the archive with `sha256sum -c SHA256SUMS`.

## Integration provenance

The release branch `betelgeuse-v6.1` was fast-forwarded to the verified native consolidation tip
`db1f05d1b` from `feat/v61-cupti-opt`. The validated native-lowbit work (packed FP16 tensor-core
prefill, fused/bf16 expert GEMM paths, Ampere low-bit GEMV kernels) is already present in that tip.
The remaining lowbit-branch changes are either experimental activation quantisation or superseded
schedules and are not included. Experimental MLA/CISA, speculative decoding and unfinished research
features are excluded.
