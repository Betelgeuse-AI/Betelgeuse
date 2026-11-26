# Betelgeuse v0.7.0 (Alpha-Orionis)

**Native Inkling 2 Mini support, built for memory-constrained hardware.**

Betelgeuse is a custom [llama.cpp](https://github.com/ggml-org/llama.cpp) fork. This release adds a
native CUDA engine for Thinking Machines Lab's Inkling 2 Mini that executes the original mixed
NVFP4, MXFP8 and BF16 checkpoint directly. No GGUF conversion, extra weight quantisation, expert
skipping or speculative decoding is used in the reported benchmarks.

Developed and optimised on a single RTX 3060 12 GB with 16 GB of single-channel DDR4 system memory.

The unified server and CLI executables are **`bel-server`** and **`bel-cli`**. The llama C API and
internal CMake target names are unchanged. The standalone **`bel-inkling2`** executable is retained
for fixed-token benchmarking and reference work.

> **Research preview.** Betelgeuse is not an upstream llama.cpp release. The native Inkling engine is
> validated on Linux x86-64 with NVIDIA Ampere SM86. See [Current limitations](#current-limitations)
> and [OPTIMISATION-MASTER.md](OPTIMISATION-MASTER.md) before using it in production.

## Performance

Test hardware: RTX 3060 12 GB, Intel Core i5-12400, 16 GB single-channel DDR4, Linux, CUDA SM86,
original Inkling 2 Mini checkpoint, greedy decoding, 512 generated tokens per short workload through
`bel-server`.

| Workload | Generation |
|---|---:|
| LRU cache with TTL | 44.51 tok/s |
| Directed cycle detection | 45.52 tok/s |
| Race-condition reasoning | 47.11 tok/s |
| Token-bucket rate limiter | 44.33 tok/s |
| Serializability reasoning | 45.27 tok/s |
| Median | 45.27 tok/s |
| Mean | 45.35 tok/s |
| Coherent 16K source review | 42.03 tok/s |

| Prompt processing | Performance |
|---|---:|
| Cold uncached 16K prefill | 818.9 tok/s |
| Warm uncached 16K prefill | 1,232.8 tok/s |
| Warm 16K phase transition | 203.4 ms |
| Initial GPU-fetch preparation | Approximately 11 seconds |

Warm uncached prefill processes the complete prompt again; it does not reuse previously calculated
prompt KV states. Historical long-context results from earlier development builds (16K/32K/64K,
128K and 262K) are recorded separately in [OPTIMISATION-MASTER.md](OPTIMISATION-MASTER.md) and are
**not** measurements of this release build.

## Build

Linux x86-64 with CMake 3.24+, a C++20 compiler and CUDA 12 or 13. SM86 is the validated target.
ccache is recommended. UI assets are embedded only when present; disable the bundled UI for a
minimal server build.

```sh
git clone https://github.com/Betelgeuse-AI/Betelgeuse.git
cd Betelgeuse
git checkout v0.7.0-alpha-orionis

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

The build produces `build/bin/bel-server`, `build/bin/bel-cli` and `build/bin/bel-inkling2`.
See [docs/betelgeuse.md](docs/betelgeuse.md) for requirements, environment variables, packaging
notes and the release validation commands.

## Run

Point `-m` at the directory containing the Inkling 2 Mini safetensors shards and tokenizer:

```sh
build/bin/bel-server -m <checkpoint-dir> -c 16896 -t 5 \
  --host 127.0.0.1 --port 8080 --no-webui

build/bin/bel-cli -m <checkpoint-dir> -c 1536 -t 5 \
  -p "Hello, my name is" -n 8 --no-display-prompt -st
```

`bel-server` selects the embedded native engine automatically for a safetensors checkpoint
directory and forces the tuned engine profile unless `INK2_ALLOW_TUNING_OVERRIDE` is set. GPU fetch
is enabled by default in the server. With `INK2_ALLOW_TUNING_OVERRIDE=1`, `INK2_GPU_FETCH=0` and
`INK2_PIN_HOST_MIB=N` are honoured; without the override they are ignored so experiment settings
cannot silently degrade a normal request.

## What's new

- Direct execution of the original NVFP4, MXFP8 and BF16 checkpoint with packed expert weights.
- Ampere low-bit GEMV kernels, packed FP16 tensor-core prefill and optimised BF16 projections.
- GPU-side greedy sampling, full recurrent execution and original expert selection.
- Architecture-aware residency: recurrent-core experts stay resident, nonresident experts use a
  bounded GPU fetch cache with decayed-LFU eviction and overlapped PCIe transfers.
- Dual-phase inference: prefill workspace is reclaimed and reused for expert residency and decode
  inside the same process, without disturbing the model, KV cache or recurrent states.
- Fused grouped-query attention, sliding-window attention, dynamic KV growth, memory-aware pool
  resizing, CUDA graph reconstruction and K8/V4 KV-cache support.
- Unified `bel-server`/`bel-cli` interface with streaming and repeated requests.

## Current limitations

- The first request can incur a one-time GPU-fetch arena preparation cost of roughly 11 seconds.
- Performance depends on expert locality, VRAM and system-memory bandwidth.
- Long-context generation is slower than short-context generation.
- Validation is concentrated on Linux and NVIDIA Ampere hardware.
- Preserved checkpoint weights can still produce numerical differences across execution and
  KV-cache precisions.
- Experimental MLA/CISA, activation quantisation, speculative decoding and unfinished research
  features are not included.

## Acknowledgements and licensing

Betelgeuse builds on the [llama.cpp](https://github.com/ggml-org/llama.cpp) ecosystem and native
Inkling engine work contributed to Strata by baptisterajaut. Thanks to the developers of llama.cpp,
Strata, Thinking Machines Lab and the wider local inference community.

The repository keeps the upstream llama.cpp MIT licence and AUTHORS file. The Inkling engine carries
`tools/inkling2/LICENSE.Strata` (MIT, copyright Niko1221 and the Strata contributors) and
`tools/inkling2/CONTRIBUTING.md`. Packaged binaries include the required CUDA toolkit licence
notices. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the full release notes.

Repository: https://github.com/Betelgeuse-AI/Betelgeuse
