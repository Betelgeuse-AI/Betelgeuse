<h1 align="center">Betelgeuse <sub><i>Alpha-Orionis</i></sub></h1>

![Betelgeuse: a deep-red block star with an amber core](assets/betelgeuse.svg)

<p align="center">
  <b>Native inference for memory-constrained hardware.</b><br>
  A C++/CUDA fork of <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a>, developed and optimised on consumer GPUs.
</p>

<p align="center">
  <a href="#release"><img alt="Release" src="https://img.shields.io/badge/release-v0.7.0-b22222"></a>
  <a href="#model-support"><img alt="Model" src="https://img.shields.io/badge/native-Inkling%202%20Mini-b22222"></a>
  <a href="#build"><img alt="Target" src="https://img.shields.io/badge/target-sm__86%20%C2%B7%20RTX%203060-76b900"></a>
  <a href="#current-limitations"><img alt="Status" src="https://img.shields.io/badge/status-research%20preview-d4a017"></a>
  <a href="#licensing-and-acknowledgements"><img alt="Engine" src="https://img.shields.io/badge/engine-MIT-blue"></a>
</p>

---

## Release

**v0.7.0 · Alpha-Orionis**

Betelgeuse now includes a native CUDA inference engine for Thinking Machines Lab's Inkling 2 Mini. It executes the original mixed NVFP4, MXFP8 and BF16 checkpoint directly, without GGUF conversion or additional weight quantisation. All originally routed experts are evaluated. The reported benchmarks use neither expert skipping nor speculative decoding.

The engine is designed around limited VRAM and host memory rather than assuming the entire checkpoint can remain on the GPU. It combines architecture-aware expert residency, bounded GPU fetching and separate prefill and decode memory profiles within a single process.

Developed on an RTX 3060 12 GB, Intel Core i5-12400 and 16 GB of single-channel DDR4.

> [!IMPORTANT]
> **Compatibility break: previous model integrations are deprecated.** v0.7.0 introduces substantial changes to the inference runtime, memory management and execution paths. DeepSeek-V4.1-Flash (1017) and Alpha-Orionis-XS (Qwen3.8 Flash Next) are no longer supported targets for this release. Their existing bundles, launch flags and paging configurations may fail or behave differently with the new build. Use the earlier revision known to work with your model. Backward compatibility is not guaranteed; deprecation does not imply that every legacy code path has been removed.

> [!WARNING]
> **Research preview.** This is an independent fork, not an upstream llama.cpp release. Native Inkling execution has been validated on Linux x86-64 with NVIDIA Ampere SM86. Other configurations require independent validation.

## Contents

- [Model support](#model-support)
- [Performance](#performance)
- [Build](#build)
- [Run](#run)
- [Architecture](#architecture)
- [Legacy models](#legacy-models)
- [Current limitations](#current-limitations)
- [Star history](#star-history)
- [Licensing and acknowledgements](#licensing-and-acknowledgements)

## Model support

| Model | Checkpoint format | v0.7.0 status |
| --- | --- | --- |
| Inkling 2 Mini | Original NVFP4 / MXFP8 / BF16 safetensors | Validated native engine |
| DeepSeek-V4.1-Flash (1017) | Legacy GGUF / Engram / DSpark | Deprecated; not revalidated |
| Alpha-Orionis-XS (Qwen3.8 Flash Next) | Legacy four-file GSQ-RCO IQ2_XS bundle | Deprecated; not revalidated |
| Other llama.cpp models | GGUF | Compatibility not established by this release |

The unified executables are `bel-server` (HTTP server) and `bel-cli` (CLI). The standalone `bel-inkling2` executable is retained for fixed-token benchmarks and reference work. The llama C API is unchanged.

## Performance

All figures in this section are reported for the v0.7.0 release build using the original Inkling 2 Mini checkpoint on the following hardware:

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA RTX 3060 12 GB (SM86) |
| CPU | Intel Core i5-12400 |
| System memory | 16 GB single-channel DDR4 |
| Platform | Linux x86-64 / CUDA |
| Decode | Greedy, non-speculative |

### Token generation

Short workloads generate 512 tokens each through `bel-server`.

| Workload | Generation |
| --- | ---: |
| LRU cache with TTL | 44.51 tok/s |
| Directed cycle detection | 45.52 tok/s |
| Race-condition reasoning | 47.11 tok/s |
| Token-bucket rate limiter | 44.33 tok/s |
| Serializability reasoning | 45.27 tok/s |
| **Median** | **45.27 tok/s** |
| **Mean** | **45.35 tok/s** |
| Coherent 16K source review | 42.03 tok/s |

### Prompt processing

| Measurement | Performance |
| --- | ---: |
| Cold uncached 16K prefill | 818.9 tok/s |
| Warm uncached 16K prefill | **1,232.8 tok/s** |
| Warm 16K phase transition | 203.4 ms |
| Initial GPU-fetch preparation | Approximately 11 s |

Warm uncached prefill processes the entire prompt again. It does not reuse previously computed prompt KV states. Warm refers to the engine's prepared state, not a prompt-cache hit.

These are workload-specific measurements, not universal throughput guarantees. Earlier 16K, 32K, 64K, 128K and 262K development results are maintained separately in [OPTIMISATION-MASTER.md](OPTIMISATION-MASTER.md). They are not measurements or context-length guarantees for v0.7.0.

## Build

Requirements: Linux x86-64, CMake 3.24+, a C++20 compiler and CUDA 12 or 13. SM86 is the validated target. Install `ccache` if using the compiler launcher options below. The bundled UI is disabled for a minimal build.

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

Expected binaries:

```text
build/bin/
├── bel-server
├── bel-cli
└── bel-inkling2
```

See [docs/betelgeuse.md](docs/betelgeuse.md) for detailed requirements, environment variables, packaging and release validation commands.

### Windows standalone engine

On Windows x64, build `bel-inkling2` with Visual Studio 2022, CMake and the CUDA Toolkit. The commands below target SM86 and were tested with CUDA 12.6.

```powershell
cmake -S . -B build-win -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_INKLING2_ENGINE=ON
cmake --build build-win --config Release --target llama-ink2 --parallel

$env:Path = "$env:CUDA_PATH\bin;$env:Path"
$env:INK2_GPU_FETCH = "1"
$env:INK2_OFF_MARGIN_MIB = "256"
& .\build-win\bin\Release\bel-inkling2.exe --model "C:\models\Inkling-2-Mini" --serve --max-context 4096 --chunk 512 --kv k8v4 --vram-reserve-mib 0 --cpu-threads 5 --kv-grow 1 --spec 0
```

Set `--model` to the original checkpoint directory. CPU expert offload requires AVX2, FMA and F16C on Windows. The engine emits `READY` when loaded and reads token IDs over stdin. The integrated `bel-server`, `bel-cli` and bundled HTTP UI use the Linux build above.

## Run

Point `-m` at the original Inkling 2 Mini checkpoint directory containing the safetensors shards and tokenizer. No intermediate GGUF conversion is required.

### Server

```sh
build/bin/bel-server -m <checkpoint-dir> -c 16896 -t 5 \
  --host 127.0.0.1 --port 8080 --no-webui
```

`bel-server` automatically selects the embedded native engine for the checkpoint directory. Streaming and repeated requests are supported.

### CLI

```sh
build/bin/bel-cli -m <checkpoint-dir> -c 1536 -t 5 \
  -p "Hello, my name is" -n 8 --no-display-prompt -st
```

### Runtime tuning

The server enforces the tuned engine profile by default, with GPU fetch enabled. Experimental environment settings are ignored unless explicitly permitted.

| Environment variable | Purpose |
| --- | --- |
| `INK2_ALLOW_TUNING_OVERRIDE=1` | Permit experimental tuning overrides |
| `INK2_GPU_FETCH=0` | Disable GPU fetch when overrides are enabled |
| `INK2_PIN_HOST_MIB=N` | Configure pinned host memory when overrides are enabled |

Leave overrides disabled for the documented release configuration.

## Architecture

The native Inkling engine differs from the original disk-paged GGUF runtime. Its main execution and memory-management components are:

- **Native mixed-precision weights.** Original packed NVFP4 and MXFP8 expert weights and BF16 tensors are retained without extra weight quantisation.
- **Ampere execution kernels.** Low-bit GEMV kernels for decode, packed FP16 tensor-core prefill and optimised BF16 projections. Native checkpoint support does not imply hardware-native FP4 or FP8 tensor-core instructions on Ampere.
- **Architecture-aware residency.** Recurrent-core experts remain resident. Nonresident experts use a bounded GPU fetch cache with decayed-LFU eviction and overlapped PCIe transfers.
- **Dual-phase inference.** Prefill workspace is reclaimed and reused for expert residency and decode without restarting the process or discarding KV and recurrent states.
- **Attention and state management.** Fused grouped-query attention, sliding-window attention, dynamic KV growth, memory-aware pool resizing, CUDA graph reconstruction and K8/V4 KV-cache support.
- **Native execution.** Full recurrent execution, original expert selection and GPU-side greedy sampling.

```text
Original safetensors checkpoint
              │
     Native Inkling engine
              │
     Packed tensor-core prefill
              │
     Workspace reclamation
     KV / recurrent state retained
              │
     Decode residency and fetch cache
              │
     Low-bit GEMV + GPU sampling
              │
          Output tokens
```

## Legacy models

The earlier Betelgeuse runtime explored out-of-core inference using NVMe expert paging, pinned host caches and bounded VRAM pools. These results remain part of the project's development history, but the corresponding models are deprecated in v0.7.0.

| Historical model | Historical decode | Historical prefill |
| --- | ---: | ---: |
| DeepSeek-V4.1-Flash (1017), 748B as documented | 18.92 tok/s | 61.80 tok/s |
| Alpha-Orionis-XS (Qwen3.8 Flash Next), ~125B | 22.03 tok/s | Not separately certified |

These figures came from warm historical workloads. DeepSeek used DSpark speculative decoding; Alpha-Orionis-XS used a repeated workload. Neither is directly comparable to the new non-speculative Inkling benchmarks.

For existing installations:

- Keep a separate checkout and build of the revision previously validated with your model.
- Do not assume old `--moe-direct-io`, expert-pack, profile or DSpark settings carry over to v0.7.0.
- The stripped Alpha-Orionis-XS dense GGUF requires its matching legacy pager and companion files. It is not a standalone GGUF for upstream llama.cpp.
- Consult the README from the corresponding historical revision for setup and benchmark methodology. No specific legacy tag is asserted here.

The [Alpha-Orionis-XS model card](https://huggingface.co/Betelgeuse-AI/Alpha-Orionis-XS) remains a reference for the historical bundle and its applicable weight terms.

## Current limitations

- The first request can incur a one-time GPU-fetch preparation cost of roughly 11 seconds.
- Throughput depends on expert locality, VRAM and host-memory bandwidth.
- Longer contexts can reduce generation throughput.
- Validation is concentrated on Linux x86-64 and NVIDIA Ampere SM86.
- Preserving checkpoint weights does not guarantee bit-identical outputs across different execution and KV-cache precisions.
- Experimental MLA/CISA, activation quantisation, speculative decoding and unfinished research features are not included in the reported native release configuration.
- Legacy model compatibility is not guaranteed. This release does not establish general compatibility with arbitrary MoE architectures.

See [OPTIMISATION-MASTER.md](OPTIMISATION-MASTER.md) and [RELEASE_NOTES.md](RELEASE_NOTES.md) for development history and release details.

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=Betelgeuse-AI/Betelgeuse&type=Date)](https://star-history.com/#Betelgeuse-AI/Betelgeuse&Date)

## Licensing and acknowledgements

Betelgeuse builds on [llama.cpp](https://github.com/ggml-org/llama.cpp), [ggml](https://github.com/ggml-org/ggml) and native Inkling engine work contributed to Strata by baptisterajaut. Thanks to the developers of llama.cpp, Strata, Thinking Machines Lab and the wider local inference community.

The repository retains the upstream llama.cpp [MIT licence](LICENSE) and AUTHORS file. The Inkling engine carries `tools/inkling2/LICENSE.Strata` (MIT, copyright Niko1221 and the Strata contributors) and `tools/inkling2/CONTRIBUTING.md`. Packaged binaries include the required CUDA toolkit licence notices.

> [!IMPORTANT]
> The engine's MIT licence does not cover model weights. Read the original checkpoint licence and any applicable redistribution terms independently. Betelgeuse is an independent project and does not imply endorsement by any upstream project or model provider.

---

[Betelgeuse-AI/Betelgeuse](https://github.com/Betelgeuse-AI/Betelgeuse)
