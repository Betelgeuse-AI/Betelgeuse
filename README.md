<p align="center">
  <!-- Hero GIF: assets/betelgeuse.gif (leave ~240px of vertical space above the title) -->
  <img src="assets/betelgeuse-splash.gif" alt="Betelgeuse: a deep-red block star with an amber core" width="320">
</p>

<h1 align="center">Betelgeuse <sub><i>Alpha-Orionis</i></sub></h1>

<p align="center">
  <b>Out-of-core mixture-of-experts inference for consumer hardware.</b><br>
  A native C++/CUDA fork of <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> that runs frontier-scale MoE models on an RTX 3060 12 GB, 16 GB of system RAM and a local NVMe drive.
</p>

<p align="center">
  <a href="#models"><img alt="Models" src="https://img.shields.io/badge/models-2-b22222"></a>
  <a href="#build"><img alt="Target" src="https://img.shields.io/badge/target-sm__86%20%C2%B7%20RTX%203060-76b900"></a>
  <a href="#limits"><img alt="Status" src="https://img.shields.io/badge/status-experimental-d4a017"></a>
  <a href="#licenses"><img alt="Engine" src="https://img.shields.io/badge/engine-MIT-blue"></a>
</p>

---

> [!WARNING]
> **Experimental.** The paged path is specific to one GPU class and one storage setup. It is not a production release and does not promise that any large MoE will fit or run well. Prefer `bel-server`. Configurations outside the documented envelope need independent validation.

Betelgeuse keeps dense computation on the GPU and streams routed experts through a bounded VRAM pool. Experts that are not resident are read from NVMe, cached in pinned host memory and promoted to VRAM as needed. Every routed expert is computed at full quantized precision. Nothing is pruned, skipped or requantized.

The executables are **`bel-server`** (OpenAI-compatible HTTP server) and **`bel-cli`** (interactive chat). The llama C API and internal CMake target names are unchanged.

---

## Contents

- [Models](#models)
- [DeepSeek-V4.1-Flash (1017)](#deepseek-v41-flash-1017)
- [Alpha-Orionis-XS (Qwen3.8 Flash Next)](#alpha-orionis-xs-qwen38-flash-next)
- [Build](#build)
- [Architecture](#architecture)
- [Limits](#limits)
- [Licenses and acknowledgements](#licenses)

---

## Models

| | DeepSeek-V4.1-Flash (1017) | Alpha-Orionis-XS (Qwen3.8 Flash Next) |
| --- | --- | --- |
| Total parameters | 748B (552B MoE backbone + 196B Engram table) | ~125B (6B active per token) |
| Layers | 40 (20 causal encoder + 20 decoder) | 48 (36 Gated DeltaNet + 12 sparse attention) |
| Routing | Top-6 of 384 experts, plus 1 shared | Top-10 of 512 experts, plus 1 shared |
| Quantization | UD-Q2_K_XL backbone, IQ4_NL Engram | GSQ-RCO IQ2_XS |
| On-disk size | ~310 GB | ~68 GB (four-file bundle) |
| Speculative drafter | DSpark (`dflash`) | None certified |
| Sustained decode | **18.92 tok/s** (warm) | **22.03 tok/s** (warm, repeated workload) |
| Prefill | **61.80 tok/s** (warm) | Not separately certified |

---

## DeepSeek-V4.1-Flash (1017)

Betelgeuse runs the 748B-parameter DeepSeek-V4.1-Flash (1017) on a 12 GB GPU, with all routed experts computed at full quantized precision, inside a 12 GiB cgroup memory ceiling.

### Server

The server runs under `systemd-run`, which enforces a 12 GiB limit covering process RSS and page cache and disables swap:

```bash
export MODELS=/path/to/Models/BIG-MOE/DeepSeek-V4.1-Flash-1017

systemd-run --user -p MemoryMax=12G -p MemorySwapMax=0 --unit=bel-ds41-srv \
  ./build/bin/bel-server \
  -m  "$MODELS/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00001-of-00007.gguf" \
  -md "$MODELS/dspark-DeepSeek-V4.1-Flash-1017-FLE-Q2K-Q8.gguf" \
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-p-min 0.50 \
  --moe-direct-io \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 512 -b 64 -ub 16 -np 1 -fit off --no-warmup
```

<details>
<summary><b>Flag reference</b></summary>

<br>

| Flag | Purpose |
| --- | --- |
| `-m` | Shard 1 of 7. The loader resolves shards 2 to 6 and the Engram companion automatically. |
| `-md` | DSpark drafter (`dflash`, 3 layers, 128 experts, top-3). Experts `Q2_K`, dense `Q8_0`, Markov and hyper-connection heads `F16`/`F32`. |
| `--spec-type draft-dspark` | Enables the multi-token speculative draft pipeline. |
| `--spec-draft-n-max 5` | Drafts a 5-token block per cycle through the chained Markov head. |
| `--spec-draft-p-min 0.50` | Drops draft positions with confidence below 0.50 before verification, which limits expert-union growth. |
| `--moe-direct-io` | Enables demand-paged expert streaming, the VRAM hot pool and the in-stream CUDA doorbell. |
| `-ngl 99` | Places dense layers, attention, shared experts and routing heads on CUDA0. Routed expert bodies are excluded by the loader. |
| `-b 64 -ub 16` | Keeps scratch buffers near 42 MiB and leaves more than 7 GiB of VRAM for the expert pool. |
| `-np 1 -c 512` | One slot. Parallel contexts are incompatible with the shared VRAM slot pool. |
| `-fit off --no-warmup` | Disables heuristic fit probes and the default empty warmup, so allocations stay static. |

</details>

### Client requests

Each pass uses the same prompt at temperature 0. Passes 1 and 4a seed the caches. Passes 2 to 4b are measured warm.

```bash
# Generation, warm (Pass 3: 68 tokens)
curl -s http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def binary_search(arr, target):\n",
    "n_predict": 68,
    "temperature": 0.0,
    "ignore_eos": true
  }'
```

<details>
<summary><b>All passes, including the prefill benchmark</b></summary>

<br>

- **Pass 1, cold learn:** the same request as above with `n_predict: 68`. It fills the per-layer working set.
- **Pass 2, warm:** `n_predict: 32`.
- **Pass 3, warm:** `n_predict: 68`, the request shown above.
- **Pass 4a, cold prefill:** a 64-token prompt (the function below) with `n_predict: 1`. It sweeps the prefill experts.
- **Pass 4b, warm prefill:** the same 64-token prompt with `n_predict: 1`, measured on resident pools.

The 64-token prompt is:

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

</details>

### Results

Measured on a 6-core / 12-thread x86-64 host with an RTX 3060 12 GB (locked 1800 to 2115 MHz, 200 W limit, +1000 MHz memory offset), 16 GB single-channel DDR4-3200 and a PCIe Gen4 NVMe drive on Btrfs.

| Phase | ColdSSD | Disk wait | VRAM hit | Throughput | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Pass 1, cold learn (68 tok) | 12.53% | 6,586 ms | 38.34% | 5.58 tok/s | Seeds the cache |
| Pass 2, warm (32 tok) | **0.00%** | **0 ms** | 55.07% | 18.15 tok/s | Fully resident |
| Pass 3, warm (68 tok) | **0.00%** | **0 ms** | 54.59% | **18.92 tok/s** | Bursts to 19.92 tok/s |
| Pass 4a, cold prefill (64 tok) | 14.65% | 4,782 ms | 46.55% | 6.83 tok/s | Prefill sweep |
| Pass 4b, warm prefill (64 tok) | **0.00%** | **0 ms** | 54.13% | **61.80 tok/s** | Layer-major prefill |

**Memory under load**

| Measure | Value |
| --- | ---: |
| CUDA0 VRAM peak | 11,396 MiB of 12,288 MiB |
| Process high-water mark (VmHWM) | 9.44 GiB |
| cgroup scope peak | 10.61 GiB, under the 12 GiB limit |
| Swap used | 0 bytes |

### Output

Pass 3 (warm, 68 tokens) reproduces the reference implementation exactly, including the loop invariant and integer division:

```python
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### How it works

<details>
<summary><b>Pipeline overview</b></summary>

<br>

```text
                          [input tokens]
                                │
          ┌─────────────────────┴──────────────────────┐
          ▼                                            ▼
 [DSpark drafter, dflash]                    [Engram staging, t = 0]
  3 layers, 128 experts, top-3                110.6 GB IQ4_NL table on NVMe
  5-token block, Markov head                  48 rows per token via io_uring
  confidence filter, p < 0.50                 copied to a 2 MB CUDA0 tensor
          │                                            │
          └─────────────────────┬──────────────────────┘
                                ▼
                 [speculative verification, 6 tokens]
                                │
          ┌─────────────────────┴──────────────────────┐
          ▼                                            ▼
 [Rank 0: primary specialist]              [Ranks 1 to 5: secondary]
  resident in the layer's pool              drawn from the layer's
  no replacement across passes              resident pool only
                                │
                                ▼
                  [execution stream]
   VRAM hot slots: direct GEMM, no PCIe transfer
   host-RAM slots: H2D DMA on stream_h2d, 17.5 GB/s
                                │
                                ▼
                 [verified tokens, 18.92 tok/s]
```

</details>

<details>
<summary><b>Cache partitioning and routing invariants</b></summary>

<br>

**Per-layer working set.** Each layer's resident set has 20 slots, split into two zones:

1. **VRAM hot zone (5 slots).** Holds the layer's most frequently used experts. Executes on tensor cores with no PCIe cost.
2. **Host-RAM zone (15 slots).** Holds secondary specialists in pinned DDR4. Transferred over `stream_h2d` at about 17.5 GB/s.

**Routing invariants.** Routing stays fixed across passes, so warm runs produce the same output as cold ones:

- Rank 0 is evaluated against the full 20-slot pool, with no replacement on cold or warm passes.
- Ranks 1 to 5 are drawn from the same resident pool, so promotion between tiers does not change routing.

Physical promotion happens at the end of a pass. The most frequently used experts are moved into the VRAM zone for the next pass. Warm disk access is 0.00% and cold-to-warm outputs match.

**Engram staging.** The 110.6 GB `per_layer_token_embd.weight` table is active only at layers 1 and 14. Its row indices come from the input n-gram window, so they are known before layer 0 runs. `io_uring` reads the 48 rows per token into a pinned buffer and copies them to a small CUDA0 tensor. The table never has to fit in system RAM.

</details>

---

## Alpha-Orionis-XS (Qwen3.8 Flash Next)

Alpha-Orionis-XS is the Qwen3.8 Flash Next GSQ-RCO IQ2_XS quantization, packaged as a four-file bundle for Betelgeuse.

The model is at **[Betelgeuse-AI/Alpha-Orionis-XS](https://huggingface.co/Betelgeuse-AI/Alpha-Orionis-XS)**. Read its model card and every applicable weight license before downloading or redistributing.

### Bundle

```text
Alpha-Orionis-XS/
├── Qwen3.8-Flash-Next-Dense.gguf    ~3.77 GB   dense tensors, tokenizer, metadata
├── experts.bin                       35.48 GB   routed experts, 4 KiB-aligned blob
├── Qwen3.8-Flash-Next-PLE.gguf       26.82 GiB  per-layer embeddings, byte-identical
└── profile-qwen4exp.bin              6,920 B    hot-expert hints
```

```bash
hf download Betelgeuse-AI/Alpha-Orionis-XS \
    Qwen3.8-Flash-Next-Dense.gguf experts.bin \
    Qwen3.8-Flash-Next-PLE.gguf profile-qwen4exp.bin \
    --local-dir /path/to/Alpha-Orionis-XS

export BUNDLE=/path/to/Alpha-Orionis-XS
```

Plan for about **68 GB** of storage. The profile is a set of hints for likely hot experts. It is not weights and not a cache snapshot.

### Server

Start with one context, one slot and a small window. This does not enable generated cache warmup.

```bash
export MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0

systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 \
    build/bin/bel-server \
    -m "$BUNDLE/Qwen3.8-Flash-Next-Dense.gguf" \
    --moe-direct-io --moe-pack-dir "$BUNDLE" \
    --moe-profile "$BUNDLE/profile-qwen4exp.bin" --moe-slots 5000 \
    -ngl 99 -np 1 -c 512 -ub 64 -t 6 \
    --cache-warmup 0 --host 127.0.0.1 --port 8080
```

### CLI

```bash
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 \
    build/bin/bel-cli \
    -m "$BUNDLE/Qwen3.8-Flash-Next-Dense.gguf" \
    --moe-direct-io --moe-pack-dir "$BUNDLE" \
    --moe-profile "$BUNDLE/profile-qwen4exp.bin" --moe-slots 5000 \
    -ngl 99 -np 1 -c 512 -ub 64 -t 6 --cache-warmup 0 \
    --color on -n 256
```

At the `You >` prompt, `/clear` resets the conversation and `/exit` quits. For a single turn, add `-st -p "Your prompt"`.

<details>
<summary><b>Paging and warmup flags</b></summary>

<br>

| Option | Meaning |
| --- | --- |
| `--moe-direct-io` | Enables demand-paged routed experts and the VRAM pool. The demand worker is buffered, so this does not mean every read is `O_DIRECT`. Required for the bundle. |
| `--moe-slots 5000` | Total VRAM expert slots across all layers. Not context length and not host-cache size. |
| `--moe-pack-dir DIR` | Directory containing `experts.bin`. |
| `--moe-profile PATH` | Seeds hot-expert choices. Workload-dependent, with no guaranteed hits. |
| `--auto-warm` | Opt-in. Generates up to 256 tokens and replays them once before serving. Adds about 42 seconds to startup in the recorded test. |
| `--cache-warmup N` | Explicit generated-warmup budget. `0` disables it. Maximum 8192. |
| `--cache-warmup-prompt TEXT` | Topic for generated warmup. A different prompt can still miss. |
| `--no-warmup` | Disables only the ordinary empty graph warmup. Use `--cache-warmup 0` to disable generated warmup. |

The host victim cache defaults to **8 GiB** on the supported DMA and CUDA Graph path and counts against the process memory budget. Set `MOE_CACHE_MIB=N` to resize it, or `MOE_CACHE_MIB=0` to disable it. The 5,000 VRAM slots are a separate tier.

</details>

### Terminal banner

Startup renders the star from `assets/betelgeuse.svg` as a true-color terminal image, using ANSI half-cell rasterization, beside a native text wordmark. `--color on` forces color, `--color off` selects plain text, `--simple-io` is available on the CLI, and `NO_COLOR` disables automatic color. Routine paging diagnostics are quiet by default. Use `--verbose` (`-v`) to see doorbell, cache and backend traces.

### Measured results

| Configuration | Token generation | Evidence |
| --- | ---: | --- |
| Upstream llama.cpp | 0.9 tok/s | User-reported, same hardware. |
| Strata | 8 tok/s | User-reported, same hardware. |
| Fork, cold / first measured request | 16.80 tok/s | Auto-warm arm. Warmup was paid separately. |
| Fork, warm repeated request | 22.03 tok/s | Separate run on a repeated workload. |

The 16.80 and 22.03 figures come from separate historical runs. They are not a paired benchmark or a universal speed claim. The [measurement record](docs/fluxaeros.md#measurements-and-interpretation) keeps each run's protocol and caveats. Storage, routing, prompt, cache state and context size all matter.

---

## Build

The paged configuration targets **Linux x86-64, CUDA `sm_86` (RTX 3060 12 GB) and local NVMe**.

You need a C++17 compiler, CMake, the CUDA toolkit and driver, liburing development files, and the standard [llama.cpp build dependencies](docs/build.md). Building the embedded logo also needs Python 3, `rsvg-convert` (librsvg) and ImageMagick. The installed binaries need none of these at runtime.

```bash
cmake -S . -B build \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DLLAMA_BUILD_TOOLS=ON

cmake --build build --config Release --target llama-server llama-cli -j6
```

CMake targets stay `llama-server` and `llama-cli`. Output files are `build/bin/bel-server` and `build/bin/bel-cli`. There are no wrapper executables. Upstream binaries and container images do not include the paged runtime.

---

## Architecture

1. **Dense model.** A small GGUF keeps the original metadata, tokenizer and all non-expert tensors. Expert descriptors are payload-free and point to contiguous virtual offsets past EOF. Expert shapes and quantization types are unchanged, and the GGUF parser stays strict.
2. **Expert store.** `experts.bin` holds the original quantized bytes. Each expert stores its gate, up and down payloads, padded to a 4 KiB stride, ordered by layer and expert.
3. **VRAM slot pool.** The pager maps the model's routed experts onto a bounded set of device slots.
4. **Host victim cache and demand worker.** Evicted experts can be reused from registered host memory. Remaining misses are served by demand-ordered `io_uring` reads and staged transfers. The `--moe-direct-io` flag name is historical.
5. **Per-layer embeddings.** A companion GGUF supplies per-layer token embeddings. It is required, not an optional draft model.

Missing expert bytes and failed reads are errors. The engine never silently runs with zero weights, and never falls back to payloads absent from the dense file. The stripped dense GGUF needs this fork's pager, so do not load it in upstream llama.cpp.

---

## Limits

- **One paged context per process.** Keep `-np 1`. Multi-user server features do not make the shared paging state safe for parallel contexts.
- **Hardware.** Custom CUDA kernels target `sm_86`. Other GPUs, CPU-only builds, non-Linux hosts and other backends are not certified for paging.
- **Filesystem.** Direct I/O requires contiguous, uncompressed sector access. On Btrfs, run `chattr +C` on model directories before copying weights in.
- **Envelope.** The recorded Qwen configuration is `-c 512 -ub 64 -t 6 -ngl 99`. Larger contexts or batch sizes need their own memory and correctness checks.
- **Speculative decoding.** The DSpark drafter is certified only for the DeepSeek configuration above. No MTP or speculative speedup is certified for the Qwen bundle, and its historical MTP smoke test diverged from the golden continuation.
- **Cold versus warm.** Peak throughput is reached on warm passes with a populated resident set. Cold starts pay disk-streaming costs in proportion to the new expert footprint.
- **Scope of numbers.** Warm repeated-workload decode is not prefill speed, cold-start latency or cross-workload throughput. Do not generalize these figures.

Read the [server documentation](tools/server/README.md), the [CLI documentation](tools/cli/README.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before changing deployment or implementation assumptions.

---

## Licenses

The engine is a fork of **[llama.cpp](https://github.com/ggml-org/llama.cpp)**, built on **[ggml](https://github.com/ggml-org/ggml)**. Engine code is MIT-licensed, with upstream notices retained in [LICENSE](LICENSE). Betelgeuse is independent. No endorsement by llama.cpp, ggml, Qwen, DeepSeek, NVIDIA, Hugging Face, Strata or ISTA-DASLab is implied.

> [!IMPORTANT]
> **Model weights are not covered by the engine's MIT license.** Check each one before use or redistribution.
>
> - **Alpha-Orionis-XS** is built from the Qwen3.8 Flash Next weights. The source quantization repository declares Apache-2.0, but the upstream [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) includes commercial MaaS and AI Work Assistant restrictions. See the [bundle's model card](https://huggingface.co/Betelgeuse-AI/Alpha-Orionis-XS).
> - **DeepSeek-V4.1-Flash** weights and the DSpark drafter are distributed under their own upstream terms. Read those before use.

### Third-party components

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib): HTTP server used by `bel-server`. MIT.
- [nothings/stb](https://github.com/nothings/stb): image decoder for the multimodal subsystem. Public domain.
- [nlohmann/json](https://github.com/nlohmann/json): JSON library used by tools and examples. MIT.
- [mackron/miniaudio](https://github.com/mackron/miniaudio): audio decoder for the multimodal subsystem. Public domain.
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h): process launcher. Public domain.
