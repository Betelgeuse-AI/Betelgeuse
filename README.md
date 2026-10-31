# Betelgeuse (Alpha-Orionis)

![Betelgeuse: a deep-red block star with an amber core](assets/betelgeuse.svg)

**A custom [llama.cpp](https://github.com/ggml-org/llama.cpp) fork for frontier-scale mixture-of-experts inference on an RTX 3060 with 12 GB VRAM, 16 GB system RAM and local NVMe.**

Betelgeuse keeps dense computation on the GPU and brings routed experts into a bounded VRAM pool only when needed. This is native C++/CUDA work, not a launcher around an upstream binary. The executables are **`bel-server`** and **`bel-cli`**; the llama C API and internal build target names remain unchanged.

> **Experimental and partly broken. Prefer `bel-server`.** The paged path is hardware-specific, not a production release or a general promise that any large MoE will fit or run well. CLI behavior, unfamiliar models and configurations outside the documented envelope need independent validation. No MTP configuration is certified for exact-output paging.

## What the numbers mean

| Result | Token generation | Evidence and limits |
| --- | ---: | --- |
| Upstream llama.cpp baseline | **0.9 tok/s** | User-reported on the same hardware; not a new controlled benchmark here. |
| Strata baseline | **8 tok/s** | User-reported on the same hardware; not a new controlled benchmark here. |
| Historical fork cold/first measured request | **16.80 tok/s** | Auto-warm-enabled arm; startup warmup was paid separately. This is not an untouched, zero-warmup cold start. |
| Historical fork warm repeated request | **22.03 tok/s** | A different run of the committed victim-cache path, on a repeated workload. |

**16.80 and 22.03 are from distinct historical runs, not a paired benchmark or a universal speed claim.** [The measurement record](docs/fluxaeros.md#measurements-and-interpretation) retains each actual pair, protocol, prefill results and caveats. These results predate the renamed, stripped bundle; they do not certify its performance. Storage, routing, prompt, cache state and context size matter. A profile file does not guarantee 22 tok/s.

## Architecture

1. **Dense model and tokenizer:** a small GGUF retains the original metadata, tokenizer and non-expert tensor bytes. Dense descriptors come first, followed by payload-free expert descriptors with contiguous virtual offsets starting at EOF. Expert shapes and quantization types are unchanged, and the standard GGUF metadata parser stays strict.
2. **NVMe expert store:** `experts.bin` contains the original quantized bytes, not pruned or requantized weights. Each expert stores adjacent gate, up and down payloads, then padding to a 4 KiB stride. Experts are ordered within each layer, with layers in model order; every expert start and the complete blob are 4 KiB aligned. The stripped bundle derives this deterministic layout without a `native_experts.txt` runtime sidecar.
3. **VRAM slot pool:** the pager resolves the model's actual routed experts into a bounded set of device slots. The tested 48-layer, 512-expert-per-layer `qwen4exp` model routes **10 experts**, not eight.
4. **Host victim cache and demand worker:** evicted experts can be reused from registered host memory; remaining misses use demand-ordered NVMe reads and staged transfers. The optimized worker uses **buffered `io_uring` reads**, despite the historical `--moe-direct-io` name. Kernel page-cache reuse remains useful.
5. **Per-layer embeddings (PLE):** a separate companion GGUF supplies the per-layer token embeddings. The dense file records its safe companion filename. It is required, not an optional draft model.

Missing expert bytes and demand read/DMA failures are errors, not a reason to silently run with zero weights or fall back into absent dense-file payloads. The stripped dense GGUF requires this fork's pager; do not pass it to upstream llama.cpp as a standalone model.

## Get the four-file bundle

Model repository: **[Betelgeuse-AI/Alpha-Orionis-XS](https://huggingface.co/Betelgeuse-AI/Alpha-Orionis-XS)**. Read its model card and all applicable weight licenses before downloading or redistributing.

Keep exactly these four runtime files together:

```text
Alpha-Orionis-XS/
  Qwen3.8-Flash-Next-Dense.gguf
  experts.bin
  Qwen3.8-Flash-Next-PLE.gguf
  profile-qwen4exp.bin
```

With the Hugging Face CLI installed:

```bash
hf download Betelgeuse-AI/Alpha-Orionis-XS \
    Qwen3.8-Flash-Next-Dense.gguf experts.bin \
    Qwen3.8-Flash-Next-PLE.gguf profile-qwen4exp.bin \
    --local-dir /path/to/Alpha-Orionis-XS
export BUNDLE=/path/to/Alpha-Orionis-XS
```

The profile seeds likely hot experts; it is neither weights nor a prewarmed cache snapshot. For the tested source files, the dense GGUF is about **3.77 GB** (3.50 GiB of dense tensor payload plus metadata), the per-expert-aligned `experts.bin` layout requires **35,481,714,688 bytes** (35.48 GB), the unchanged PLE companion is **28,800,138,432 bytes** (26.82 GiB), and the profile is **6,920 bytes**. Allow roughly 68 GB of storage for the complete runtime bundle. All non-expert tensors are retained; this is not a 3.2 GB dense-only pruning scheme. Original two-shard GGUFs and pack-building indexes are conversion inputs, not extra runtime requirements.

## Build native binaries

The paged configuration targets **Linux x86-64, CUDA `sm_86`, RTX 3060 12 GB and local NVMe**. Install a C++17 compiler, CMake, CUDA toolkit/driver, liburing development files and the normal [llama.cpp build dependencies](docs/build.md).

```bash
cmake -S . -B build \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DLLAMA_BUILD_TOOLS=ON
cmake --build build --config Release --target llama-server llama-cli -j6
```

CMake targets remain `llama-server` and `llama-cli` for dependencies; their native output files are `build/bin/bel-server` and `build/bin/bel-cli`. There are no compatibility wrapper executables. Upstream release binaries and container images do not contain this fork's paged runtime.

## Run: server recommended

Start with one model context, one server slot and a small context. The following does **not** enable generated cache warmup:

```bash
export BUNDLE=/path/to/Alpha-Orionis-XS
export MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 \
    build/bin/bel-server \
    -m "$BUNDLE/Qwen3.8-Flash-Next-Dense.gguf" \
    --moe-direct-io --moe-pack-dir "$BUNDLE" \
    --moe-profile "$BUNDLE/profile-qwen4exp.bin" --moe-slots 5000 \
    -ngl 99 -np 1 -c 512 -ub 64 -t 6 \
    --cache-warmup 0 --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` or use the OpenAI-compatible API. The systemd scope enforces the historical 12 GiB process and zero-swap limits on the 16 GB host; it does not allocate extra memory. Keep the server bound to localhost unless you have separately reviewed authentication and deployment security.

For a single-turn CLI request, use the same memory envelope and paging parameters:

```bash
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 \
    build/bin/bel-cli \
    -m "$BUNDLE/Qwen3.8-Flash-Next-Dense.gguf" \
    --moe-direct-io --moe-pack-dir "$BUNDLE" \
    --moe-profile "$BUNDLE/profile-qwen4exp.bin" --moe-slots 5000 \
    -ngl 99 -np 1 -c 512 -ub 64 -t 6 --cache-warmup 0 \
    -st -p "Explain why stars appear to twinkle." -n 128
```

### Paging and warmup flags are different controls

| Option | Meaning |
| --- | --- |
| `--moe-direct-io` | Enable demand-paged routed experts and the VRAM pool. The optimized demand worker is buffered, not a claim that all reads use `O_DIRECT`. Required for the stripped bundle. |
| `--moe-slots 5000` | Total VRAM expert slots shared across layers, not context length or host-cache size. This is the tested setting, not a universal fit. |
| `--moe-pack-dir DIR` | Directory containing `experts.bin`. The stripped bundle needs no external text index. |
| `--moe-profile PATH` | Seed hot-expert choices from a profile. Workload-dependent hints, not guaranteed cache hits or guaranteed throughput. |
| `--auto-warm` | Opt in to generating up to 256 tokens and replaying that sequence once before serving. About **42 seconds** of extra startup work in the recorded test, not free acceleration. |
| `--cache-warmup N` | Set that generated warmup budget explicitly; `0` disables it, maximum 8192, further bounded by context and EOG. `--cache-warmup-prompt TEXT` chooses a representative topic. |
| `--no-warmup` | Disable the ordinary empty graph warmup only. It does **not** disable generated cache warmup; use `--cache-warmup 0` for that. |

Cache warmup resets inference/KV state, sampler RNG and request timers while retaining reusable expert caches. A different prompt can still miss. The host victim cache defaults to 8 GiB on the supported DMA/CUDA Graph path and is **inside** the process memory budget; `MOE_CACHE_MIB=N` adjusts it, while `MOE_CACHE_MIB=0` or `MOE_VICTIM_CACHE=0` disables it. The 5,000 VRAM slots are a separate tier.

## Bundle format and provenance

The published bundle separates the original shards without changing their quantized weights: dense tensors and tokenizer remain in the dense GGUF, routed experts move to the aligned blob, and the original PLE shard remains byte-identical. Dense metadata marks stripped experts with `betelgeuse.stripped_experts=true` and records a safe companion filename. Expert tensor shapes and types remain available to the runtime even though their original payloads are absent.

Only packs matching the per-expert-aligned bundle layout can be reused directly; a historical pack with unaligned expert strides must be repacked without changing its quantized payloads. Existing local pack reuse must preserve the original files; never modify hardlinked weights in place. The four published runtime files, rather than temporary conversion or publication scripts, are the deployment interface. See [the detailed layout and historical evidence](docs/fluxaeros.md).

## Limits and status

- One paged model context per process. Keep `-np 1`; general server multi-user features do not make shared paging state safe for parallel contexts.
- The custom CUDA path targets `sm_86`. Other GPUs, CPU-only builds, non-Linux systems and other backends are not certified for paging.
- The recorded envelope is `-c 512 -ub 64 -t 6 -ngl 99`, with effective eight-token paging chunks. Larger contexts and different batch sizes need their own memory/correctness checks.
- No certified MTP or speculative-decoding speedup. The historical MTP smoke test diverged from the golden continuation despite accepting drafted tokens; no MTP sidecar belongs to this bundle.
- Warm repeated-workload decode is not prefill speed, cold-start latency or cross-workload throughput. Do not generalize the historical numbers.
- Read [server documentation](tools/server/README.md), [CLI documentation](tools/cli/README.md) and [contribution guidance](CONTRIBUTING.md) before changing deployment or implementation assumptions.

## Licenses and acknowledgements

The engine is a fork of **llama.cpp**, built on **[ggml](https://github.com/ggml-org/ggml)**. Engine code remains under the [MIT license](LICENSE), with upstream copyright and third-party notices retained. Betelgeuse is independent: no endorsement by llama.cpp/ggml, Qwen, NVIDIA, Hugging Face or Strata is implied.

**Weights are separately licensed; the engine's MIT license does not cover them.** The source quantization repository declares Apache-2.0, while the underlying [Qwen3.8-Flash-Next LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) supplies **Qwen Community License 1.0**, including commercial MaaS/AI Work Assistant restrictions. Do not treat the bundle as unconditionally Apache-2.0. Consult the [bundle's model card and preserved notices](https://huggingface.co/Betelgeuse-AI/Alpha-Orionis-XS/tree/main), the upstream license and any applicable terms before use or redistribution.

Retained third-party README attributions:

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `bel-server` - MIT license.
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by the multimodal subsystem - Public domain.
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT license.
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by the multimodal subsystem - Public domain.
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain.
