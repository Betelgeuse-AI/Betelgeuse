# Betelgeuse demand-paged MoE (historical FluxAeros record)

Betelgeuse is a custom llama.cpp fork with a VRAM expert pool, demand-ordered I/O, a registered host victim cache and optional automatic cache warmup. Selected experts and stored quantized weights are not pruned or requantized. It is experimental and partly broken; prefer `bel-server`. This document combines current four-file bundle setup with explicitly historical FluxAeros measurements, not a portability or production-readiness claim.

## Historical tested envelope and current limits

- Linux x86-64, NVIDIA RTX 3060 12 GB, CUDA `sm_86`, 16 GB system RAM, local NVMe.
- The `qwen4exp` Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS model with 48 trunk layers and 512 experts per layer. Its actual router selects 10 experts, not 8.
- Historical tests used two original GGUF shards: all routed expert tensors were in the first, with per-layer token embeddings in the second. The current bundle instead uses stripped dense metadata/tensors, `experts.bin` and a PLE companion as described below.
- One paged model context per process, one server slot, context 512, 5,000 VRAM slots, `-ub 64` with the existing effective eight-token paging chunks.
- Process envelope: `MemoryMax=12G`, `MemorySwapMax=0`. The host victim cache defaults to 8 GiB when the DMA/CUDA Graph path is available. It is not an additional 8 GiB outside that limit.
- `MOE_HOT_K=0` and `MOE_PREKICK=0`. No background disk warmer competes with demand reads.

Other GPUs, operating systems, backends, models, larger contexts and parallel paged contexts are not certified by these results. The current custom CUDA sources target `sm_86`; a CPU-only build of this fork is not part of the supported configuration. Do not expose the server to an untrusted network without a separate authentication and deployment review.

## Build all targets

Install a C++17 compiler, CMake, a working CUDA toolkit/driver, the liburing development package, and the normal llama.cpp build dependencies. Python 3.10+ and NumPy are needed for the packer and its tests; it imports the in-tree `gguf-py` reader.

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build -j12
ctest --test-dir build --output-on-failure -j2
python3 -m unittest discover -s tests -p test_moe_pack.py
```

This builds native `bel-cli` and `bel-server`, completion/benchmark tools, configured examples and tests. The CMake target names remain `llama-cli` and `llama-server`; only their output filenames change. Some existing tests fetch external model metadata. A network failure or timeout in those tests must be reported separately, not counted as a successful test. Do not regenerate reference snapshots to hide failures.

## Prepare the four-file bundle

Download the runtime files from [Betelgeuse-AI/Alpha-Orionis-XS](https://huggingface.co/Betelgeuse-AI/Alpha-Orionis-XS), separately from the MIT-licensed engine. The quantization source declares Apache-2.0, but the underlying model carries [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), including commercial MaaS/AI Work Assistant restrictions. Read the model card and all preserved notices; do not assume unrestricted Apache-2.0 weight licensing or endorsement.

```text
Alpha-Orionis-XS/
  Qwen3.8-Flash-Next-Dense.gguf
  experts.bin
  Qwen3.8-Flash-Next-PLE.gguf
  profile-qwen4exp.bin
```

Set `BUNDLE` to this directory. The stripped dense GGUF preserves original metadata, tokenizer, non-expert bytes and expert descriptors. Dense descriptors precede payload-free expert descriptors, with contiguous virtual offsets starting at EOF; the standard GGUF metadata parser stays strict. Its companion filename identifies the PLE GGUF. It is not a standalone upstream-compatible model. Use `--moe-direct-io` and `--moe-pack-dir`; missing expert bytes must fail, never fall back to absent payloads.

Set the local runtime path before launching:

```bash
export BUNDLE=/path/to/Alpha-Orionis-XS
```

The published bundle preserves the dense tensors and tokenizer, separates the original quantized expert bytes, and retains the byte-identical PLE shard. Temporary bundle-building/publication scripts are not a runtime interface. Existing local pack reuse must leave its originals unchanged; never modify hardlinked weights in place.

The expert layout is deterministic: each expert stores adjacent original gate, up and down payloads, followed by padding to a 4 KiB stride. Experts are ordered within each layer, then layers in model order. Every expert start and the complete blob must be 4 KiB aligned; final file padding alone is insufficient. The stripped runtime derives its index from tensor metadata, so **no `native_experts.txt` or `expert-pack.json` is required at runtime**. For the tested source files, this layout requires an `experts.bin` of 35,481,714,688 bytes (35.48 GB); the unchanged PLE is 28,800,138,432 bytes (26.82 GiB), and the profile is 6,920 bytes. Retaining every dense tensor gives 3,759,953,920 bytes of dense payload plus about 11 MB of metadata: roughly 3.77 GB for the dense GGUF, not 3.2 GB. Allow roughly 68 GB for the complete bundle.

The historical original-shard pack record listed 35,454,976,000 bytes (about 33.02 GiB), not the new aligned bundle size. The existing source pack has unaligned expert strides and cannot simply be hardlinked as the new runtime blob; it must be repacked with padding while preserving all expert payload bytes and leaving originals untouched. The measurements below describe the historical engine and pack, not performance validation of the new layout.

The older `tools/moe_pack.py --model SHARD1 --output PACK` remains useful for preparing or verifying original-shard packs. Its `native_experts.txt` and `expert-pack.json` describe that conversion format, not extra files to ship in the stripped runtime bundle. `--verify` samples the first, middle and last expert in each layer; `--verify-all` compares every expert. A sampled check is not full verification.

For reproducible bundle conversion, the existing packer also accepts `python3 tools/moe_pack.py --model SHARD1 --shard2 SHARD2 --output BUNDLE`. Add `--existing-pack PACK` to sample-verify and reuse the legacy pack's bytes, adding per-expert padding when needed; unchanged PLE is hardlinked or symlinked across filesystems. Add `profile-qwen4exp.bin` separately. Existing output files are never overwritten.

The optimized demand worker deliberately uses buffered `io_uring` reads so the kernel page cache remains useful. The historical option name `--moe-direct-io` does not mean every demand read bypasses the page cache.

## Run

```bash
export MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 \
    build/bin/bel-server -m "$BUNDLE/Qwen3.8-Flash-Next-Dense.gguf" -ngl 99 \
    --moe-direct-io --moe-pack-dir "$BUNDLE" --moe-slots 5000 \
    --moe-profile "$BUNDLE/profile-qwen4exp.bin" --cache-warmup 0 \
    -ub 64 -np 1 -c 512 -t 6 --host 127.0.0.1 --port 8080
```

For the CLI, substitute `build/bin/bel-cli` and use `-st -p "Your prompt" -n 128` instead of the server address options. Keep the same model, paging, memory and thread settings when comparing performance. The command above explicitly disables generated cache warmup; it does not promise a warm-cache result.

`MOE_DROP_MMAP`, `MOE_SPLIT`, `MOE_PLE_STAGING`, `MOE_DMA` and the supported victim-cache path default on for paging. `MOE_VICTIM_CACHE=0` or `MOE_CACHE_MIB=0` disables the victim cache. `MOE_CACHE_MIB=N` changes its requested size; it does not raise the process memory limit. Decrease it when other contexts or models need host memory.

`--moe-slots 5000` controls the total device expert slots shared across layers, not context length or host-cache size. `--moe-profile PATH` seeds likely hot experts from a profile; it is not a cache snapshot and does not guarantee 22 tok/s. Workload routing and available startup seeding budget still matter.

Routine paging statistics use debug logging. Add `--verbose` (`-verbose` or `-v`) for doorbell, cache/audit and backend traces; set `MOE_IO_DIAG=1 MOE_DIAG=1` as well for demand bytes, cache hits/saves and CUDA Graph capture/replay. Those environment variables do not bypass verbosity. `MOE_VERIFY_ROUTER=/local/file` records raw router logits for exact comparisons. Traces contain workload-dependent data and should not be bundled in public archives.

Demand read/DMA failure is fatal rather than substituting zero expert weights. This protects output correctness; it is not automatic request recovery. Shut down the process, repair the storage/pack problem and restart. The single-context ownership guard rejects a second paged context rather than sharing global worker state unsafely.

## Automatic cache warmup

Warmup is opt-in and completes before requests or CLI generation begin:

```bash
build/bin/bel-server ... --auto-warm
build/bin/bel-cli ... --cache-warmup 128 \
    --cache-warmup-prompt "Explain the kinds of programming tasks I usually ask about."
```

- `--auto-warm` generates up to 256 tokens and replays the same sequence once.
- `--cache-warmup N` sets the budget; `0` disables it, and the maximum accepted value is 8192. `LLAMA_ARG_CACHE_WARMUP` is the environment equivalent.
- `--cache-warmup-prompt TEXT` selects a representative prompt. The default is a general explanatory prompt, not the benchmark prompt.
- The budget is additionally bounded by the context space remaining after the prompt. An EOG token ends generation early.
- Warmup clears KV/recurrent state, resets sampler RNG state and request performance counters, but keeps reusable expert caches. It requires a decoder-only generation context.
- `--no-warmup` controls the pre-existing empty graph warmup, not this separate option. Use `--cache-warmup 0` to disable cache warmup.

Startup cost is real and reported separately. On the tested workload, the 256-token generation plus replay took about 42 seconds. Warming one topic cannot guarantee cache hits for unrelated topics, nor can an 8 GiB cache hold the entire model. Do not report its cost as a free cold-start speedup.

## Measurements and interpretation

Corrected same-hardware baselines reported by the user are **0.9 TG tok/s for llama.cpp** and **8 TG tok/s for Strata**. They are not new controlled measurements performed for this bundle.

The often-quoted fork values **16.80 cold/first-request TG** and **22.03 warm TG** come from different historical runs below. The 16.80 result used generated startup warmup, so "cold" does not mean zero startup work; 22.03 is a repeated-workload second request in the committed-cache repeat. They are not a paired result, nor a guarantee for the renamed stripped bundle.

Measurements are from October 30, 2026. The controlled protocol drops the expert-pack page cache before startup, performs a 64-token warmup, then issues two identical 256-token greedy completions with a 22-token prompt and prompt caching disabled. Both requests remain in the same process. Physical MiB/generated-token includes the request's prompt reads and divides by 256 generated tokens.

| Configuration | TG Run 1 / Run 2, tok/s | PP Run 1 / Run 2, tok/s | Physical MiB/generated-token |
| --- | ---: | ---: | ---: |
| Cache-off control | 13.87 / 14.27 | 14.89 / 17.95 | 44.66 / 39.67 |
| Committed 8 GiB victim cache, `be9328717` repeat | 16.29 / 22.03 | 13.85 / 19.35 | 35.37 / 12.43 |
| Release lifecycle/warmup code, auto-warm disabled, traced | 16.32 / 21.43 | 13.89 / 18.62 | 35.38 / 12.47 |
| Same code, auto-warm enabled, traced | 16.80 / 21.85 | 16.18 / 18.29 | 30.54 / 12.88 |

20+ tok/s is a warm repeated-workload result, not a cold-start guarantee. Prefill has not improved consistently across both runs. Keep paired values and repeat measurements rather than selecting only the fastest run.

CUPTI main-stream profiling of the committed victim-cache path measured approximately:

| Phase | Main-stream span | Resolve/copy family | Other GPU compute | Gaps |
| --- | ---: | ---: | ---: | ---: |
| Run 1 decode | 60.26 ms/token | 34.89 | 24.59 | 0.78 |
| Run 2 decode | 44.30 ms/token | 18.97 | 24.54 | 0.79 |
| Paired prompt processing | 55.25 ms/token | 46.32 | 8.12 | 0.81 |

These are GPU main-stream spans, not complete request wall times. D2H and H2D overlap the main stream; do not add their duration again. Fetch/promotion stalls remain the main target for further work.

An additional prefill-capture ablation measured 16.19 / 21.86 TG and 13.43 / 18.77 PP. It did not achieve stable prefill CUDA Graph replay or a repeatable improvement, so that experimental change was removed rather than enabled by default.

## Stripped bundle validation (October 31, 2026)

- Dense GGUF: 3,770,978,720 bytes; aligned expert pack: 35,481,714,688 bytes; unchanged PLE: 28,800,138,432 bytes; profile: 6,920 bytes.
- All 1,079 dense tensor payloads were compared byte-for-byte with the source, with all 144 expert descriptors retained. The aligned pack passed first/middle/last expert comparisons in every layer (144 experts sampled); this is not a new full-pack comparison.
- All configured targets built. Ten packer/bundle tests and `test-arg-parser`, `test-c`, and `test-gguf` passed. The historical 60-test suite and long model benchmarks were not repeated.
- `bel-server` loaded the four-file bundle under `MemoryMax=12G`, no swap, 5,000 slots, context 512 and `-ub 64`. With `--cache-warmup 0`, the 22-token computing-history prompt produced exactly the eight reference tokens `[271, 248068, 198, 760, 1156, 6587, 264, 11346]` from the retained `09e69a047` binary.
- The test exercised prefill chunks with no final-layer output rows; those empty FFNs must not introduce references to absent expert payloads. Nonempty routed layers still require the paging pool.
- `/health`, `/v1/models`, tokenizer round-trip and an eight-token OpenAI chat completion passed. Native `bel-cli` connected to that server, showed the Betelgeuse banner, streamed an eight-token reply and exited. These short functional checks are not throughput measurements.
- Tiny malformed inputs were rejected: stripped model without paging, non-expert offset outside the file, and truncated PLE. GGUF descriptor offsets remain subject to the standard contiguous-layout check.
- The complete four-file bundle was moved to the Btrfs model directory and all four published SHA-256 hashes matched. The original duplicate shards/pack/index and Windows bundle copy were removed; the separate MTP sidecar was preserved.
- Direct `bel-cli -m ...` was exercised with two interactive chat turns, `/clear` and `/exit`, without `--server-base`; it exited with code 0. Both native binaries displayed the embedded SVG star rendering in captured terminal output, and the server became healthy and exited cleanly.
- The direct-CLI shutdown test exposed cache discard touching released mmap holes reused by anonymous mappings. `drop_cache` and `advise_random` now operate only on owned fragments. The deterministic `test-mmap` canary regression failed before the fix and passed after it; `test-arg-parser` also passed.
- Quiet colored chat completed two turns with no routine MoE diagnostics; monochrome and simple-I/O modes also exited cleanly without color sequences. The exact `-verbose` flag restored doorbell telemetry and opt-in I/O counters. Reasoning-to-answer label/color transitions were exercised, and the native terminal screen was inspected. These are functional checks, not new throughput benchmarks.

## Historical release validation (before the stripped bundle)

- All then-configured build targets completed, including the CLI and server (now emitted as `bel-cli` and `bel-server`).
- 60 CTests pass. `test-quant-type-selection` previously timed out fetching external Hugging Face metadata and is excluded from the final local pass; it is not claimed as passing.
- Six packer tests pass, including mixed layer types, exact round trips, corruption/truncation detection and refusal to overwrite existing packs.
- Full comparison of the existing model pack verifies all 24,576 experts against the original GGUF bytes. No second full-size pack was needed.
- Both server warmup arms match 576 golden output tokens each and all 28,080 original router records. The enabled trace contains exactly one extra contiguous block of 24,768 warmup records.
- Seeded CLI generation (seed 1234, temperature 0.8, 32 output tokens, backend sampling enabled) is identical with cache warmup off/on after excluding the displayed performance counters. Both CLI processes exit cleanly.
- Native CUDA tests cover all-hit recency, D2H victim preservation, H2D slot reuse, repeated graph execution, eviction invalidation and completion-flag resets.
- Controlled server runs have zero cgroup swap/OOM kills, clean shutdown and sampled memory below 10.6 GiB. Startup reclaim can reach the 12 GiB cap.

These historical binaries were built with GCC 16.1.1, CUDA 13.3, glibc 2.43 and driver 610.43.02. The checks above do not claim that the new stripped bundle or renamed binaries have been revalidated. They are local-build results, not a portable binary certification for older Linux distributions or different CPUs. Rebuild and verify on the deployment system. Engine archives contain no model weights or MTP sidecar; the separately licensed model bundle is distributed independently.

## MTP status

The available one-layer `qwen4exp` MTP sidecar loads and drafts tokens. It is about 2.58 GiB and must be provided separately. With the draft on CPU and the host victim cache reduced to 6 GiB, a 16-token smoke test accepted 6 of 8 drafted tokens (75%) but did not match the golden continuation: the first difference was generated token 8. The matched non-MTP control reproduced the golden tokens.

MTP is therefore not enabled or certified for the exact-output paged release. Acceptance rate alone is not evidence of either speed or exact routing. Its metadata declares a different, unspecified `other` license; no sidecar is included in release artifacts.

## Validation checklist

- Build all configured targets and record compiler/CUDA/device versions.
- Run C/C++ and packer tests; retain failures and external-test limitations.
- Check golden tokens and full router traces on the exact binary, including CUDA Graph replay.
- Test cache warmup on/off, sampler reset, CLI execution and clean startup/shutdown.
- Keep the 12 GiB memory and zero-swap envelope; check cgroup OOM counters.
- Verify the expert pack against the user's GGUF before deployment.
- Include source, LICENSE, setup documentation, binary dependencies and SHA-256 checksums in engine artifacts; keep separately licensed model weights in the model distribution and exclude private prompts and raw routing traces.
- Leave MTP and speculative disk prefetch disabled. Do not label untested platforms or workloads stable.
- A local commit/archive is not publication. Publishing, tagging and uploading remain separate human decisions.
