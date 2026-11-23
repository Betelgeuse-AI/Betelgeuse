# Strata Inkling-2-Mini RTX 3060 decode handoff

Date: 2026-11-22

## Current state

- Source: Strata PR #6520 worktree, detached at `dd7fa18dc838607724191270a28d2d9e41ebb1bb`.
- Worktree path: `~/strata-inkling2-test`.
- This worktree contains uncommitted changes in `engine/src/inkling2/ink2_engine.cu`, `engine/src/inkling2/ink2_cpu.cpp`, and `engine/src/inkling2/ink2_cpu.hpp`.
- Local model: `/home/zyxciss/Models/Inkling-2-Mini`.
- Last built executable: `~/strata-inkling2-build/strata-ink2` (3,617,688 bytes). It was built from this worktree before the worktree move.
- The 321 MiB build tree was copied to `~/strata-inkling2-build` and CMake was reconfigured against the relocated `engine/` source; the executable remains available there.
- Machine: RTX 3060 12 GB, Intel i5-12400 (6 physical cores, AVX2), 15 GB system RAM.
- No model or GGUF files were downloaded during these runs.

## Result at a glance

Best observed short-context plain-decode result: **40.1 tok/s** over 512 generated tokens. The stable same-run 16K prompt plus 512-token decode setting uses a 4,096 chunk, --kv-grow 1, --kv-grow-step 4096, and an 8% MXFP8 pool. Two runs measured 690.5 pp/s and 38.1 tok/s, then 693.4 pp/s and 38.0 tok/s, with 37.147 s and 37.088 s combined inference time. The generated IDs matched. See the stabilized results in evidence/paired-refill-stability.txt.

The best run used a route-ranked expert placement, 1,799 NVFP4 and 85 MXFP8 GPU slots (8.15 GiB total expert pool, about 56% of all experts), and the file-backed expert source. Decode-time GPU expert fetch was reported as 0.0 MB/token. The model reported 11.32 GiB VRAM at load and 11.58 GiB used after generation.

## Run method

The benchmark used the same fixed tokenized prompt file (`~/Docs/strata-inkling2-rtx3060/evidence/strata-ink2-prompt-257.ids`) for the short-context comparisons. The engine reports prompt tokens divided by prompt wall time and generated tokens divided by decode wall time. Model load time is reported separately and excluded from both rates. A 512-token generation was used for the sustained decode comparisons; 128-token runs were used for initial tuning. Generation was greedy. The context capacity was 32,768, but only about 769 positions were occupied in the 512-token run; this is a short-context result, not a long-context benchmark.

Best observed command and environment:

```sh
STRATA_NUMA_POOL=0 \
INK2_CPU_ACT=int8 \
INK2_CPU_OPT=29 \
INK2_OFF_MARGIN_MIB=64 \
INK2_MX_POOL_FRACTION=0.08 \
~/strata-inkling2-build/strata-ink2 \
  --model /home/zyxciss/Models/Inkling-2-Mini \
  --max-context 32768 --chunk 256 --kv i8r --tc 1 \
  --vram-reserve-mib 0 --cpu-threads 5 \
  --tokens-file ~/Docs/strata-inkling2-rtx3060/evidence/strata-ink2-prompt-257.ids --n 512 \
  --expert-frequencies ~/Docs/strata-inkling2-rtx3060/evidence/inkling2-routes-profiled.freq
```

`INK2_CPU_OPT=29` disables CPU software prefetch for this comparison. It reached 40.1 tok/s once; an earlier default CPU option run also reached 40.1, so the data does not show a reliable gain from disabling prefetch. `INK2_CPU_ACT=int8` is the optional CPU activation quantization path. `STRATA_NUMA_POOL=0` uses a single CPU worker pool across online cores. The frequency file ranks outer experts for initial placement; layers 6-17 remain first in placement because they are looped three times.

The run's captured summary was:

```text
expert placement: read 3072 route-count entries
offload: 8.22 GiB free after the rest, reserve 0 MiB -> 1799 NVFP4 + 85 MXFP8 expert slots (8.15 GiB, 56 % of the experts)
cpu experts: avx2, NVFP4 activations int8, 1 node(s), threads per node: 5
load 11.6 s, 11.32 GiB in VRAM (ctx 32768)
expert fetch: prompt 53.8 MB/token, decode 0.0 MB/token; VRAM used 11.58 GiB of 11.63
prompt 257 tokens 5578 ms (46.1 tok/s), decode 512 tokens 12768 ms (40.1 tok/s)
```

## Baseline-to-best measurements

All rates are reported from the Strata engine's own wall-clock timing. Runs share the same short prompt unless noted. A prompt rate from one run is not directly comparable to another because timings varied noticeably between launches.

| Stage / configuration | Prompt rate | Decode rate | Notes |
| --- | ---: | ---: | --- |
| First memory-safe run, default expert order, 257 prompt / 128 generated | 48.1 pp/s | 25.9 tg/s | File-backed source; short-context baseline. |
| 16K prompt run, chunk 1024 | 464.3 pp/s | 23.1 tg/s | Decode was only one token, so it is not a sustained decode measurement. |
| Fixed-prefix prefill, chunk 256 | 47.4 pp/s at 257; 116.4 at 1K; 163.3 at 4K; 169.8 at 16K | Not measured (one output token only) | Same prefix IDs and pool settings across all four lengths. |
| Fixed-prefix 16K prefill, chunk sweep | 467.8 pp/s at 1K; 636.4 at 2K; 726.6 at 4K; 806.8 at 8K; 669.0 at 10K | Not measured (one output token only) | Pool size falls as chunk workspace grows; see PREFILL-SCALING.md. |
| Kernel tuning on short run | - | 25.0-25.9 tg/s | `INK2_CPU_ACT=int8`: 25.7; selected-row grouping 4: 25.5; grouping 1: 25.0; default grouping 2: 25.9. |
| Route-ranked outer placement, first profile | - | 32.2 tg/s at 128 tokens; 32.0 tg/s at 512 | Core loop experts kept resident; hot outer experts ordered by observed route counts. |
| Route-ranked placement, default 20% MXFP8 budget, 512 tokens | 51.2 pp/s | 32.8 tg/s | Comparator before the 8% MXFP8 split. |
| Route-ranked placement, 8% MXFP8 split, 512 tokens | 44.2-61.4 pp/s | 38.6-40.1 tg/s | Includes repeat at 38.6 and runs at 40.1. The 40.1 rate was observed twice with different CPU-option settings. |
| 8% split, default CPU fp32 activations | 60.5 pp/s | 37.7 tg/s | Same route profile and five workers. |
| 6.1% MXFP8 split (65 MXFP8 slots), 512 tokens | 60.1 pp/s | 37.8 tg/s | More NVFP4 slots did not improve decode. The 6.0% setting itself allocated only 63 MX slots and exited because the minimum is 64. |
| 8% split, four CPU workers, `INK2_CPU_OPT=29` | 46.5 pp/s | 39.8 tg/s | Close to five workers, slightly below the best observed run. |

A route-aware placement run improved the short-context baseline from 25.9 to 32.2 tok/s at 128 generated tokens, and held 32.0 tok/s at 512. Tuning the expert-pool split and CPU activation path produced the best observed 40.1 tok/s. This remains a single-stream plain-decode result with a short context.

## Initial paired 16K sweep before elastic refill

These first paired runs omitted --kv-grow and did not refill experts evicted by prompt workspace. They are retained as the baseline for the fix; see evidence/paired-prefill-decode-results.txt.

## Stabilized paired result after elastic pool refill

With --chunk 4096, --kv-grow 1, --kv-grow-step 4096, and an 8% MXFP8 split, a 16K prompt followed by 512 decode tokens measured 690.5 pp/s and 38.1 tok/s, then 693.4 pp/s and 38.0 tok/s on repeat. Combined phase times were 37.147 s and 37.088 s, excluding model load. The 512 generated IDs matched. The engine restored 310 NVFP4 and 14 MXFP8 experts (1.40 GiB) after prefill. This is the current stable paired setting for the fixed prompt. Full method and cause are in evidence/paired-refill-stability.txt.

## Final decode tuning and Marlin assessment

The best observed decode setting remains INK2_GEMV_MINB=164 and INK2_SEL_RPW=2 at 40.1 tok/s. The 112-block threshold reached 39.7 tok/s; eight selected-expert GEMV rows per warp reached 39.2 tok/s. These single runs do not show a better setting. See evidence/decode-kernel-tuning.txt.

The standalone Strata Inkling engine has no Marlin dispatch path. Its decode path uses custom selected-expert GEMV kernels for the checkpoint's packed NVFP4 and MXFP8 layouts; prefill uses a chunked grouped expert path. RTX 3060 is SM86 and cannot use Blackwell-native NVFP4 tensor-core execution. Marlin W4A16 is not a drop-in for this packed checkpoint layout, and no local benchmark shows it would win. It could be prototyped for prefill after mapping or converting the stored values and scales; batch-one decode may favor the current GEMV kernel. The present kernels are not proven to be at peak.

The earlier decode drop came from the prefill workspace shrinking the expert pool, then leaving returned slots empty in the lazy mmap path. The new refill step restores the startup route-ranked experts after the full prompt. The paired repeats show stable decode and prefill rates for the 4K elastic-buffer setting; larger chunks still reduce usable residency during the prompt.

## Placement and route replay details

Inkling-2-Mini has 24 physical layers. Layers 6-17 are applied three times, so all 1,536 NVFP4 experts in those 12 layers were placed first. The remaining capacity was used for route-ranked outer NVFP4/MXFP8 experts. At the best split, the observed count varied slightly with available CUDA free memory: about 1,790-1,799 NVFP4 slots and 84-85 MXFP8 slots.

The `--record-routing` files are int32 arrays sized `[48 layer applications][32768 positions][8 selected experts]`, or 50,331,648 bytes each. To build physical-layer frequency counts, map application slots 0-23 to their corresponding physical layers; map slots 24-47 to `6 + (slot - 24) % 12`. For the 257-token prompt and 512-token generation, the recorded generation positions used for the profile were 257 through 768. The route profile is static and sample-dependent; it is not adaptive placement during the run.

For the same captured route traces, the 8% pool layout (about 254 outer NVFP4 plus 84 MXFP8 slots) covered roughly 73-75% of outer expert selections, depending on which saved profile ranked the candidates. A 6.1% layout (about 300 outer NVFP4 plus 65 MXFP8 slots) had slightly higher raw route coverage but decoded more slowly. These are offline replay counts, not speed predictions.

## Memory-fit change

The first two attempts to start the CPU-offload path were killed because the original code allocated and pinned about 14.59 GiB of expert data, before other engine memory. The local change uses a read-only mmap of the safetensors shard. CPU workers read only routed misses through split tensor views. At startup, selected experts are packed into a small temporary buffer and copied into GPU slots. Prefill uses bounded 64 MiB host/device staging halves. This avoids eagerly pinning the full expert bank.

The lazy mmap mode disables background GPU promotions in this prototype. The captured decode runs report 0.0 MB/token of decode expert fetch, but CPU computation still services non-resident expert misses. Latest-run host RAM peak was not sampled, so do not infer a peak from the VRAM output.

## Known limitations

- No DSpark, MTP, or speculative decoding was enabled. In this branch `--spec 1` is rejected when `--cpu-threads` is active.
- The latest throughput measurements are short-context only. They do not establish 16K/64K/256K decode rates.
- No logit/KL comparison was performed for `INK2_CPU_ACT=int8`; do not claim numerical equivalence from throughput results.
- The 40.1 tok/s result is the maximum observed among these trials; repeats ranged down to 37.7 tok/s. The additional 112-block and eight-row-per-warp tests did not improve it.
- The route profile was collected on one prompt and is vulnerable to overfitting. Rebuild it from a separate held-out prompt before treating it as general placement.
- There is no saved raw stdout file for every run. The tables above transcribe the terminal summaries captured during the runs. The route traces, route-count files, and tokenized prompts are preserved alongside this report.

## Source changes in this worktree

- `engine/src/inkling2/ink2_engine.cu`: mmap safetensors for lazy CPU-offload; build expert tensor views; bound prefill staging; route-frequency initial GPU placement; tune the NVFP4/MXFP8 pool split with `INK2_MX_POOL_FRACTION`.
- `engine/src/inkling2/ink2_cpu.cpp` and `.hpp`: let the CPU kernels read split expert tensor views from the mapped checkpoint.
- All three source files remain modified and uncommitted. No commit or push was made.

## Preserved evidence

The companion evidence folder in `~/Docs/strata-inkling2-rtx3060/evidence/` contains the route recordings, derived frequency profiles, prompt IDs, prefill prefix files, and captured prefill summaries. `summarize_routes.py` reproduces physical-layer frequency counts from a raw route recording. PREFILL-SCALING.md records the fixed-prefix prompt-length and chunk-size sweep.
