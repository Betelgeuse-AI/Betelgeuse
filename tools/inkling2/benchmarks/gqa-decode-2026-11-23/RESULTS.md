# Inkling-2-Mini GQA decode attention

Date: 2026-11-23

Hardware: RTX 3060 12 GB, i5-12400, 15 GB RAM. The model is the official Inkling-2-Mini safetensors checkpoint. Rates exclude model load. Each run prefills the listed prompt and then generates 512 greedy tokens in one process.

## Result

The fused GQA decode kernel cuts long-context decode KV traffic by about 3x. At 262,144 positions with K8/V4 it moves decode from 13.6 to 16.3 tok/s (+19.9%) while prefill stays at 468 pp/s. The gain is small at 8K-16K because attention is not the decode bottleneck there; it grows with context.

The existing paired table is kept unchanged. The new Betelgeuse column below uses the same contexts, KV modes, flags, and route profile.

| Total positions | Prompt + generated | KV | Strata prefill (pp/s) | Betelgeuse prefill (pp/s) | Prefill delta | Strata decode (tok/s) | Betelgeuse decode (tok/s) | Decode delta |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 7,680 + 512 | K8/V4 | 698.6 | 664.6 | -4.9% | 36.0 | 37.3 | +3.6% |
| 8,192 | 7,680 + 512 | i8r | 738.7 | 687.8 | -6.9% | 35.9 | 36.6 | +1.9% |
| 16,384 | 15,872 + 512 | K8/V4 | 737.3 | 809.2 | +9.8% | 35.1 | 36.1 | +2.8% |
| 16,384 | 15,872 + 512 | i8r | 745.4 | 789.9 | +6.0% | 37.9 | 35.4 | -6.6% |
| 65,536 | 65,024 + 512 | K8/V4 | - | 744.4 | - | - | 32.8 | - |
| 262,144 | 261,632 + 512 | K8/V4 | 469.6 | 468.1 | -0.3% | 13.6 | 16.3 | +19.9% |

The 16K numbers are two-run medians. The 8K numbers are three-run medians for K8/V4 and two-run medians for i8r. The 64K number is a single run; the 262K number is a two-run median. Strata values are from the earlier paired runs: `prefill-tuning-16k` for 16K and `scale-8k-16k-262k` for 8K and 262K.

## Same-binary check

The 16K i8r decode row is below the earlier 37.7 tok/s single run. A binary built from the source before these changes measured 35.2 tok/s for both KV modes in the same session. The fused kernel measured 36.1 tok/s for K8/V4 and 35.4 tok/s for i8r. The i8r difference against the earlier table is therefore a run-condition difference, not a kernel regression.

| KV | Baseline prefill / decode | GQA prefill / decode | Prefill | Decode |
| --- | ---: | ---: | ---: | ---: |
| K8/V4 | 830.6 / 35.2 | 809.2 / 36.1 | -2.6% | +2.6% |
| i8r | 773.2 / 35.2 | 789.9 / 35.4 | +2.2% | +0.6% |

The baseline is the pre-change Betelgeuse binary. Prefill varies by several percent between launches; decode is stable.

## What changed

- Decode attention now runs one 128-thread block per KV head and chunk. The three query heads of the group share one K / V load, one key dot product stream, and one value stream. The partial max, sum, and output buffers keep the old per-query-head layout, so the merge kernel is unchanged. The decoder launches 8 x nchunk blocks instead of 24 x nchunk.
- The expert pools now restore the configured MXFP8 slot count first after a prompt and give the remaining VRAM to NVFP4. Previously both pools grew in proportion, which could leave NVFP4 short after a large prefill workspace was unmapped.
- The large prefill scratch matrices `PF.wf` and `PF.wh` now use the same elastic mapping as the chunk buffers and are unmapped after the prompt. This returns their memory to the expert pools before decode.
- Expert refill keeps the placement order when filling empty slots after a shrink. This makes the post-prompt resident set deterministic.

## Long-context ceiling

Prefill at 262K is limited by expert residency, not by the attention kernel. The 3.19 GiB K8/V4 cache leaves about 5.6 GiB for experts, so the prompt streams 3.9 MB of expert weights per token. Larger chunks reduce streaming per token but increase the prompt workspace:

| Context | Chunk | KV | Prefill (pp/s) | Decode (tok/s) | Note |
| ---: | ---: | --- | ---: | ---: | --- |
| 65,536 | 4,096 | K8/V4 | 744.4 | 32.8 | best measured balance |
| 65,536 | 8,192 | K8/V4 | 823.1 | 29.1 | prefill +10.6%, decode -11.3% (pre-GQA kernel) |
| 262,144 | 4,096 | K8/V4 | 468.1 | 16.3 | best measured balance |
| 262,144 | 6,144 | K8/V4 | 489.3 | 12.2 | prefill +4.3%, decode -11.6% (pre-GQA kernel) |
| 262,144 | 8,192 | K8/V4 | 432.5 | 10.2 | workspace shrinks the pools; reject |

The 6K and 8K rows used the pre-GQA decode kernel. GQA raises their decode rates, but not enough to pass the 4K setting. At 262K the 8K prompt workspace even forces pool churn and lowers prefill. Use 4,096 tokens as the dependable long-context chunk. The remaining prefill lever is a smaller KV cache or a larger expert pool, not a larger prompt batch.

## Method

- Commands used `--chunk 4096 --tc 1 --vram-reserve-mib 0 --cpu-threads 5 --kv-grow 1 --kv-grow-step 4096`, `STRATA_NUMA_POOL=0`, `INK2_CPU_ACT=int8`, `INK2_CPU_OPT=29`, `INK2_OFF_MARGIN_MIB=64`, and `INK2_MX_POOL_FRACTION=0.08`.
- Prompt token IDs are prefixes of `prefill-prefix-16384.ids`; the 65K and 262K prompts repeat the source cyclically and truncate to the target length.
- The route profile is `inkling2-routes-profiled.freq`. Speculative decoding is off.
- The 262K run used 3.19 GiB of K/V and 11.56 GiB of total VRAM.
- Greedy token IDs can differ from older logs because the expert pool changes which experts are resident on the GPU and which run on the CPU. The fused and old attention kernels produced the same output at the first decode layer in a debug A/B.

## Reproduce

Run `./run.sh`. Set `RUN_256K=0` to skip the 10-minute 262K case. Raw logs are in `raw/`.
