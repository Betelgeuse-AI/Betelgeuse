# Inkling-2-Mini prompt prefill scaling

Date: 2026-11-22

## Result

On one fixed 16,384-token prompt, increasing the expert prompt chunk raised measured prefill from 467.8 pp/s at chunk 1,024 to a best observed 806.8 pp/s at chunk 8,192. The 10,240-token chunk fell to 669.0 pp/s. This is the best measured prefill point in this sweep, not a verified hardware ceiling; each setting was run once.

The earlier fixed-prefix length sweep with chunk 256 measured 47.4 pp/s at 257 tokens, 116.4 at 1,024, 163.3 at 4,096, and 169.8 at 16,384. This shows prefill throughput scales with prompt length until the small-chunk overhead is amortized, then plateaus.

## Method

All prompt inputs are prefixes of the same archived 16,384-token ID sequence. The separate 257-token decode prompt was not used here. Each run used 32K context, i8r KV, tensor-core dense projections enabled, five CPU expert threads, and one greedy output token. Load time and the single output token are excluded from prefill rate.

Fixed settings:

- STRATA_NUMA_POOL=0
- INK2_CPU_ACT=int8
- INK2_CPU_OPT=29
- INK2_OFF_MARGIN_MIB=64
- max context 32,768, KV i8r, --tc 1, --vram-reserve-mib 0, --cpu-threads 5
- route profile: inkling2-routes-profiled.freq

The MXFP8 expert-pool fraction was .08 for chunks 256 through 2,048, .09 for chunk 4,096, .30 for chunk 8,192, and .35 for chunk 10,240. Larger chunks reserve more GPU workspace, so the engine had to assign more of its remaining pool to MXFP8 to keep the minimum number of MXFP8 slots. The pool size and chunk changed together in the larger tests.

The prompt runs used prefill-prefix-257.ids, prefill-prefix-1024.ids, prefill-prefix-4096.ids, or prefill-prefix-16384.ids, with --n 1. For reproducibility, the main command shape was strata-ink2 --model <checkpoint-dir> --max-context 32768 --chunk N --kv i8r --tc 1 --vram-reserve-mib 0 --cpu-threads 5 --tokens-file PREFIX --n 1 --expert-frequencies inkling2-routes-profiled.freq.

Hardware: RTX 3060 12 GB; Intel i5-12400, 6 physical cores; 15 GB system RAM. No model files were downloaded.

## Measurements

| Prompt tokens | Chunk | MX pool fraction | Prompt time | Prefill rate | GPU expert slots and pool | Expert fetch |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 257 | 256 | .08 | 5.418 s | 47.4 pp/s | 1,799 NVFP4 + 85 MXFP8; 8.15 GiB | 53.8 MB/token |
| 1,024 | 256 | .08 | 8.794 s | 116.4 pp/s | 1,799 NVFP4 + 85 MXFP8; 8.15 GiB | 27.0 MB/token |
| 4,096 | 256 | .08 | 25.076 s | 163.3 pp/s | 1,799 NVFP4 + 85 MXFP8; 8.15 GiB | 27.0 MB/token |
| 16,384 | 256 | .08 | 96.494 s | 169.8 pp/s | 1,799 NVFP4 + 85 MXFP8; 8.15 GiB | 27.0 MB/token |
| 16,384 | 1,024 | .08 | 35.026 s | 467.8 pp/s | 1,700 NVFP4 + 80 MXFP8; 7.70 GiB | 7.2 MB/token |
| 16,384 | 2,048 | .08 | 25.744 s | 636.4 pp/s | 1,536 NVFP4 + 72 MXFP8; 6.96 GiB | 4.0 MB/token |
| 16,384 | 4,096 | .09 | 22.549 s | 726.6 pp/s | 1,222 NVFP4 + 65 MXFP8; 5.59 GiB | 3.0 MB/token |
| 16,384 | 8,192 | .30 | 20.307 s | 806.8 pp/s | 482 NVFP4 + 112 MXFP8; 2.87 GiB | 2.7 MB/token |
| 16,384 | 10,240 | .35 | 24.491 s | 669.0 pp/s | 235 NVFP4 + 68 MXFP8; 1.50 GiB | 3.1 MB/token |

The engine reported about 11.58 GiB of 11.63 GiB VRAM used after the runs. Host RAM peak was not sampled.

## Interpretation and limits

Larger chunks batch more prompt rows through the grouped expert path. This reduces repeated dispatch and weight-fetch overhead per prompt token, explaining the increase from 169.8 pp/s at chunk 256 to 806.8 pp/s at chunk 8,192. Chunk 10,240 was slower and retained only 10% of experts in its GPU pool, compared with 20% at chunk 8,192; the data points to workspace and residency pressure past the best tested chunk, but the one-run measurements do not isolate those effects.

The original prefill-only and short-prompt decode results were separate. The paired runs below measure both phases in one launch. They showed that permanent prompt scratch reduced expert residency, and that elastic buffers need a cache refill when returned slots become available.

These are single-run measurements. A 16K one-chunk run was not attempted. No 64K or longer prompt was tested. The 16K/8,192 point is the highest measured rate in this sweep, not a claim that the GPU or implementation has reached its ceiling. The single output token is not a meaningful decode benchmark.

## Initial paired sweep before elastic pool refill

Each row is one launch: the same 16,384-token prompt prefix was processed, then the engine generated 512 greedy tokens. Both rates therefore come from the same run. All runs used 32K context, i8r KV, five CPU workers, the profiled route file, CPU int8 activations, CPU option 29, and tensor-core dense projections. The MXFP8 pool fraction was adjusted to fit the minimum slot count for each chunk.

| Chunk | MX pool fraction | GPU expert pool | 16K prefill time / rate | 512-token decode time / rate | Prefill + decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | .08 | 7.70 GiB | 35.526 s / 461.2 pp/s | 14.718 s / 34.8 tok/s | 50.244 s |
| 2,048 | .08 | 6.96 GiB | 27.957 s / 586.0 pp/s | 18.779 s / 27.3 tok/s | 46.736 s |
| 4,096 | .09 | 5.59 GiB | 22.658 s / 723.1 pp/s | 24.446 s / 20.9 tok/s | 47.104 s |
| 8,192 | .30 | 2.87 GiB | 20.803 s / 787.6 pp/s | 42.785 s / 12.0 tok/s | 63.588 s |

These runs omitted --kv-grow and are superseded by the stabilized paired runs below. They show why chunk-only tuning was misleading: workspace size changed the resident pool. Model load time is excluded from combined times.

Decode slows as the chunk grows and the resident expert pool shrinks in these runs, but each point is a single sample and chunk size changes workspace, pool size, and MXFP8 allocation together. Treat the trend as evidence of a tradeoff to investigate, not a causal measurement.

## Stabilized paired result after elastic pool refill

The current paired setting uses --chunk 4096, --kv-grow 1, --kv-grow-step 4096, and an 8% MXFP8 pool fraction. Each row is a 16,384-token prompt followed by 512 generated tokens in the same launch.

| Run | Prefill time / rate | Decode time / rate | Combined inference time |
| --- | ---: | ---: | ---: |
| 1 | 23.726 s / 690.5 pp/s | 13.421 s / 38.1 tok/s | 37.147 s |
| 2 | 23.629 s / 693.4 pp/s | 13.459 s / 38.0 tok/s | 37.088 s |

The 512 generated token IDs matched between runs. After the prompt, the engine restored 310 NVFP4 and 14 MXFP8 experts (1.40 GiB) from the mapped checkpoint. Prompt fetch includes this refill. This is repeatable on the fixed prompt; it is not a broad workload result. Details and cause are in evidence/paired-refill-stability.txt.

## Marlin W4A16 assessment

The standalone Strata Inkling engine has no Marlin dispatch path. Decode uses custom selected-expert GEMV kernels for the checkpoint's packed NVFP4 and MXFP8 layouts; prefill uses a chunked grouped expert path. On this RTX 3060 (SM86), Blackwell-native NVFP4 tensor-core execution is unavailable.

Marlin W4A16 is therefore not a drop-in switch for this checkpoint. A useful comparison needs a kernel that reads the checkpoint's packed values and scales, or a documented conversion to the layout Marlin expects. Such a conversion may change quantization behavior. Batch-one decode may also favor the current GEMV path over a Marlin GEMM. Marlin could still be a prefill experiment, but no local benchmark establishes a speedup. The current kernels are not proven to be at peak.
