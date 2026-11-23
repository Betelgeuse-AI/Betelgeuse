# Inkling-2-Mini 16K prefill tuning

Date: 2026-11-22

## Result

Betelgeuse has no consistent 16K prefill deficit against Strata in this paired sweep. At the 4,096-token chunk, Betelgeuse measured 760.0 pp/s with K8/V4 and 752.6 pp/s with i8r; Strata measured 737.3 and 745.4 pp/s. Decode rates and total inference time were also close.

A 6,144-token chunk with an 18% MXFP8 pool produced the best first-run paired results: 803.5 pp/s with K8/V4 and 799.3 pp/s with i8r on Betelgeuse. That prefill gain did not repeat. The repeated Betelgeuse runs were 729.2 and 727.1 pp/s. An 8,192-token chunk reduced Betelgeuse decode to 31.8 tok/s (K8/V4) and 31.4 tok/s (i8r), and did not improve combined inference time against its 4,096-token results. Keep 4,096 as the dependable setting; 6,144 is a candidate for another controlled repeat, not a demonstrated ceiling.

## Method

Each main run processed the same 16,384-token prompt and then generated 512 greedy tokens in one process. Maximum context was 32,768 positions. The fixed input was `prefill-prefix-16384.ids`; the route profile was `inkling2-routes-profiled.freq`. Speculative decoding was off. The main sweep used five CPU threads and compared Strata with Betelgeuse for K8/V4 and i8r KV.

Shared settings were `--tc 1 --vram-reserve-mib 0 --kv-grow 1 --kv-grow-step 4096`, `STRATA_NUMA_POOL=0`, `INK2_CPU_ACT=int8`, `INK2_CPU_OPT=29`, and `INK2_OFF_MARGIN_MIB=64`. The chunk / MXFP8 pool pairs were 4,096 / .08, 6,144 / .18, and 8,192 / .30. The 6,144 case was repeated with five CPU threads, then tested with six threads and with the MXFP8 pool fraction reduced to .08.

The separate 16K-total-position test used a 15,872-token prompt plus 512 generated tokens and a 16,384 maximum context. It is kept in its own raw-log subdirectory and is not mixed into the 16K-prompt results below.

## Main paired results

Rates are prompt prefill / generated-token decode. Combined time is prompt time plus the 512-token decode time, excluding model load.

| KV | Chunk | MX pool | Strata pp/s / tok/s | Strata combined | Betelgeuse pp/s / tok/s | Betelgeuse combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K8/V4 | 4,096 | .08 | 737.3 / 35.1 | 36.808 s | 760.0 / 35.7 | 35.897 s |
| K8/V4 | 6,144 | .18 | 763.9 / 37.0 | 35.284 s | 803.5 / 34.4 | 35.276 s |
| K8/V4 | 8,192 | .30 | 762.4 / 34.9 | 36.143 s | 781.8 / 31.8 | 37.038 s |
| i8r | 4,096 | .08 | 745.4 / 37.9 | 35.503 s | 752.6 / 37.7 | 35.352 s |
| i8r | 6,144 | .18 | 748.8 / 37.1 | 35.700 s | 799.3 / 36.9 | 34.358 s |
| i8r | 8,192 | .30 | 814.5 / 28.6 | 38.039 s | 771.9 / 31.4 | 37.528 s |

## Repeat and extra tuning

The 6,144 / .18 five-thread repeat did not reproduce the first-run prefill gain:

| KV | Engine | Prefill | Decode | Combined |
| --- | --- | ---: | ---: | ---: |
| K8/V4 | Strata | 727.6 pp/s | 36.0 tok/s | 36.757 s |
| K8/V4 | Betelgeuse | 729.2 pp/s | 37.0 tok/s | 36.291 s |
| i8r | Strata | 746.6 pp/s | 33.8 tok/s | 37.109 s |
| i8r | Betelgeuse | 727.1 pp/s | 36.8 tok/s | 36.441 s |

At 6,144 / .18 with six CPU threads, the paired combined times remained within about 0.7% between engines: 35.130 vs 35.239 s for K8/V4 and 36.561 vs 36.367 s for i8r. At 6,144 / .08 with five threads, combined times were 35.078 vs 35.582 s for K8/V4 and 35.413 vs 35.475 s for i8r. Neither change gave a consistent improvement across KV formats.

Peak reported VRAM was about 11.55-11.56 GiB of 11.63 GiB. Host RAM peak was not sampled. The large-chunk runs used fewer resident NVFP4 experts due to workspace and MXFP8 pool allocation; the 8,192 / .30 runs restored roughly 994-1,006 NVFP4 and 95-102 MXFP8 experts after prefill, versus about 1,840-1,865 NVFP4 and 84-86 MXFP8 experts after the 6,144 / .18 refill.

## Interpretation

The 4,096-token chunk bridges the Betelgeuse prefill path to the Strata baseline for both KV formats. The 6,144-token chunk can be faster in a single run, but the repeat varies too much to claim a stable gain. The 8,192-token chunk loses decode throughput and does not reduce paired end-to-end time. Six CPU threads and the .08 MXFP8 split also did not produce a repeatable win.

No engine source or kernel changes were made for this sweep. Raw logs and scripts are archived beside this report.

## 2026-11-23 decode update

The current Betelgeuse build adds a fused GQA decode attention kernel and restores the configured MXFP8 expert pool share after prefill. At the 4,096-token chunk, the two-run medians are 809.2 pp/s / 36.1 tok/s for K8/V4 and 789.9 pp/s / 35.4 tok/s for i8r. The prefill result above is unchanged within launch variance; the decode gain grows with context. See `../gqa-decode-2026-11-23/RESULTS.md`.
