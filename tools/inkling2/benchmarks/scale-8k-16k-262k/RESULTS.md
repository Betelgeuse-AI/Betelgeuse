# Inkling paired context scale: Strata vs Betelgeuse

Date: 2026-11-22

Hardware: RTX 3060 12 GB, i5-12400, 15 GB RAM. The Betelgeuse side is the `bel-inkling2` executable built from this worktree. Both sides use the copied Strata Inkling engine implementation. Speculative decoding is off.

Each completed row used one paired launch: prefill the listed prompt, then greedily generate 512 tokens. The 8K and 16K Strata/Betelgeuse cases were run back-to-back with identical token IDs. The 262K Strata values are the existing measured run in `~/Docs/strata-inkling2-rtx3060/evidence/paired-high-context-2026-11-22.txt`; Betelgeuse was run against the same settings and repeated 16K prefix. Rates exclude model load.

## Results

| Total positions | Prompt + generated | KV | Strata prefill (pp/s) | Betelgeuse prefill (pp/s) | Prefill delta | Strata decode (tok/s) | Betelgeuse decode (tok/s) | Decode delta |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 7,680 + 512 | K8/V4 | 698.6 | 683.1 | -2.2% | 36.0 | 36.4 | +1.1% |
| 8,192 | 7,680 + 512 | i8r | 738.7 | 668.3 | -9.5% | 35.9 | 36.2 | +0.8% |
| 16,384 | 15,872 + 512 | K8/V4 | 791.0 | 751.8 | -5.0% | 35.2 | 35.2 | 0.0% |
| 16,384 | 15,872 + 512 | i8r | 834.9 | 789.6 | -5.4% | 35.1 | 35.1 | 0.0% |
| 262,144 | 261,632 + 512 | K8/V4 | 469.6 | 469.6 | 0.0% | 13.6 | 13.6 | 0.0% |

The 262K Betelgeuse K8/V4 row completed at 3.19 GiB KV and 11.55 GiB total VRAM use. Its rounded prefill and decode rates match the existing Strata measurement. The unfinished i8r attempt is excluded from the comparison.

The 8K and 16K Betelgeuse decode rates are within 1.1% of the paired Strata runs. Betelgeuse prefill was 2.2-9.5% lower in these single-run pairs. Treat those smaller-context prefill differences as indicative, not stable medians. In particular, the fresh 8K K8/V4 pair differs from the earlier two-run medians, confirming that short-context prefill varies between launches. The 262K K8/V4 comparison is the strongest high-context result: measured rates match to the reported precision.

## Method

- Context positions include the 512 generated tokens: 8,192, 16,384, and 262,144.
- Prompt token IDs are prefixes of `prefill-prefix-16384.ids`; the 262K prompt repeats the same 16,384-token source cyclically and truncates it to 261,632 IDs.
- Both runtimes used `--chunk 4096 --tc 1 --vram-reserve-mib 0 --cpu-threads 5 --kv-grow 1 --kv-grow-step 4096`, the same route-frequency profile, and `STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 INK2_MX_POOL_FRACTION=0.08`.
- KV modes are K8/V4 and i8r. The engine's K8/V4 is int8 K plus int4 V; i8r is scaled int8 K/V.
- Raw logs and the rerun script are in this directory. `betelgeuse-i8r-262144.log` is the intentionally interrupted run.

## 2026-11-23 update: fused GQA decode

The current Betelgeuse build uses a fused GQA decode attention kernel. The three query heads of each KV head share one K / V load and one partial-attention block. The original paired table above is unchanged for historical comparison.

At 262,144 positions with K8/V4, the two-run median prefill is 468.1 pp/s (was 469.6) and decode is 16.3 tok/s (was 13.6, +19.9%). At 16K, two-run medians are 809.2 pp/s / 36.1 tok/s for K8/V4 and 789.9 pp/s / 35.4 tok/s for i8r. See `../gqa-decode-2026-11-23/RESULTS.md` for the full table, same-binary baseline, and the 6K / 8K chunk ceiling tests.
