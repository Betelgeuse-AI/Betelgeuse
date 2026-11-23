# Strata and Betelgeuse baseline

These are fixed-token comparisons on the RTX 3060. Betelgeuse is the `llama-ink2` target built from this worktree; the same Strata Inkling engine source and local engine patches are used in both binaries. This measures integration/build parity, not the separate native llama.cpp inference path.

Each cell was run twice, once in the initial pass and once in an interleaved repeat. The table reports the median of those two measurements. Prefill varies noticeably between launches, so treat two-run prefill medians as preliminary. Decode used 512 generated tokens in every run.

| Context | KV | Strata prefill (pp/s) | Betelgeuse prefill (pp/s) | Strata decode (tok/s) | Betelgeuse decode (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 4K | K8/V4 | 566.8 | 548.4 | 36.9 | 36.9 |
| 4K | i8r | 505.8 | 535.0 | 37.1 | 37.1 |
| 8K | K8/V4 | 653.3 | 759.8 | 36.6 | 36.3 |
| 8K | i8r | 720.9 | 685.8 | 36.1 | 36.2 |

All eight Strata/Betelgeuse pairs emitted identical 512-token sequences. The decode rates are within 0.8% across the four configurations. The two-run prefill medians range from 4.9% below to 16.3% above the Strata measurements; this spread is larger than the decode variation and needs another paired run when the CUDA driver is available.

## Method

- Contexts: 4,096 tokens (3,584 prompt + 512 generated) and 8,192 tokens (7,680 prompt + 512 generated).
- KV: K8/V4 and i8r.
- Shared settings: `--chunk 4096 --tc 1 --vram-reserve-mib 0 --cpu-threads 5 --kv-grow 1 --kv-grow-step 4096` and the same route-frequency profile.
- Shared environment: `STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 INK2_MX_POOL_FRACTION=0.08`.
- Inputs are prefixes of the saved token IDs under `/tmp/ink2-prefix-3584.ids` and `/tmp/ink2-prefix-7680.ids`.
- Raw stdout/stderr is kept beside this report. `ink2-baseline-*` and `betelgeuse-ink2-*` are the initial pass; `repeat-*` are the interleaved repeat.

An additional repeat was attempted on 2026-11-22 but could not start: CUDA reported no device and `nvidia-smi` could not communicate with the driver. No larger-context speed tests were run as part of this comparison.

After recording these runs, a build-flag mismatch was found: the Strata host sources use strict C++20, while the first Betelgeuse build used GNU C++20. The Betelgeuse target now sets `CXX_EXTENSIONS OFF` and rebuilds as strict C++20. The table predates that correction and must be refreshed when CUDA is available.
