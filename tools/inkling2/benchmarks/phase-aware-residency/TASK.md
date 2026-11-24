# Phase-aware residency experiment

## Audit and control measurements

- [x] Preserve `feat/selective-expert-staging` and its commits/results; do not pursue Phase 2B or GPU staging.
- [x] Create `feat/phase-aware-residency` from `e1538e995` mmap-backed CPU runtime.
- [x] Audit source allocation formulas, lifetimes, expert sizes, KV/chunk interaction, graph invalidation, and current refill path.
- [ ] Capture actual GPU/RAM availability and allocation snapshots when the GPU is free.
- [ ] Build coherent source-review prompts from the fixed base revision; store prompts and IDs under this experiment.
- [ ] Measure 4K vs 8K with identical settings and fixed sliding-ring capacity; record routes, KV, pool counts, and performance.

## Implementation

- [ ] Reuse VMM backing handles across prefill workspace, decode expert pools, and growable KV.
- [ ] Add a fixed ring-capacity override for controlled chunk comparisons, preserving the current default.
- [ ] Accumulate actual prompt route frequencies for post-prefill admission.
- [ ] Record CPU worker latency separately for NVFP4 and MXFP8 during actual inference.
- [ ] Refill only newly available expert slots from mmap, preserving existing residents and accounting for future KV growth.
- [ ] Log reclaimed bytes, transition latency, resident counts, CPU misses, and allocation peaks.
- [ ] Rebuild and run correctness and long-context stability checks.

## Benchmarks and closeout

- [ ] Run real technical/programming prompts at 1K, 4K, 8K, and 16K, with production prompt caching enabled.
- [ ] Compare baseline and new runtime prefill throughput, decode throughput, TTFT, transition latency, RAM/VRAM, and CPU misses.
- [ ] Save commands, logs, prompt source, and a concise results report.
- [ ] Commit the audit, implementation changes, and benchmark report as independent changes.
