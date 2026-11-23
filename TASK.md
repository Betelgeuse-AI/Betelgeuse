# Inkling-2-Mini Mission 1: Looped-Core Residency and DSpark Audit

Date: 2026-11-24
Branch: `betelgeuse-v6.1`

## Goal and constraints

Recreate the warm 16,384-prompt + 512-token direct-checkpoint `bel-server` result, reserve all 1,536 NVFP4 experts in physical layers 6-17 in VRAM, audit host placement and the DSpark verify path, and keep greedy output identical. Preserve work in `~/fluxaeros-data/`; do not use `/tmp` or push. Run at most one baseline-versus-candidate benchmark session.

The supplied warm target was 890.1 pp/s, 40.29 tok/s, 31.09 s combined. Mission notes cite 41.5 tok/s plain decode at 1,981 NVFP4 + 94 MXFP8 slots, and 16.1-18.0 tok/s with an older DSpark path. Treat these as references until identified with a specific build and configuration.

## Completed checkout audit

- [x] Replaced the previous task ledger with this mission plan.
- [x] Rebuilt `bel-server`, `bel-cli`, and `bel-inkling2` from the current source.
- [x] Confirmed the embedded request queue, direct-checkpoint defaults, automatic route-profile selection, and K8/V4 and i8r support.
- [x] Preserved the pre-change `bel-server` and `libllama-server-impl.so` under `/home/zyxciss/fluxaeros-data/inkling-m1/baseline-lib/`.
- [x] Confirmed core-first initial placement and that lazy safetensors mode disables expert promotions.
- [x] Found that prompt-time KV growth can shrink the pool, then `pf_unmap()` refills it before decode. The normal placement order restores loop-core experts first.
- [x] Confirmed the process memlock soft and hard limits are both 8 MiB, effective capabilities are zero, and 13 GiB RAM is available. The engine does not pin the 6,076 MiB source range; no system limit was changed.

## Task 0 - Paired warm 16K reproduction

- [x] Ran the single authorized comparison: preserved server, then rebuilt loop-core server; each served two greedy 16,384 + 512 requests. Prompt IDs, commands, JSON results, logs, memory samples, and placement records are under `/home/zyxciss/fluxaeros-data/inkling-m1/`.
- [x] Both variants generated 512 tokens on both requests. All four generated-text hashes match (`7980d86a...52b397e`).
- [x] Preserved build warm request: 857.35 pp/s, 40.17 tok/s, 31.831 s. Cold request: 778.74 pp/s, 37.42 tok/s, 34.695 s.
- [x] Loop-core build warm request: 505.20 pp/s, 40.35 tok/s, 45.096 s. Cold request: 617.45 pp/s, 37.98 tok/s, 39.991 s.
- [x] Compared with the supplied warm target: baseline decode is close, prefill is 3.7% lower, and combined time is 0.74 s slower. The loop-core candidate preserves decode but loses 41% of warm prefill throughput and adds 13.27 s combined versus the preserved build. It does not reproduce the target.

## Task 1 - Loop-core slot protection and host misses

- [x] Tested a 1,536-slot NVFP4 floor, then removed the floor and phase flag after measurement. The normal post-prompt refill restores the loop core before decode.
- [x] Added opt-in `INK2_MISS_DIAG=1` per-layer counters for plain CPU-expert decode. The candidate reported zero routed misses on layers 6-17 for both 512-token requests.
- [x] The measured floor candidate started at 1,911 NVFP4 + 90 MXFP8 slots and retained 1,546 + 64 during the prompt. The preserved build fell to 1,276 + 64 during prompt growth, then refilled to 1,861 + 90 before decode. Since refill uses the loop-first order, the baseline also had all core experts resident for decode.
- [x] Candidate outer-layer routed misses were 8,878 and 8,808 over 512 generated tokens; per-layer counts are in `loop-core-server.log`. CPU miss execution uses mapped source views and the doorbell path; decode staged bytes were zero.
- [x] Peak sampled VRAM was 11,832 MiB for both variants. Process `VmHWM` was 12,266 MiB baseline and 12,585 MiB candidate. Candidate reported 6,076 MiB of nonresident expert source data; `VmLck` and `VmPin` remained zero.
- [ ] Pin the nonresident expert source in bounded host RAM. Current 8 MiB `RLIMIT_MEMLOCK` cannot pin the 6,076 MiB source; resolve the process limit before implementing or claiming pinned-host coverage.
- [ ] Meet the expected decode gain of 50-60 tok/s while retaining the 16K prefill result. The floor candidate measured 40.35 tok/s and 505.20 pp/s warm, only +0.18 tok/s decode versus baseline and 41% less prefill. The floor was removed.
- [x] Exact output parity and zero loop-core routed misses passed in this comparison.

## Task 2 - Read-only DSpark verify audit

- [x] Traced the current embedded-engine path. `spec_continue()` uses the fixed `INK2_SPEC_BLOCK` draft window, then verifies target rows in chunks of two. The embedded engine does not accept `--spec-draft-n-max`; that flag configures the separate `common/speculative.cpp` path.
- [x] Current CPU verification does not stage full experts. In `moe_b()`, `cpu_verify` selects `k_cache_cpu_batch()` for unique nonresident experts across verify rows. `expert_parts()` returns without staging when `PF.rows_argmax && E.cpu`; `k_cpu_wait()` joins CPU work and copies outputs. The CPU workers read `ExpertParts` from the safetensors mmap.
- [x] If CPU verification is not selected, `PF.verify_route_misses` reads the routed IDs and stages only routed nonresident experts. It does not stage every nonresident expert in a layer. The older 214.4 MB/token forced-DSpark log and comments describing file-backed weight staging predate the current CPU-batched verify path.
- [x] CPU + spec + graph is rejected by the CLI because captured CPU verification is not considered equivalent to the uncaptured path. `forward_prefill()` only captures a verify when `T == DBLK`; the current verifier submits at most two rows per call. A DBLK2 graph1 experiment in the existing DSpark notes diverged from graph0, so graph capture is not cleared for use.
- [x] The current source has no `draft_sample_method` identifier. The generic DSpark path uses `dflash.sample_from_anchor`, `params.n_max`, confidence filtering, and its sampler; the embedded engine uses the anchor plus a fixed DSpark window and greedy target-row comparisons.
- [x] Embedded target feature taps are the hidden states entering visits 5, 23, and 41 (physical layer 5; physical layer 11 on loop pass 2; physical layer 17 on loop pass 3), plus final normalized output. These are consumed by the DSpark feature mixer.
- [x] Existing logs are not a clean comparison of `--spec-draft-n-max 1` vs. `2`: older forced runs show 0.58 accepted drafts/verify at block 8, 0.72 at block 2, and 1.07 at block 3, but incur severe staging cost. Later CPU-hybrid DBLK2 runs show about 0.51 accepted drafts/verify and 30.2 tok/s versus 37.1 plain, with exact output and no staging over 128 tokens. One draft was the best recorded speculative shape, but no speculative setting beat plain decode. No DSpark run was made in this mission.
- [x] Audit findings are read-only; no verify, graph, sampling, or draft behavior was changed.

## Wrap-up

- [x] Updated the benchmark report with the new run, regression, placement, miss counters, RAM limit, and DSpark audit.
- [x] Kept the single comparison session and all benchmark artifacts under `/home/zyxciss/fluxaeros-data/inkling-m1/`.
- [x] Commit the integrated engine and this measured loop-core prototype as a local checkpoint, with the prefill regression and incomplete host pinning recorded as open items. Do not push.
- [ ] Recover warm 16K prefill while keeping all 1,536 loop-core experts resident for decode; the measured core-floor build is not a performance pass.
- [x] Removed the experimental 1,536-slot hard floor and phase flag. The paired data shows no decode gain, while existing post-prompt refill restores loop-core experts first.
- [ ] Pin the bounded nonresident expert source in host RAM after resolving the 8 MiB process memlock limit; do not claim pinned coverage before measuring `VmLck`/`VmPin`.
