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

## Mission 2 - Small-batch DSpark verify

### M1 audit summary

1. M1 found CPU experts plus DSpark were rejected with graph capture because the captured CPU verifier was not considered equivalent to uncaptured execution; capture also required `T == DBLK`, while verify submitted two rows at a time.
2. The old forced `--graph 0` run staged 252.8 MB per verify from the route-only path at about 3 GB/s; this predates the current CPU-batched verifier.
3. Current code uses `k_cache_cpu_batch()` for unique nonresident experts across verify rows and skips expert staging for CPU verify; without CPU verify, it stages only routed misses.
4. Historical acceptance varies by block and path: M1 recorded 0.52-0.72 at one draft in the supplied runs and 0.94 at two drafts, while its later CPU-hybrid DBLK2 record was about 0.51. These are not a controlled `n_max=1` versus `2` comparison, and parity still needs an exact-hash check.

### Task 1 - Graphed `nt=2,3` resident and CPU-miss verify

- [x] Added runtime `--spec-draft-n-max 1|2`. These modes verify 2 or 3 target rows in one call; the default fixed window and plain `nt=1` logic remain on their existing paths.
- [x] Kept nonresident verification experts on `k_cache_cpu_batch()` and added opt-in counters for unique CPU experts, routed rows, and staged bytes.
- [x] Captured verify graphs for 2 rows (`n_max=1`) and 3 rows (`n_max=2`); both generated 512 tokens. Graph logs report `rows=2` and `rows=3` captured.
- [x] Added `INK2_SPEC_DRAFT_N_MAX=1|2` to the direct server backend and rebuilt `bel-inkling2`, `bel-server`, and `bel-cli` in `/home/zyxciss/fluxaeros-data/inkling-m2/build/` after the final KV growth fix.
- [x] Small-batch mode releases the 4K prompt workspace before reserving compact verify buffers. Runtime placement had 1,536/1,536 loop-core experts resident before verify; after `kv_fit`, the pools held 1,583 NVFP4 + 64 MXFP8 slots.
- [x] Runtime counters for both modes show 0.000 core CPU misses and 0.000 MB expert staging per verify. Outer CPU misses averaged 89.786 unique experts / 100.513 routed rows for `n_max=1`, and 125.583 / 151.288 for `n_max=2`.
- [x] Both modes generated 512 greedy tokens matching the plain SHA-256 `7980d86af4291459d792ad069c25b36316371f06ea210dc9bb9fd449f52b397e`; no expert weights were requantized or omitted.
- [x] `kv_fit(p + DBLK)` before drafting fixed the out-of-range draft K/V access observed in the first parity attempt.
- [ ] Pin the nonresident expert source in host RAM. Current logs still report mmap-backed expert files and an 8 MiB `RLIMIT_MEMLOCK`; no pinned coverage is claimed.

### Task 2 - One controlled comparison

- [x] Used the single cgroup comparison session for the plain control and both draft modes. The plain arm completed; neither draft mode reached server readiness. Logs and samples are under `/home/zyxciss/fluxaeros-data/inkling-m2/`.
- [x] The warm plain request processed 16,384 + 512 tokens at 905.84 pp/s / 40.31 tok/s (30.764 s combined). Its SHA-256 `7980d86af4291459d792ad069c25b36316371f06ea210dc9bb9fd449f52b397e` matches M1.
- [x] The one controlled comparison session's candidate attempts exited during pool setup (5.12 GiB, 1,116 NVFP4 + 52 MXFP8, below the pool floor). After the compact-workspace and KV fixes, parity-only requests measured `n_max=1`: 763.68 pp/s / 20.31 tok/s; `n_max=2`: 773.45 pp/s / 18.87 tok/s. These single-request correctness runs are not a controlled paired speed comparison.
- [x] Parity-run verify telemetry: `n_max=1` draft/verify 10.85/62.28 ms, acceptance 0.4839, 1.484 tokens/verify; `n_max=2` 10.86/75.37 ms, acceptance 0.3125, 1.625 tokens/verify. Both reported zero staged bytes and zero core misses.
- [x] Cgroup values were `memory.high=14495514624`, `memory.max=15032385536`, `memory.swap.max=0`; `memory.peak=13054365696` bytes. `memory.events` reported `high=0`, `max=0`, `oom=0`, and `oom_kill=0`. Sampled VRAM peaked at 11,832 MiB during the plain arm; draft startup reached 6,420 MiB before exiting.
- [x] VRAM sampled at 11,830 MiB (`n_max=1`) and 11,832 MiB (`n_max=2`), both below the 11,895 MiB limit. `n_max=1` cgroup peak was `14398156800` bytes with `memory.events.high=0`. The first `n_max=2` parity run recorded peak `14505472000` and 15,968 high events while its sampler repeatedly launched `nvidia-smi` inside the cgroup. An isolated rerun with the observer outside the model cgroup recorded peak `14489419776` bytes and `memory.events.high=0`; max, swap, and OOM events were also zero.
- [x] Rebuilt and validated the draft K/V growth fix with exact output hashes. The single comparison allowance is used; do not run another speed comparison. Do not push.
- [x] Repeated only `n_max=2` parity/resource validation with VRAM monitoring outside the model cgroup. It again generated 512 tokens with the plain hash, captured the 3-row graph, and stayed below both memory limits with zero `memory.events.high` events. This was not a speed comparison.
- [ ] Complete a valid paired plain/DSpark speed comparison and meet the expected decode target. The sole comparison slot was used before the candidate pool fix; parity-only timings show a regression and cannot replace a paired comparison.

### M2 run outcome

The fixed 4K prompt workspace plus the loaded drafter left too little VRAM for expert pools in the only controlled comparison, so it produced a valid warm plain baseline but no DSpark comparison. The compact workspace and draft-KV growth fixes then passed greedy hash parity for both verify sizes. Parity-only decode measured 20.31 tok/s (`n_max=1`) and 18.87 tok/s (`n_max=2`), far below plain's 40.31 tok/s. Both modes met the isolated memory/VRAM envelope with the observer outside the model cgroup; the first instrumented n_max=2 run recorded high events. Graph correctness, hash parity, zero core misses, and zero staging are verified, but host pinning and the expected speed gain are not. The speed-comparison allowance is exhausted, so do not claim a paired DSpark speed result or rerun it.
