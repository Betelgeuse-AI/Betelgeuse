# Inkling-2-Mini optimisation ledger

Branch: `feat/phase-aware-gpu-fetch`

## Working baseline

- Deterministic direct-engine environment in `bel-server`: tuned `INK2_*` values forced unless
  `INK2_ALLOW_TUNING_OVERRIDE` is set; DSpark environment ignored; explicit `--spec 0`.
- CPU Hyper-Threading opt-in `INK2_CPU_HT=1`: `-t 12` gives 10 effective workers on the i5-12400.
- Phase-aware VRAM reclamation at the prefill-to-decode boundary: `pf_unmap()` releases prompt
  workspace and refills the permanent expert pool.
- Mmap-backed CPU expert path retained as the fallback and for unregistered layers.

## Environment change

`/etc/security/limits.d/99-memlock.conf` was added with soft/hard `memlock unlimited` for
`zyxciss`, and the current shell reports `ulimit -l = unlimited`. No other system-wide change was
made. The implementation does not require unlimited memlock; `cudaHostRegister` succeeded above the
old 8 MiB limit in earlier benchmarks. Register only bounded regions.

## Phase-aware GPU fetch

`INK2_GPU_FETCH=1` enables a decode-phase-only GPU fetch path:

- During prefill nothing new is pinned; experts stay in the original `MAP_PRIVATE` mmap and Linux
  keeps the pages reclaimable.
- At the prefill-to-decode boundary, a writable private alias of the checkpoint is mapped and a
  bounded set of route-ranked whole NVFP4 layers is registered with
  `cudaHostRegisterMapped | cudaHostRegisterPortable`.
- No anonymous copy of expert weights is made. The four packed parts are copied from their original
  mmap tensors directly into GPU pool slots by `k_fetch_parts`.
- The decode graph is invalidated and recaptured when registrations are enabled or released.
- Registrations are released at the start of the next substantial prefill.
- The recurrent core (layers 6-17) is never evicted by the GPU-fetch cache.
- MXFP8 layers stay on the CPU path in this first implementation.
- `INK2_GPU_FETCH_MIB` sets the registration budget (default 640 MiB, enough for one 128-expert
  NVFP4 layer). `INK2_GPU_FETCH_LAYERS` caps the number of registered layers (default 1). Larger
  multi-layer configurations were not stable and are capped until the graph interaction is fixed.

Flags:

```sh
INK2_GPU_FETCH=1 INK2_GPU_FETCH_MIB=640 INK2_GPU_FETCH_LAYERS=1 \
STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 \
INK2_OFF_MARGIN_MIB=64 INK2_MX_POOL_FRACTION=0.08
```

Bel server/CLI inherit `INK2_GPU_FETCH`; no server flag is needed.

## Measured results

Standalone `bel-inkling2`, 16,896 context, i8r, chunk 4096, graph 1, spec 0, 5 CPU workers,
256K-decode route profile, cgroup 14 GB / no swap:

| Prompt | Generated | CPU path | GPU fetch, 640 MiB |
| --- | ---: | ---: | ---: |
| Python-cache prompt, 83 tokens | 2048 | ~26.5 tok/s | **33.4 tok/s** |
| Source-review archived 16K IDs | 512 | ~800 pp/s / ~40 tok/s | 716.8 pp/s / 35.6 tok/s |

`bel-server`, same context and profile, 16K source-review prompt:

| Request | Prefill | Decode |
| --- | ---: | ---: |
| cold | 696.07 pp/s | 37.22 tok/s |
| warm | 863.41 pp/s | 39.46 tok/s |

`bel-server`, 83-token Python prompt, 512 generated:

| Request | Prefill | Decode |
| --- | ---: | ---: |
| cold | 42.63 pp/s | 28.94 tok/s |
| warm | 44.15 pp/s | 31.32 tok/s |

The phase-aware path preserves the coherent 16K prefill far better than the earlier Strata
8.19 GiB pinned arena, which fell to about 259 pp/s. Decode on the short prompt improves, but is
below the 48.3 tok/s Strata result because that implementation pinned every evictable expert and
therefore fetched from a much larger registered set.

## Correctness

- Every routed expert is still executed; no expert is skipped and no requantisation is used.
- CPU and GPU expert kernels have different summation orders. On the Python prompt their argmax can
  diverge at near ties, as documented for Strata. This is not a shape or routing change.
- Exact greedy-hash parity against the CPU path is therefore not expected on near-tie prompts.
  Use KL/logit checks for numerical validation.

## Budgets tested

- 512 MiB: no whole NVFP4 layer fits; the path falls back to the CPU.
- 640 MiB: one NVFP4 layer, 128 experts, 546 MiB registered; stable.
- 2 GiB: three layers were registered but graph replay hit an illegal memory access; now capped to
  one layer by default. Do not raise `INK2_GPU_FETCH_LAYERS` until that is fixed.

## Open work

- Make multi-layer registration graph-safe.
- Include MXFP8 layers after validating their four-part fetch into the MX pool.
- Add per-expert reuse/eviction counters and PCIe-byte counters to the engine log.
- Test 32K, 64K, 128K and 262K contexts with generation headroom.
- Compare against the best HT CPU baseline (10 workers) at every context.
- Add TTFT breakdown for registration and phase transition.

## Multi-layer and selective fetch correction (2026-11-25)

### Root cause of the illegal access

The multi-layer crash was not a slot-index or tensor-offset bug. `k_cache_plan` was filling the
shared four-part fetch lists from a source table that only contained registered experts. Layers
whose experts were not registered produced null source pointers, and `k_fetch_parts` dereferenced
them during graph replay. Under graph capture the error surfaced at `cudaGraphLaunch`; with
`--graph 0` and `CUDA_LAUNCH_BLOCKING=1` it surfaced at the following D2D copy.

Fixes:

- Added a device eligibility mask (`INK2_GPU_FETCH` only marks registered experts).
- `k_cache_cpu` skips eligible experts, so they are no longer computed twice.
- `k_cache_plan` fetches only eligible misses, so no null source pointer reaches `k_fetch_parts`.
- `k_fetch_parts` also guards null sources defensively.
- Added a guard for an empty `expert_freq` vector (a crash when no route profile was given).
- On KV-growth pool shrinkage, registrations are released and execution falls back to mmap CPU;
  this keeps 16K with `--kv-grow` stable.
- `compute-sanitizer --tool memcheck` could not run the host-registered file mappings on this
  driver (it exited before model load), so the diagnosis used `--graph 0`,
  `CUDA_LAUNCH_BLOCKING=1`, gdb backtraces, and staged instrumentation.

### Selective registration

Registration is now per expert, not per layer. Candidates are ranked by actual current-prompt
routes plus decode counts, with the route profile as a weak tie-breaker. Only nonresident experts
that fit the budget are registered. Their page-aligned tensor parts are coalesced into a small set
of `cudaHostRegisterMapped | cudaHostRegisterPortable` ranges. All other experts stay on the
mmap-backed CPU path. Permanently resident experts are excluded from registration.

### MXFP8 support

MXFP8 layers 20-23 are now eligible. Their four packed parts use the same fetch path with their
own byte sizes and the MXFP8 GPU pool. `INK2_GPU_FETCH_NV=0` and `INK2_GPU_FETCH_MX=0` select
NVFP4-only or MXFP8-only experiments.

### Instrumentation

`expert fetch:` now also reports a four-counter line:
`fetches`, `slot hits`, `evictions`, `cpu-eligible misses`.

### Corrected results

Standalone, 16,896 context, i8r, chunk 4096, graph 1, spec 0, 5 CPU workers, 256K route profile:

| Configuration | Prefill | Decode / 512 | Decode / 2048 |
| --- | ---: | ---: | ---: |
| CPU baseline, Python 83-token prompt | 21.3 tok/s | 25.7 tok/s | ~26.5 tok/s |
| GPU fetch, 512 MiB, Python prompt | 24.2 tok/s | 31.4 tok/s | - |
| GPU fetch, 1024 MiB, Python prompt | 15.8 tok/s | 28.9 tok/s | - |
| GPU fetch, 2048 MiB mixed, Python prompt | 19.3 tok/s | - | 30.0 tok/s |
| GPU fetch, 640 MiB, archived 16K prompt | 747.1 pp/s | 37.1 tok/s | - |
| CPU baseline, archived 16K prompt | ~800 pp/s | ~40 tok/s | - |

`bel-server`, GPU fetch 640 MiB, 16K source-review prompt: cold 702.15 pp/s / 39.36 tok/s,
warm 867.11 pp/s / 40.54 tok/s. Repeated requests exercised release at the next prefill and
re-registration at decode.

### Format comparison on the Python prompt, 512 generated

| Mode | Registered experts | Decode | Fetch stats (fetch/hit/evict/cpu-miss) |
| --- | ---: | ---: | --- |
| NVFP4 only, 2048 MiB | 479 | 21.5 tok/s | 359 / 31556 / 359 / 853 |
| MXFP8 only, 2048 MiB | 261 | 25.1 tok/s | 1343 / 6712 / 1343 / 8329 |
| Mixed, 2048 MiB | 341 | 22.3 tok/s | 310 / 28543 / 310 / 20299 |

At this budget the route-ranked 512 MiB mixed setting was the best short-prompt decode (31.4
tok/s). Larger budgets increased registration latency without improving decode.

### Budget and TTFT observations

| Budget | Registered | Registration / phase transition | Decode / 512 |
| ---: | ---: | ---: | ---: |
| 256 MiB | 33 | 481.9 ms | 26.1 tok/s |
| 512 MiB | 67 | 617.3 ms | 29.4 tok/s |
| 640 MiB | 85 | 523.4 ms | 23.3 tok/s |
| 1024 MiB | 156 | 859.7 ms | 30.9 tok/s |
| 2048 MiB | 341 | 1601.6 ms | 22.3 tok/s |

Registration latency is charged inside the prefill-to-decode transition, not prompt processing.
For short prompts a large budget does not pay back its transition cost.

### Open items

- Long-context 32K/64K/128K/262K still need a full sweep.
- Fetch stats show most misses still use CPU; registration selection needs decode-history feedback
  across multiple turns in a persistent server.
- Reuse distance and exposed CPU/GPU dependency timing are not yet logged separately.

## Correction: retire archived synthetic word-salad benchmark (2026-11-25)

The archived 16K synthetic word-salad prompt (`~/fluxaeros-data/inkling-m2/prompt-16384.ids`) is
**invalid for real-world performance comparisons**. All historical results from it, including the
905.84 pp/s / 40.31 tok/s controlled run and the 40 tok/s Betelgeuse warm results, are marked
invalid. It is not to be run again. Accepted workloads are actual Python programming tasks and the
coherent 16,338-token source-review prompt
(`tools/inkling2/benchmarks/phase-aware-residency/prompts/source-review-prompt.txt`).

## Direct reproduction attempt: Strata mechanism in Bel

The selective/scoring/k_fetch_parts machinery was removed. Bel now uses the original successful
Strata mechanism: after prefill it packs every evictable outer expert into one pinned host arena,
publishes device pointers through `E.xsrc_d`, and uses the original `k_fetch` GPU cache path. The
arena is released before the next substantial prefill. The recurrent core stays in VRAM. CPU
mmap fallback remains for any configuration where the arena cannot be allocated or registers are
released.

Identical settings for both engines: Python-cache prompt, 83 tokens + 2048 generated, i8r,
chunk 4096, graph 1, spec 0, `--vram-reserve-mib 0`, `--cpu-threads 1`, KV growth 4096,
256K-decode route profile, cgroup 14 GB / no swap.

| Engine | Prefill | Decode | Fetch counters (fetch/hit/evict/cpu-miss) | H2D/token |
| --- | ---: | ---: | --- | ---: |
| Strata `feat-latest` original GPU fetch | 12.8 tok/s | **48.3 tok/s** | not exposed by that binary | 8.4 MB |
| Bel CPU fallback | 27.5 tok/s | 18.4 tok/s | none | 0.0 MB |
| Bel packed GPU fetch, first attempt | 7.0 tok/s | 20.1 tok/s | 30503 / 166105 / 30503 / 0 | 121.3 MB |
| Bel packed GPU fetch, loop-core protected | 7.4 tok/s | 36.3 tok/s | 52006 / 144602 / 52006 / 0 | 163.6 MB |

Bel did **not** reproduce 48.3 tok/s. Loop-core eviction protection raised decode from 20.1 to
36.3 tok/s, but the Bel cache plan still fetches about 25 experts/token, versus about one to two in
the Strata result. Enabling the original promotion counters did not change it. This remains the
single largest unresolved difference.

## Coherent source-review prompt (16,338 tokens + 512)

| Configuration | Prefill | Decode |
| --- | ---: | ---: |
| Bel CPU fallback, 5 workers | 687.8 pp/s | 29.0 tok/s |
| Bel packed GPU fetch | - | failed after KV growth; CPU fallback required |

GPU fetching on the coherent long prompt becomes unstable when `--kv-grow` shrinks the MX pool
after the prefill-to-decode boundary. Registrations are released on shrink, but a graph replay
still hit an illegal memory access. Use the CPU fallback for this workload until fixed. The CPU
fallback preserves correctness and the existing prefill rate.

## State

- The original Strata 48.3 tok/s result is reproduced on the Strata binary, not yet in Bel.
- Archived synthetic word-salad numbers are invalid and must not be used as a baseline.
- No quantisation, expert skipping, speculative decoding or scheduler was added.
- All routed experts are still executed; the CPU fallback remains intact.

## Routing-replay isolation (2026-11-25)

Recorded Strata's decode routing for the Python prompt and replayed it through Bel with
`--force-routing`, keeping every other setting identical.

| Engine | Decode | H2D/token | Fetches/token |
| --- | ---: | ---: | ---: |
| Strata GPU fetch, ordinary decode | 47.6 tok/s | 17.3 MB | ~2 |
| Bel GPU fetch, forced identical routing | **47.8 tok/s** | 15.2 MB | 1121/512 = 2.2 |

This proves the Bel cache, pinned arena, fetch counters and slot reuse are equivalent to Strata.
The 36.2 tok/s ordinary Bel result is caused by a different generated token sequence and therefore
different expert routing, not by cache thrashing or a different H2D mechanism. The counter semantics
match within workload differences.

Strata's refill implementation was ported into Bel directly; it did not change the ordinary decode
result, confirming refill policy is not the discrepancy.

## KV-growth illegal access

Changed pool shrinkage to suspend GPU-fetch state (clear masks, invalidate graphs, keep the pinned
arena) instead of freeing the arena mid-resize. The 16,338-token source-review prompt still hits an
illegal memory access after `--kv-grow` shrinks the MX pool from 90 to 84 slots. This remains an
open correctness issue; the CPU fallback is the stable path for that workload. No permanent
disable was added.

## Current state

- 48 tok/s class decode is reproduced in Bel when expert routing is identical (47.8 tok/s).
- Ordinary Bel decode remains 36.2 tok/s because generated tokens diverge from Strata at near-ties.
- Do not add further scheduling or kernel work to chase this; the next step is numeric/logit parity
  of the attention/GEMV path that decides those tokens.

## Natural-routing multi-task benchmark (2026-11-25)

Five distinct real programming/reasoning tasks, 512 generated tokens each, natural generation,
identical prompts and settings for both engines (i8r, chunk 4096, graph 1, spec 0, 1 CPU thread,
KV growth, 256K route profile, cgroup 14 GB). No forced routing and no archived word-salad prompt.

| Task | Strata decode | Strata H2D/token | Bel decode | Bel H2D/token | Bel fetches/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU + TTL implementation | 49.2 tok/s | 2.4 MB | 36.3 tok/s | 172.5 MB | 26.4 |
| Directed cycle detection | 47.8 tok/s | 17.2 MB | 37.1 tok/s | 160.0 MB | 24.4 |
| Race-condition reasoning | 47.8 tok/s | 17.3 MB | 39.5 tok/s | 121.2 MB | 18.5 |
| Token-bucket rate limiter | 36.8 tok/s | 158.4 MB | 36.3 tok/s | 171.7 MB | 26.1 |
| Serializability reasoning | 42.8 tok/s | 72.5 MB | 37.4 tok/s | 155.4 MB | 23.2 |

Medians: Strata 47.8 tok/s, Bel 37.1 tok/s. Strata itself ranges from 36.8 to 49.2 tok/s
depending on natural routing, so 48-50 tok/s is not universal even on Strata. Bel does not reach
the 48 tok/s class on any natural prompt in this set.

`INK2_GPU_FETCH_EARLY=1` was added to pin the packed arena before prefill, matching Strata's
startup placement. It did not change Bel's natural first token or decode rate. Porting Strata's
per-query-head `k_attn_partial` into Bel also made no difference. The first divergence is not in
those two paths.

Forced routing remains a diagnostic only: with identical expert routes Bel reached 47.8 tok/s and
about 2.2 fetches/token, proving the cache implementation is equivalent. The natural gap is driven
by generated-token divergence and consequent cold routing. This is not yet resolved.

## KV-growth status

Fetch state is suspended (masks cleared, graphs invalidated, arena retained) when a pool shrinks,
but the coherent 16K source-review prompt still hits an illegal memory access after the MX pool
shrinks from 90 to 84 slots. It has not been fixed. CPU fallback remains stable for that workload.
No permanent disable was added.

## Do not use for accepted measurements

- forced routing;
- archived word-salad prompt.

## Fused GQA history (2026-11-25)

`git log -S'k_attn_partial_gqa'` shows it entered in `970dae908`, bundled with the loop-core
placement work. The isolated same-binary check recorded in
`benchmarks/gqa-decode-2026-11-23/RESULTS.md` measured +19.9% decode at 262K, +2.6% at 16K K8/V4,
and +0.6% at 16K i8r, with no demonstrated correctness regression. It was a diagnostic mistake to
replace it with Strata's per-head kernel during the numerical investigation; it has been restored.

## Four-configuration prefill check

Identical input tokens, `--dump-logits`, same flags.

| Configuration | Last-position argmax |
| --- | ---: |
| Strata CPU-only | 114489 |
| Bel CPU-only | 114489 |
| Strata GPU-fetch | 115467 |
| Bel GPU-fetch | 114489 |

CPU-only logits are bit-identical across all 83 prompt positions. The first divergence appears only
when GPU fetch is enabled in Strata: its packed pinned expert source changes prefill logits, while
Bel's post-prefill pinned arena does not. `INK2_GPU_FETCH_EARLY=1` made Bel pin before prefill but
did not reproduce Strata's first-token logits. This is the exact remaining discrepancy; it is in
prefill, not decode attention. No expert is skipped and no weight is requantised in either engine.

## KV-growth crash mitigation

A captured decode graph keeps the old pool-slot launch arguments. When `--kv-grow` shrinks a pool,
Bel now invalidates the graphs and switches that process to the uncaptured decode path. GPU fetch
state is suspended (masks cleared, graphs invalidated, arena retained until the next prefill) rather
than freed mid-resize. The coherent 16K source-review prompt now completes with GPU fetch enabled
after the MX pool shrinks from 90 to 84 slots. This is a runtime fallback for the shrink event, not
a permanent GPU-fetch disable.

## Rerun of five natural workloads (after restoring GQA)

Identical natural generation, 512 tokens, i8r, chunk 4096, 1 CPU thread, 256K route profile.

| Task | Strata decode | Strata H2D/token | Bel decode | Bel H2D/token | Bel fetches/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU + TTL | 49.2 | 2.4 MB | 35.9 | 172.5 MB | 26.4 |
| Directed cycle | 47.8 | 17.2 MB | 36.7 | 160.0 MB | 24.4 |
| Race condition | 47.7 | 17.3 MB | 39.2 | 121.2 MB | 18.5 |
| Rate limiter | 36.9 | 158.4 MB | 36.0 | 171.7 MB | 26.1 |
| Serializability | 42.8 | 72.5 MB | 37.0 | 155.4 MB | 23.2 |

Median: Strata 47.7 tok/s, Bel 36.7 tok/s. The generalisable 48-50 tok/s target is not met.
The remaining work is the prefill numerical discrepancy, not cache policy or fused GQA.

## Exact source of the first-token logit difference (2026-11-25)

The four-way check was repeated with current binaries: Strata CPU 114489, Bel CPU 114489,
Strata GPU-fetch 115467, Bel GPU-fetch 114489.

The difference is **not** an expert kernel, router, accumulation order or residual bug. It is pool
sizing. Strata allocates its pinned expert arena inside `expert_memory()` before the offload pool
sizes are computed. That allocation reserves CUDA device-address mapping resources and reduces free
VRAM from 8.61 GiB to 8.59 GiB. The expert pools change from **1885 NVFP4 + 89 MXFP8** slots to
**1882 NVFP4 + 89 MXFP8**. Three fewer NVFP4 slots change which experts are resident at the start
of prefill, so router logits and the first generated token change. Bel pins after pool setup, so its
pool stays at 1885 + 89 and its logits match the CPU-only path.

This was confirmed by the `offload:` lines in the two dump-logit runs. No expert is skipped and no
weight is modified. Strata's behaviour is internally consistent but is a residency side effect of
allocation order, not a different numerical implementation.

A native reference comparison could not be completed: `ink2ref` needs all 24 layers resident in
VRAM, about 14.6 GiB, and exits with CUDA OOM on this 12 GB card. CPU-only Strata and Bel are
bit-identical on all 83 prompt positions.

## Proper KV-shrink fix

Replaced the graph-off/suspend workaround with `gpu_fetch_rebuild()`:

- after `k_pool_evict` and unmap, copy the device pointer tables back into `E.htabA`/`E.htabB`;
- destroy and recapture the decode graphs;
- keep `E.gpu_fetch_active`, the pinned arena, the source pointer table and the eligibility mask
  intact.

GPU fetching remains operational after KV growth. Verified with graph capture enabled on the
coherent 16,338-token source-review prompt: it completed with 143 fetches, 241 slot hits, 143
evictions and no illegal access. CPU fallback remains available but is no longer forced by a shrink.

## Final five-workload rerun

Natural generation, 512 tokens, identical prompts and settings.

| Task | Strata decode | Strata H2D/token | Bel decode | Bel H2D/token | Bel fetches/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU + TTL | 49.3 | 2.4 MB | 35.9 | 172.5 MB | 26.4 |
| Directed cycle | 47.8 | 17.2 MB | 36.6 | 160.0 MB | 24.4 |
| Race condition | 47.7 | 17.3 MB | 39.1 | 121.2 MB | 18.5 |
| Rate limiter | 36.7 | 158.4 MB | 36.1 | 171.7 MB | 26.1 |
| Serializability | 42.8 | 72.5 MB | 37.0 | 155.4 MB | 23.2 |

Median: Strata 47.7 tok/s, Bel 36.6 tok/s. The 48-50 tok/s general target is still not met by
Bel. The remaining difference is the residency side effect from allocation order, which changes
natural generated tokens and routing. No further architecture change was introduced.

## Independent root-cause verification and Bel optimisation (2026-11-26)

### Root cause of Strata's apparent 47.7 tok/s result

Inspection of both generated token logs and `feat-latest/engine/src/inkling2/ink2_engine.cu` shows
that the three-slot pool difference (1882 vs 1885 NVFP4 slots) was not the cause of Strata's speed
or prefill logit divergence:

- All five Strata `final3-*-strata.log` decode outputs are degenerate loops (`"Sample "` x510 on
  `lru_ttl`, `"1_ 1_ ..."` on `directed_cycle`, `"3. 3. ..."` on `race_condition`, verbatim prompt
  echo on `rate_limiter`, `"# 1\n# 2\n# 3..."` on `serializability`).
- In Strata `feat-latest`, `load()` calls `expert_memory()` before the layer loop populates
  `E.expert_parts`, leaving the 8.19 GiB pinned arena zero-initialized. Warmup then loads zeroed
  experts into VRAM and decode misses fetch all-zero weights.
- Under controlled parity (`task-lru_ttl.ids`, 128 tokens, `INK2_GPU_FETCH=0` on Strata vs
  `INK2_GPU_FETCH=0/1` on Bel at matched pool reserve), all three produced `max_diff = 0.0` across
  all 83 prompt logit positions and `128/128` identical decode tokens.

### Hardware ceiling and Bel optimisations implemented

Measured limits on this hardware (RTX 3060 12 GB, ~366 GB/s VRAM, i5-12400, 16 GB single-channel
DDR4, ~23.0-23.5 GB/s H2D PCIe bandwidth):

- Zero-miss GPU decode floor on `bel-inkling2` (diagnostic `zero_miss2.routes.bin`): `49.5 tok/s`
  (`20.20 ms/tok`), `50.0 tok/s` (`20.02 ms/tok`) with `INK2_RPW=8 INK2_GEMV_MINB=56`.
- H2D transfer cost: `~23.0 GB/s` via `k_fetch` (`195 us` per 4.48 MB NVFP4 expert, `356 us` per
  8.21 MB MXFP8 expert); Belady OPT lower bound on `lru_ttl` routes is `~70-71 MB/tok`.

Changes committed to [tools/inkling2/ink2_engine.cu](tools/inkling2/ink2_engine.cu) and
[tools/server/server-context.cpp](tools/server/server-context.cpp):

- SM-aware GEMV tuning (`g_gemv_min_blocks = 2 * sm_count`, `g_rpw = 8` for `sm_count <= 32` unless
  overridden by `INK2_GEMV_MINB` / `INK2_RPW`).
- Two-stage GPU greedy argmax (`k_logits_argmax_partial` + `kd_argmax_final`) to transfer 4 bytes
  instead of `VOCAB * 4` bytes (~800 KB) per greedy decode token.
- Decayed-LFU victim selection (`0.97` per-token decay via `k_decay`, `INK2_PLACE=freq` default) in
  `k_cache_plan`, with active-call slot pinning and descending initial slot stamp tiebreak.
- Null-source guard in `k_promote_plan` / `promote_args` before post-prefill `gpu_fetch_enable()`,
  and skipping empty CPU promotion/ring kernels on GPU-fetch layers when loop core fits in VRAM.
- Overlapped `k_fetch` (`E.fetch_stream`, default `INK2_FETCH_BLOCKS=2` CTAs per copy to limit SM
  contention) running concurrently with Pass-1 resident-hit `k_gemv_*_sel`, followed by Pass-2 misses.
- Updated default `INK2_MX_POOL_FRACTION` from `0.20`/`0.08` to `0.15`.

Verified results (`bel-inkling2`, `--kv i8r --kv-grow 1 --kv-grow-step 4096 --chunk 4096 --tc 1 --graph 1 --cpu-threads 5 --spec 0`, `INK2_GPU_FETCH=1`, default tuning, 512 tokens, all coherent outputs and `128/128` CPU parity verified on `task-lru_ttl.ids`):

| Task | Baseline Bel decode | Updated Bel decode | Updated Bel H2D/token |
| --- | ---: | ---: | ---: |
| LRU + TTL | 35.9 tok/s | 40.7 tok/s | 133.4 MB |
| Directed cycle | 36.6 tok/s | 40.6 tok/s | 135.4 MB |
| Race condition | 39.1 tok/s | 42.2 tok/s | 110.0 MB |
| Rate limiter | 36.1 tok/s | 40.8 tok/s | 132.0 MB |
| Serializability | 37.0 tok/s | 41.7 tok/s | 118.7 MB |

Median decode improved from `36.6 tok/s` to `40.8 tok/s` (peak `42.2 tok/s`), and coherent 16K
`context-16384.ids` (`--kv-grow 1 --ctx 32768`) completed at `522.8 pp/s` prefill / `38.8 tok/s` decode.
The remaining gap to `48-50 tok/s` is set by the single-channel DDR4 PCIe bandwidth (`23.0 GB/s`)
at the online cache miss rate (`~110-135 MB/tok`).

**Invalidation notice**: All Strata GPU-fetch performance numbers recorded above from `feat-latest`
are invalid because of the zero-filled 8.19 GiB expert arena bug and resulting degenerate loops.
Commit `39c6ca980` (`40.8 tok/s` 5-task median, `38.8 tok/s` at 16K) is the verified baseline.

## CUPTI critical-path breakdown and compute optimisations (2026-11-26)

### CUPTI profile of `39c6ca980`

Hardware verified via `cudaGetDeviceProperties`: RTX 3060, 28 SMs, 7501 MHz memClk, 192-bit bus,
**360.0 GB/s** theoretical peak DRAM bandwidth (`~5.525 GB/tok` model weights -> `~15.35 ms/tok`
theoretical weight-read floor before attention/convs/residuals/launch gaps).

CUPTI Activity trace (`cupti_lru64.tsv`, 63 steady decode tokens on `task-lru_ttl.ids`, `155.3 MB/tok` H2D):

- Total GPU active union: `24.905 ms/tok` (`19.868 ms` compute union + `5.037 ms` exposed fetch;
  `6.507 ms` raw `k_fetch` with `1.470 ms` overlapped behind Pass-1 `k_gemv` hits).
- Kernel breakdown of `39c6ca980` (`1015` graph nodes+memcpys/tok):
  - `moe_nv` (`k_gemv_nvfp4_sel<2>`): `5.677 ms` (`3.628 ms` gate_up + `2.049 ms` down; 52 calls)
  - Dense BF16 projections (`proj` + `qkv`): `5.529 ms` (`2.313 + 2.285 + 0.931 ms`; `80-88 regs`,
    up to `27.6 KB` smem -> 2-3 CTAs/SM occupancy)
  - `unembed` (`k_gemv_bf16<8>`): `2.605 ms` (`353.9 GB/s`)
  - `wo` (`k_gemv_bf16<4>`, 48 calls, 72 CTAs): `2.063 ms`
  - `attention` (`k_attn_prep` / `k_attn_partial_gqa` / `k_attn_merge` / `k_attn_tail`): `1.451 ms`
    short context, rising to `2.592 ms` at 16K context (`21.298 ms` 16K compute union)
  - `moe_mx` (`k_gemv_mxfp8_sel<2>`): `1.026 ms` (`0.649 + 0.377 ms`, `9.2 KB` / `4.6 KB` smem,
    redundant per-CTA `swiglu1`)
  - `k_cache_cpu` + `k_cpu_wait` still executing on all 36 loop-core calls: `0.226 ms` (`72` launches)

### Implemented compute & host staging improvements

1. **Zero-smem `__ldg` BF16 GEMVs + compact pointer arithmetic**: Replaced `shared__ float xs[]` staging
   and `w[R]` pointer arrays in `gemv_bf16_dot`, `k_gemv_bf16`, `k_gemv_bf16_qkv`, and `k_gemv_bf16_gate`
   with direct `__ldg(const float4*)` reads and a single compact base pointer; launched `wo`-sized
   matrices (`n <= 4096`) with 128 threads (`144` CTAs across 28 SMs). `unembed` reached `2.531 ms`
   (`364 GB/s`), `proj+qkv` dropped `5.529 -> 5.340 ms`, `wo` dropped `2.063 -> 2.009 ms`.
2. **Half-warp NVFP4 `down` GEMV (`g_nv_h16`, `16` lanes/row, `128` threads)**: `K=1152` (`144` `uint32_t`
   words/row) leaves 50% of 32-lane warp inactive on iteration 5; 16 lanes run `9` full iterations
   (`2.049 -> 1.940 ms` in CUPTI).
3. **Zero-smem fused-ACT MXFP8 (`k_gemv_mxfp8_sel`)**: Computes SwiGLU once in `gate_up` into `E.act`
   and reads `x`/`E.act` via `__ldg` (`0` smem), dropping MXFP8 compute `1.026 -> 0.849 ms`.
4. **Loop-core CPU ring skip & pinned host buffers**: Gated `expert_cpu_ring` + `k_cpu_wait` on
   `cpu_active_for_decode() && !gf` (cutting `1015 -> 943` launches/tok) and pre-allocated pinned
   `E.h_emb`, `E.h_pos`, and `E.h_argmax`.

Final CUPTI (`cupti_final2_lru64.tsv`): compute union dropped `19.868 -> 19.026 ms/tok` (`-0.842 ms/tok`).

### Updated benchmark results (512 tokens, default settings, `INK2_GPU_FETCH=1`)

All generated outputs remain coherent; `task-lru_ttl.ids` matches CPU reference `128/128` tokens.

| Task | `39c6ca980` baseline | Updated decode | H2D/token |
| --- | ---: | ---: | ---: |
| LRU + TTL | 40.7 tok/s | **41.8 tok/s** | 133.6 MB |
| Directed cycle | 40.6 tok/s | **41.9 tok/s** | 132.3 MB |
| Race condition | 42.2 tok/s | **43.9 tok/s** | 103.6 MB |
| Rate limiter | 40.8 tok/s | **41.6 tok/s** | 135.9 MB |
| Serializability | 41.7 tok/s | **42.9 tok/s** | 118.1 MB |
| 16K (`context-16384.ids`, `--n 128`) | 38.8 tok/s | **40.3 tok/s** | 122.8 MB |

- **5-task median**: `40.8 -> 41.9 tok/s` (mean `42.42 tok/s`, peak `43.9 tok/s`).
- **Coherent 16K prefill / decode**: `530.0 pp/s` prefill, `40.3 tok/s` decode (`--kv-grow 1 --ctx 32768`).
- **Physical ceiling note**: With GPU compute kernels now running at `93-101%` of RTX 3060 `360.0 GB/s`
  peak DRAM (`19.03 ms` compute union) and `23.0 GB/s` single-channel DDR4 PCIe H2D (`~5.2 ms` exposed),
  reaching `45-50 tok/s` under natural routing on this hardware requires either dual-channel system RAM
  (~`2x` PCIe H2D bandwidth) or a major reduction in outer-layer miss bytes beyond standard decayed-LFU.

## Phase 4: Cache Balance, Priority Overlap, and 32K-262K Scaling (after `5218f01ae`)

### Diagnosis of Transfer Traffic & Overlap Ceilings

1. **Belady OPT vs online decayed-LFU (`route_*_512.bin`, 5 real workloads)**:
   At 0K evictable slots (`319 NV + 169 MX`), Belady OPT transfers `63.3 MB/tok` mean (`58.5-68.4 MB/tok`)
   vs `127.7 MB/tok` for un-capped `dLFU(0.97)` on the extracted 512-token traces. Time-shuffling tests
   confirmed ~85% of the OPT gap comes from clairvoyance over high-entropy tail picks rather than missed
   local temporal patterns. Capping `cnt <= 4.0` with `decay = 0.96` (`125.2 MB/tok` on trace sim) clears
   stale phase-hot experts faster than uncapped decay (`~34` vs `~117` tokens).
2. **Pool split at long contexts**: Sizing `MXFP8` slots as a fixed `0.15` fraction of total VRAM
   (including the `6.88 GB` pinned loop core) starved the evictable NVFP4 pool as KV grew (`319 -> 202` at
   32K, `-> 95` at 64K, `-> <1536` total NV crash at 128K). Sizing `MX` to `0.48` of *evictable* headroom
   (`evict_budget = budget - 1536 * NV_BYTES`) improves 32K/64K transfer bytes and keeps `128K k8v4` in
   full GPU-fetch (`1677 NV + 70 MX`).
3. **Transfer overlap ceiling (`pcie_bench` / `overlap_bench3` / `ddr4_honest` / CUPTI)**:
   Hardware `cudaMemcpyAsync` H2D ceiling is `23.45 GB/s`; `k_fetch` hits `23.23 GB/s` and `split_fetch`
   already hides `1.35 ms/tok` (~100% of outer-layer hit GEMVs), leaving `~4.87 ms/tok` exposed inside `moe()`.
   Cold 512 MB DDR4 CPU+GPU contention tests confirmed single-channel DDR4 saturates under combined CPU
   read + PCIe H2D (`586.4 us` for 2 GPU + 1 CPU vs `601.6 us` for 3 GPU, before CPU compute/bell sync).

### Changes implemented (`tools/inkling2/ink2_engine.cu`, `tools/server/server-context.cpp`)

1. **Cache & evictable-headroom pool allocation**: Capped `c.cnt[key]` at `4.0f` (`k_cache_plan`,
   `k_cache_cpu`), lowered default `g_decay` from `0.97f -> 0.96f`, and sized `E.nslot_mx` / `E.mx_target`
   using shared `mx_slots_for_budget()` (`0.48` of evictable budget unless `INK2_MX_POOL_FRACTION` is set;
   dropped hardcoded `0.15` override in `server-context.cpp`).
2. **High-priority fetch stream & streaming stores**: Created `E.fetch_stream` with highest stream priority
   (`cudaDeviceGetStreamPriorityRange`) and used `__stwt` in `k_fetch` to avoid L2 cache thrashing.
3. **128K/262K `< 1536` NV pool crash fix**: Guarded `best == ~0ull` out-of-bounds slot indexing in both
   `k_cache_plan` and `k_promote_plan`, disabled NV `gpu_fetch` / `g_promote_nv` cleanly when
   `E.nslot_nv < LOOP_NV_SLOTS + FETCH_MAX` (falling back to CPU for NV/loop misses while keeping MX on
   GPU-fetch), and updated `gpu_fetch_rebuild()`.

### Final benchmark results (`bel-inkling2`, default settings, `INK2_GPU_FETCH=1`, `--spec 0`)

All generated responses decoded cleanly and remained coherent; `task-lru_ttl.ids` matches CPU `128/128`.

| Workload | `5218f01ae` | Updated decode | H2D/token | Pools (`NV + MX`) |
| --- | ---: | ---: | ---: | --- |
| LRU + TTL (`512` tok) | 41.8 tok/s | **41.6 tok/s** | 135.5 MB | `1864 + 164` |
| Directed cycle (`512` tok) | 41.9 tok/s | **42.2 tok/s** | 127.9 MB | `1864 + 164` |
| Race condition (`512` tok) | 43.9 tok/s | **44.4 tok/s** | 97.6 MB | `1864 + 164` |
| Rate limiter (`512` tok) | 41.6 tok/s | **41.6 tok/s** | 136.8 MB | `1864 + 164` |
| Serializability (`512` tok) | 42.9 tok/s | **43.1 tok/s** | 116.0 MB | `1864 + 164` |
| 16K `i8r` (`16338` prompt, `128` dec) | 40.3 tok/s | **41.1 tok/s** | 113.4 MB | `1840 + 152` |
| 32K `i8r` (`32750` prompt, `128` dec) | 36.2 tok/s | **37.8 tok/s** | 143.5 MB | `1808 + 137` |
| 64K `i8r` (`65443` prompt, `128` dec) | 30.8 tok/s | **32.8 tok/s** | 194.6 MB | `1747 + 105` |
| 128K `i8r` (`129510` prompt, `128` dec) | crashed (token 1) | **23.8 tok/s** | 105.0 MB | `1574 + 64` |
| 128K `k8v4` (`129510` prompt, `128` dec) | crashed (token 1) | **28.6 tok/s** | 233.5 MB | `1677 + 70` |
| 262K `k8v4` (`251550` prompt, `128` dec) | not run (`<1536` bug) | **15.8 tok/s** | 90.1 MB | `1328 + 64` |

- **5-task summary**: median `41.9 -> 42.2 tok/s`, mean `42.42 -> 42.58 tok/s`, mean transfer `124.7 -> 122.8 MB/tok`.
- **Prefill speeds (`--kv-grow 1 --chunk 4096 --tc 1`)**: 16K `522.2`, 32K `649.1`, 64K `677.0`, 128K `i8r` `587.3`, 128K `k8v4` `596.2`, 262K `k8v4` `467.5 tok/s`.

