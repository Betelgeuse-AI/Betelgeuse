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
