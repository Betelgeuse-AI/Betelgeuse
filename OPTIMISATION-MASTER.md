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
