# Phase-aware residency allocation audit

Base revision: `e1538e9954a191d352cae0697b5908b0e477c913` (`betelgeuse-v6.1` mmap-backed runtime)

This is a source-level audit of the direct-checkpoint CUDA engine. Runtime allocation snapshots and the 4K/8K baseline still need a free GPU. At audit time, `bel-server` owned 11,832 MiB of the RTX 3060's 12,288 MiB, so I did not stop or benchmark over it.

## Allocation ownership and lifetime

| Allocation group | Owner / allocation site | Needed during | Current physical lifetime |
| --- | --- | --- | --- |
| Dense checkpoint tensors and norms | `dmat()`, `load()`; `cudaMalloc` plus upload | Prefill and decode | Entire process |
| Resident expert pools | `setup_offload()`; growable `VRange` when `--kv-grow 1` | Decode; some resident experts also accelerate short routed prefill | Entire process, except pool tail pages returned for prompt workspace or KV growth |
| Nonresident expert source | `SafeTensorFile` mapping and `ExpertParts` | CPU misses, prefill streams, and expert refill | File mapping for process lifetime; pages faulted on use. A separately budgeted subset may be copied into pinned memory. The full 14.6 GiB source is not copied when `--cpu-threads` selects the lazy path. |
| K/V state | `alloc_state()`; sliding layers use `cudaMalloc`, full-attention layers use growable `VRange` with `--kv-grow` | Prefill and decode | Sliding allocation lasts for process lifetime; full-attention physical pages grow and shrink by context need |
| Convolution and recurrent state | Per-call `Slot` arrays in `alloc_state()` | Prefill and decode | Entire process |
| Decode activations, router, expert outputs, attention scratch, logits | `alloc_state()` | Decode | Entire process, even though these buffers are not used by the batched prefill path |
| Batched prefill workspace | `alloc_prefill()`; `pf_alloc()` sends buffers >= 8 MiB to VMM ranges when `g_pf_elastic` is true | Prompt processing | VMM addresses remain reserved; physical pages are mapped for the chunk and unmapped after prefill. Small control arrays stay `cudaMalloc`-backed for process lifetime. |
| Prefill expert staging | Two 64 MiB VMM device halves plus two 64 MiB pinned host halves | Long/non-routed prefill streaming | Device pages are mapped only when streaming is selected; host halves live for process lifetime |
| Optional DSpark draft and verify state | `draft_load()`, `alloc_state()` and `alloc_prefill()` | Draft and verify | Entire process when enabled; excluded from the plain-decode figures below |
| Optional vision workspace | `vis_run()` temporary device arena | Image prefill only | Temporary; released after image processing |
| CUDA graphs and cuBLAS internal workspace | CUDA/cuBLAS runtime | Captured decode or GEMM calls | Runtime-owned and shape/device dependent; not included in the source-derived PF byte totals |

`pool_resize()` synchronizes, evicts only the pool tail, and destroys the decode and verify graph executables. `pf_unmap()` synchronizes, unmaps all mapped prefill ranges, then calls `pools_fill()`. `kv_fit()` can take pool slots back before mapping more full-attention KV. These paths preserve the model, existing prefix slots, CPU workers, and K/V state; they do not reload the model.

The current handoff does return physical pages after prefill, but each page is `cuMemUnmap`'d and its allocation handle is released. A later pool expansion creates new handles. The proposed arena will retain released handles for immediate remapping between prefill ranges, expert pools, and growable KV, and will drain unused handles before unrelated `cudaMalloc` calls.

## Persistent decode and state allocations

For the 48 layer visits, `alloc_state()` also allocates the following independently of the batched prefill table. They remain allocated throughout inference even when a request is using only the prefill path.

| Allocation | Size / capacity | Lifetime |
| --- | ---: | --- |
| Decode activations and logits scratch in `alloc_state()` | 1.6775 MiB, excluding the attention split buffers below | Process lifetime |
| `E.pm`, `E.pl`, `E.po` attention split workspace at 32K context | 1.5234 MiB | Process lifetime; sized from configured max context |
| Convolution / recurrent histories | 19.5 MiB for 48 visits | Process lifetime; state is updated by prefill and decode |
| Sliding-window K/V at chunk 4096 | 318.75 MiB | Process lifetime; capacity is `WINDOW + PF_RING_EXTRA` |
| Sliding-window K/V at chunk 8192 | 573.75 MiB | Process lifetime; 255 MiB more than chunk 4096 |
| Full-attention K/V, initially mapped at the 8192-cell grow step | 102 MiB | Growable VMM pages; up to 408 MiB at 32K, 204 MiB at 16K, or 3264 MiB at 256K |
| Graph executables | CUDA runtime-owned | Decode graph exists after warmup; pool size changes destroy it and the next decode captures it again |

The 1.6775 MiB decode scratch total is the exact sum of the `dalloc()` arrays in `alloc_state()` other than KV, histories, `E.pm/pl/po`, model weights, expert pools, and optional routing/DSpark allocations. The full-context K/V capacities are based on K8/V4's 1,632 bytes per cell and 8 full plus 40 sliding visits.

Dense model weights are uploaded by `dmat()` and `load()`, and are retained for the process lifetime. Their exact allocated byte count depends on the safetensors header and loader decisions (for example, host-only embeddings and lazy expert mappings are not device uploads); the post-load runtime snapshot is still required. The resident pool payload is exact from slot counts: `nslot_nv * 4,478,976 + nslot_mx * 8,211,456` bytes. CUDA graph and cuBLAS internal allocations are runtime-owned and must be measured from device/process snapshots, not included in these source-derived totals.

## Exact expert storage

These sizes include the quantized codes and their per-block scales. The device expert pointer tables, owner arrays, and slot metadata are separate allocations.

| Format | Part A (gate/up) | Part B (down) | One expert |
| --- | ---: | ---: | ---: |
| NVFP4 | 2,985,984 B | 1,492,992 B | 4,478,976 B (4.272 MiB) |
| MXFP8 | 5,474,304 B | 2,737,152 B | 8,211,456 B (7.831 MiB) |

The scale bytes are `2 * FF * H / 16` and `H * FF / 16` for NVFP4, and `2 * FF * H / 32` and `H * FF / 32` for MXFP8, with `H=2304`, `FF=1152`. Each expert also has a small host/device lookup entry; it does not add to the packed slot payload.

## Prefill-only GPU buffers at full chunk size

Derived from `alloc_prefill()` for `--tc 1`, mmap-backed offload, and a chunk that takes the streaming path. MiB uses 1,048,576 bytes. VMM entries are physically mapped during the prompt; the small allocations remain committed across decode. `PF.stg` is 2 x 64 MiB. The host staging halves are not GPU memory.

| Buffer(s) | 4096 chunk | 8192 chunk | Notes |
| --- | ---: | ---: | --- |
| `g_region` | 519.375 MiB | 1038.750 MiB | Shared attention, MoE, and optional logits region; max of its overlaid phases |
| `PF.B`, `PF.preB` | 288 MiB each | 576 MiB each | Batched block-state stacks |
| `PF.prev` | 288 MiB | 576 MiB | Previous-block state |
| `PF.msgs` | 144 MiB | 288 MiB | Message state |
| `PF.a` | 108 MiB | 216 MiB | Batched activations |
| `PF.x0`, `h`, `out`, `pre_h`, `xq`, `xk`, `xv`, `attn_out`, `att2`, `r`, `m`, `mn`, `moe`, `y2` | 36 MiB each | 72 MiB each | 14 separate arrays |
| `PF.kc`, `PF.vc` | 16 MiB each | 32 MiB each | Convolution output rows |
| `PF.att` | 48 MiB | 96 MiB | Batched attention output |
| `PF.cat`, `PF.gfg` | 72 MiB each | 144 MiB each | Gate workspace |
| `PF.wf` | 76.148 MiB | 76.148 MiB | FP32 dense-weight conversion buffer |
| `PF.wh` | 38.074 MiB | 38.074 MiB | FP16 dense-weight conversion buffer |
| `PF.xh` | 72 MiB | 144 MiB | FP16 activation copy |
| `PF.qh` | 24 MiB | 48 MiB | FP16 query copy |
| `PF.stg[2]` | 128 MiB total | 128 MiB total | Two device streaming halves |
| `PF.gdown`, `PF.rl`, `PF.wts` | 1, 2, 0.125 MiB | 2, 4, 0.25 MiB | Small `cudaMalloc` arrays |
| Route/plan arrays and device table copies | 0.383 MiB total | 0.762 MiB total | `ids`, `rows`, `slot`, tile tables, offsets, and two `dtab` arrays |
| **Physical workspace at full chunk** | **2705.105 MiB** | **5167.984 MiB** | Includes VMM and small persistent arrays |
| Of which mapped VMM workspace | 2701.598 MiB | 5160.973 MiB | Eligible for handoff after synchronization |
| Of which small persistent allocations | 3.508 MiB | 7.012 MiB | Currently not unmapped after prefill |

This is the maximum full-chunk footprint. `pf_map()` scales ordinary buffers with the active chunk, maps whole-buffer conversion scratch, and omits both staging halves for routed short prompts. The runtime log must report actual mapped bytes for each request; the table is not a claim that every request maps the maximum.

## KV interaction and chunk comparison control

For K8/V4, one K/V cell uses 1,024 B K codes, 512 B V codes, 32 B K scales, and 64 B V scales: 1,632 B total. Inkling has 8 full-attention layer visits and 40 sliding-window visits per token (`NCALLS=48`).

`PF_RING_EXTRA` is set to the prefill chunk. Thus, without a fixed ring capacity, the 4K and 8K runs also change sliding KV by 255 MiB: 318.75 MiB at chunk 4096 versus 573.75 MiB at chunk 8192. That changes expert capacity and invalidates an isolated chunk-size comparison. Baseline comparison must hold the ring extra at 8192 for both chunks, keeping context, K/V format, MX fraction, reserve, route profile, and all other arguments equal. The implementation will expose this as a benchmark/runtime override while retaining the current chunk-based default.

Full-attention K/V at 16K is 204 MiB; at 32K it is 408 MiB; at 256K it is 3,264 MiB. Sliding K/V is fixed by `WINDOW + PF_RING_EXTRA`, independent of conversation length. `kv_fit()` rounds full-attention growth to `--kv-grow-step` and shrinks expert pools first; `pf_fit()` includes planned KV growth and the configured VRAM margin before admitting a chunk.

## Issues to address

1. Reuse unmapped VMM physical handles across prefill, expert pools, and full-attention KV to avoid release/create churn. Preserve virtual addresses and only remap after device synchronization.
2. Add post-prefill telemetry for actual reclaimed bytes, transition time, pool counts, and peak process/device memory.
3. Count routes from the actual prompt and retain existing expert residents. Fill only newly available slots, ranking cold candidates with prompt route count and measured CPU worker latency per expert byte.
4. Record CPU worker latency by NVFP4/MXFP8 from real decode rings. The existing `InkBell.us_job` measures ring-to-ack latency per job but mixes formats; the repeated-expert standalone bandwidth result is not used.
5. Add a fixed sliding-ring-capacity control for the 4K/8K chunk comparison. Preserve the current default behavior.
6. Keep the existing grow guard and pool floors so later KV growth can reclaim decode slots safely.

## Pending runtime measurements

The experiment must fill in the active GPU free memory, RAM limit, exact loaded dense/state allocations, baseline 4K/8K pool counts, and all requested throughput/memory metrics. Those measurements were not run during this audit because the running `bel-server` occupied 11,832 MiB of the GPU. Do not infer those values from the static workspace calculation.
