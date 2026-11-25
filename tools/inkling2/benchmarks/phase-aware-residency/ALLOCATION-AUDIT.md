# Phase-aware residency allocation audit

Base revision: `e1538e9954a191d352cae0697b5908b0e477c913` (`betelgeuse-v6.1` mmap-backed runtime)

This is a source-level audit of the direct-checkpoint CUDA engine at the base revision, with runtime measurements from the phase-aware branch below. The stale `bel-server` that initially occupied the GPU was confirmed by the user and terminated before benchmarking.

## Allocation ownership and lifetime

| Allocation group | Owner / allocation site | Needed during | Current physical lifetime |
| --- | --- | --- | --- |
| Dense checkpoint tensors and norms | `dmat()`, `load()`; `cudaMalloc` plus upload | Prefill and decode | Entire process |
| Resident expert pools | `setup_offload()`; growable `VRange` when `--kv-grow 1` | Decode; some resident experts also accelerate short routed prefill | Entire process, except pool tail pages returned for prompt workspace or KV growth |
| Nonresident expert source | `SafeTensorFile` mapping and `ExpertParts` | CPU misses, prefill streams, and expert refill | File mapping for process lifetime; pages faulted on use. A separately budgeted subset may be copied into pinned memory. The full 14.6 GiB source is not copied when `--cpu-threads` selects the lazy path. |
| K/V state | `alloc_state()`; sliding layers use `cudaMalloc`, full-attention layers use growable `VRange` with `--kv-grow` | Prefill and decode | Sliding allocation lasts for process lifetime; full-attention physical pages grow and shrink by context need |
| Convolution and recurrent state | Per-call `Slot` arrays in `alloc_state()` | Prefill and decode | Entire process |
| Decode activations, router, expert outputs, attention scratch, logits | `alloc_state()` | Decode | Entire process, even though these buffers are not used by the batched prefill path |
| Batched prefill workspace | `alloc_prefill()`; `pf_alloc()` sends buffers >= 8 MiB to VMM ranges when `g_pf_elastic` is true | Prompt processing | VMM addresses remain reserved; physical pages are mapped for the chunk and moved to a shared VMM handle cache after the prompt. Small control arrays stay `cudaMalloc`-backed for process lifetime. |
| Prefill expert staging | Two 64 MiB VMM device halves plus two 64 MiB pinned host halves | Long/non-routed prefill streaming | Device pages are mapped only when streaming is selected; host halves live for process lifetime |
| Optional DSpark draft and verify state | `draft_load()`, `alloc_state()` and `alloc_prefill()` | Draft and verify | Entire process when enabled; excluded from the plain-decode figures below |
| Optional vision workspace | `vis_run()` temporary device arena | Image prefill only | Temporary; released after image processing |
| CUDA graphs and cuBLAS internal workspace | CUDA/cuBLAS runtime | Captured decode or GEMM calls | Runtime-owned and shape/device dependent; not included in the source-derived PF byte totals |

`pool_resize()` synchronizes, evicts only the pool tail, and destroys the decode and verify graph executables. `pf_unmap()` synchronizes, unmaps mapped prefill ranges into the shared VMM handle cache, then calls `pools_fill()`. New expert slots can map those same handles. `kv_fit()` can take pool slots back before mapping more full-attention K/V. These paths preserve the model, existing prefix slots, CPU workers, and K/V state; they do not reload the model.

At base revision `e1538e995`, an unmapped page's allocation handle was released, and a later pool expansion created a new handle. This branch retains released handles for reuse between prefill ranges, expert pools, and growable KV. `vram_free()` counts cached handles as reusable capacity; `dalloc()` releases cached handles before a regular `cudaMalloc` needs that memory.

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

## Phase-aware implementation

- `VRange::unmap_to()` now places physical handles in `g_vmm_cached`. `VRange::map_to()` reuses them before creating new handles. The virtual addresses stay fixed.
- `vram_free()` adds cached VMM bytes to CUDA's reported free memory. `dalloc()` drains the cache before a regular allocation so unused VMM handles do not block other engine allocations.
- `pf_unmap()` reports actual mapped bytes, handles reused, transition time, pool counts, and refill size.
- The prefill route planner counts selected experts for the active prompt. The post-prompt refill keeps current residents and fills empty slots using prompt routes, the static profile, decayed decode counts, and measured CPU ring-to-ack cost per layer and format. Looped core layers 6-17 are admitted first.
- `InkBell` records per-format and per-layer CPU service time from real inference jobs. No standalone repeated-expert throughput is used for placement.
- `INK2_RING_EXTRA` fixes the sliding K/V capacity in the 4K/8K comparison. The default remains the configured chunk size; the benchmark set it to 8192 for both variants.

## Runtime measurements

Hardware was an RTX 3060 with 12,288 MiB VRAM and 16,151,204 kB system RAM. `nvidia-smi` showed 11,857 MiB total GPU use during the runs, including 11,834 MiB for `bel-server`. The server's peak RSS was 12,137 MiB for the 8K run and 12,439 MiB for the 4K run. The lowest sampled `MemAvailable` was 12,046 MiB and 12,132 MiB, respectively.

The engine loaded the checkpoint experts from a 6,246 MiB file-backed mapping. It kept a bounded 511 MiB pinned host copy; it did not create or pin a 14.6 GiB anonymous expert arena. Startup placement was 1,875 NVFP4 + 88 MXFP8 slots for the 8K run and 1,876 + 88 for the 4K run. All 1,536 experts in physical layers 6-17 were resident at startup and had zero CPU misses in the captured decode traces.

At a 16K prompt, K/V growth reached 20,480 cells and 0.27 GiB. After the prompt, the 4K variant held 1,826-1,837 NVFP4 + 88 MXFP8 slots across its three repeats. The 8K variant held 1,836 NVFP4 + 88 MXFP8 slots in its warm matrix run and route-capture run. The 8K chunk increased prompt workspace and prefill rate, but did not increase final decode residency at 16K because both variants returned their workspace to the pool and retained the same K/V capacity.

Actual workspace at handoff was 2,706 MiB for a 4K chunk and 5,166 MiB for an 8K chunk. The 4K run remapped about 2,704-2,708 MiB of VMM handles at the handoff and admitted 589-590 NVFP4 + 24 MXFP8 slots. The 8K run remapped about 5,164-5,168 MiB and admitted 1,165-1,166 NVFP4 + 24 MXFP8 slots. A fresh 16K route-capture run measured a 1,168.6 ms transition; the warm matrix run was about 0.90-0.92 s. The 4K 16K transitions were about 0.47-0.48 s.

The working data and caveats are in [RESULTS.md](RESULTS.md). The benchmark prompts are complete Inkling messages ending with the review question and model prefix. Prompt caching stayed enabled. Repeated identical prompts reported `cache_n=0`, so these measurements do not demonstrate a cache hit or conversation continuation. Exact greedy output matching across repeats was not achieved. Only prompts through 16K were run; 32K and 256K stability remain unverified.
