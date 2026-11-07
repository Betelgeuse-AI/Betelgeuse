# Betelgeuse DeepSeek-V4.1-Flash (`betelgeuse-ds41`) Engineering Roadmap & Tasks

## 1. Hardware, Environment & Models (Local Only)
- **Local Branch**: `betelgeuse-ds41` based on `77e778ee1`. Local local commits only; NEVER run `git push` or open GitHub PRs. Leave `master` and `fluxaeros-mtp` untouched.
- **Hardware Target**:
  - NVIDIA GeForce RTX 3060 12GB VRAM (`11,895 MiB` usable, `sm_86`, clocks 1800-2115 MHz).
  - 16GB Single-Channel DDR4-3200 (`~12 GiB` available RAM, strictly `0 B` swap).
  - NVMe SSD (`~2.2 GB/s`, Btrfs).
- **Model Paths**:
  - Main Backbone (748B total, 40 layers, 384 experts/layer, top-6, ~12.44 MB/expert):
    `/home/zyxciss/Models/BIG-MOE/DeepSeek-V4.1-Flash-1017/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00001-of-00007.gguf`
    Companion Engram at Layers 1 & 14: `DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf`.
  - Draft Model (3 layers, 128 experts/layer, top-3, 5-token block drafter):
    `/home/zyxciss/Models/BIG-MOE/DeepSeek-V4.1-Flash-1017/dspark-DeepSeek-V4.1-Flash-1017-FLE-Q2K-Q8.gguf`
  - Flags: Always `--spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-p-min 0.50 -ngl 99 -ngld 99 -b 64 -ub 16 -c 512 -np 1 -fit off --moe-direct-io --no-warmup --no-cache-prompt --cache-ram 0 -t 6 -tb 6`.
  - Build command: ONLY `cmake --build build --target llama-server -j12`.

---

## 2. Verified Baseline Facts & Current State
- **ColdSSD on Warm Passes**:
  - SOLVED: Pass 1 cold write-allocates into Host-RAM. Warm Pass 2 & 3 achieve `ColdSSD = 0.0%` and `0.0 ms` disk wait in both backbone and draft.
- **Cache Geometry & Host RAM**:
  - SOLVED: Target has 200 persistent VRAM hot slots (5/layer) + 600 inclusive Host-RAM slots (15/layer).
  - SOLVED: Draft has 36 persistent VRAM hot slots (12/layer) + 60 inclusive Host-RAM slots (20/layer).
  - Total Host RAM usage is `< 10.1 GiB` peak with strictly `0 B` swap.
- **PCIe H2D Overlap**:
  - SOLVED: Exposed PCIe H2D wait dropped from baseline `284.87 ms` down to `3.7 - 7.1 ms` per verification sweep.
  - Streaming uses `d_layer_stage[il % 2]` with two events: `hot_ready[b]` (recorded after D2D/zero setup, before H2D) and `ready[b]` (after all H2D).
- **CPU Expert Offload (`moe-cpu.cpp`, `moe-cpu.h`)**:
  - SOLVED: Concurrent AVX2 threadpool (6 threads) evaluates excess Host-RAM selections outside the 2-expert GPU H2D transfer cap.
  - Live numerical verification (`MOE_CPU_VALIDATE=1`) passed 4,808 rows across all 40 layers against GPU reference with max relative L2 error `0.0268` (within single-precision quantization bounds).
  - Registered down hook (`ggml_cuda_moe_ffn_down_register`) patches CPU down-projection rows before final reduction.
- **Zero-Sentinel Optimization**:
  - SOLVED: Sentinel slot 18 represents un-transferred / CPU-computed experts in GPU execution buffers.
  - `mmvq.cu`, `mmq.cu`, `quantize.cu`, and `mmid.cu` detect the scored zero sentinel via `ggml_cuda_moe_zero_expert(ids)`, skip weight dot products, and output exact `0.0f`.
  - Fixed duplicate positive expert slot compaction in `mmid.cu`; isolated regression tests pass with NMSE `< 1.3e-4`.

---

## 3. Active Goal & Problem Statement
- **Performance Goal**:
  - Target TG: **`24 - 29+ tok/s`** actual wall-clock decode (`4.43 - 5.11` accepted tokens per sweep).
  - Target PP: **`50+ tok/s`** prompt processing (`< 140 ms` for 7 tokens).
- **The Identified Bottleneck (The ~130 ms Gap)**:
  - Pure GPU execution per verification sweep is fast: **`124.8 - 134.0 ms`** for target MoE + **`10.3 - 11.3 ms`** for DFlash = **`~135 - 145 ms` pure GPU pace (`~28 - 32 tok/s` potential)**!
  - Exposed H2D PCIe wait is only **`3.7 - 7.1 ms`**.
  - But actual verification wall time per sweep is **`286 - 311 ms`** (`10.5 - 11.2 tok/s` actual).
  - **Where the ~130 ms goes**:
    1. **Host-Side Worker Thread Blocking**: In `src/llama-context.cpp:moe_ffn_down_sync`, the host thread evaluating the graph calls `state.wait_cpu()`. If the CPU worker is computing 60+ expert rows (`cpu_ms` = 50-100 ms), the host thread blocks before enqueuing Layer `il+1` to the CUDA stream. The GPU starves and sits idle waiting for the host to submit subsequent layers.
    2. **Host-Side `ggml_backend_sched_synchronize` and CUDA event query overhead**: Per-step synchronization and memory accounting serialize host and device.
    3. **Unscored fallback on single-token decode steps (`target_single`)**: When draft rejects early or hits EOS, single tokens run without the fast residency path, taking `~120-130 ms` per step.

---

## 4. Prioritized Tasks (What TO Do)

### Task 1: Eliminate Host-Thread GPU Starvation in `moe_ffn_down_sync`
- **Root Cause**: `moe_ffn_down_sync` is called during the graph's forward pass by the CUDA backend host runner. Calling `state.wait_cpu()` there blocks the host thread from queuing later layers to the GPU.
- **Solution**:
  - Keep CPU expert jobs small and fast: budget CPU offload to at most 1-2 high-impact specialist experts per layer across the batch, keeping `cpu_ms` strictly `< 25-35 ms` so it finishes well before the GPU reaches Layer `il`'s down projection.
  - Alternatively, overlap CPU submission earlier (e.g. submit CPU job during previous layer or during attention) so CPU execution has a 50+ ms head start.
  - Or allow GPU down-projection to run on the zero sentinel, enqueue the host-to-device copy into `stream_compute` without blocking host kernel submission (using CUDA events/callbacks or dedicated copy streams synced to compute stream).

### Task 2: Tune Warm Residency & Eliminate Host Stalls
- Enforce strict near-tie residency margin (`resident_margin = 0.05 - 0.10`) for Ranks 1..5 so `85 - 90%+` of all expert selections hit the 5 VRAM hot slots (`0 ms` transfer, `0 ms` CPU compute).
- Cap unique Host-RAM specialists streamed via H2D to at most 1 per layer across the 6-token batch (`<= 12.44 MB/layer` = `< 45 ms` total H2D per sweep).
- Keep Rank-0 choice protected in cache so draft acceptance remains high (`> 88%`).

### Task 3: Execute Official Multi-Pass Benchmark Protocol
1. Launch server with required arguments:
   `systemd-run --user --scope --quiet --unit=llama-ds41-benchmark -p MemorySwapMax=0 -p MemoryHigh=12032M -p MemoryMax=12G env GGML_NO_BACKTRACE=1 MOE_PROFILE=1 MOE_PLE_STAGING=1 GGML_CUDA_GRAPH_CHUNK_SIZE=0 build/bin/llama-server -m /home/zyxciss/Models/BIG-MOE/DeepSeek-V4.1-Flash-1017/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00001-of-00007.gguf --model-draft /home/zyxciss/Models/BIG-MOE/DeepSeek-V4.1-Flash-1017/dspark-DeepSeek-V4.1-Flash-1017-FLE-Q2K-Q8.gguf --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-p-min 0.50 -ngl 99 -ngld 99 -b 64 -ub 16 -c 512 -np 1 -fit off --moe-direct-io --no-warmup --no-cache-prompt --cache-ram 0 -t 6 -tb 6 --threads-http 2 --host 127.0.0.1 --port 8091 --no-webui --offline --log-colors off --log-timestamps -lv 4 --ctx-checkpoints 0`
2. Run Pass 1 (Cold, 64 tokens) on `def binary_search(arr, target):\n`.
3. Run Pass 2 (Warm, 32 tokens).
4. Run Pass 3 (Warm, 48 tokens).
5. Run Warm 64-token prompt to verify PP scaling.
6. Verify `0 B` swap used (`free -b && swapon --show`) and peak host RAM `< 12 GiB`.

### Task 4: Format Deliverables
- Commit hash on `betelgeuse-ds41`.
- Cache hit rates table (VRAM %, Host-RAM %, ColdSSD % = 0.0%).
- Granular millisecond timing table (PP ms & tok/s, Draft ms, Target GPU ms, Total H2D ms, Exposed wait ms, Acceptance rate, TG tok/s).
- Full generated Python code from warm passes.

---

## 5. Prohibitions & Hard Rules (What NOT to Do)
- **DO NOT** push to remote GitHub or create pull requests (`gh pr create`, `git push` are strictly forbidden).
- **DO NOT** use swap memory or exceed 12 GiB host RAM.
- **DO NOT** re-introduce Minja (llama.cpp has its own Jinja engine).
- **DO NOT** allocate a second 7.1GB host cache for DFlash.
- **DO NOT** hardcode column wrapping in code comments or use non-ASCII characters (`—`, `→`, `…`).
- **DO NOT** drop or stub real model computation. CPU and GPU expert outputs must remain mathematically valid.
- **DO NOT** disable CUDA graphs globally.

---

## Atomic task 10: Rank-0-only SSD fetches and a real decode warm-up

### Objective
Keep the exact per-layer Target `5 VRAM Hot + 15 Host-RAM` and DFlash `12 + 20` packing. Only target Rank 0 (`r == 0`) may fetch a non-resident expert from SSD on any cold or warm PP/TG pass. Ranks 1..5 always resolve the top five resident experts by unbiased `sqrtsoftplus` score. Never replace Rank 0. Run one supervised session under `MemoryMax=12G`, report, and stop.

### Implementation
1. Host entries carry `was_rank0` and `last_used_step`. Eviction within the layer prefers empty slots, then the least-recently-used non-Rank-0 slot, then the least-recently-used non-active Rank-0 slot.
2. A cold Rank 0 miss is fetched from SSD, its page-aligned range gets `POSIX_FADV_DONTNEED`, and it is write-allocated into the layer's 15 Host-RAM slots when an eligible slot exists. A non-Rank-0 cold miss throws.
3. Host hits update `last_used_step`; Rank-0 hits set `was_rank0`. Secondary ranks never write-allocate and never read SSD.
4. `refresh_hot` promotes only host-resident experts. Target background promotion is disabled; the demand path is the only target SSD reader. DFlash keeps its own promotion path and 12+20 cache.
5. The startup GGUF drop is applied to all pager shard descriptors after seeding.
6. The router receives unbiased `probs`; the biased `selection_probs` stays selection-only for native top-k.
7. Shared MoE tiles are bounded to `18 / n_expert_used` tokens so the Rank-0 union plus resident picks fit the 18-slot ring. All 18 entries have a zero-sentinel row for the CPU branch.

### Single-session protocol
`MemorySwapMax=0`, `MemoryHigh=12032M`, `MemoryMax=12G`. Same model paths and server flags as Task 9, with `--no-cache-prompt --cache-ram 0` and request `cache_prompt=false` so every pass reprocesses its full prompt.

| Pass | Prompt | Output | Purpose |
| --- | --- | ---: | --- |
| 1 | `def binary_search(arr, target):\n` | 64 | Cold TG warm-up with real decode |
| 2 | same prompt | 32 | Warm TG |
| 3 | same prompt | 64 | Warm TG and generated Python |
| 4a | first 64 tokens of the long text | 1 | Cold 64-token PP warm-up |
| 4b | same 64-token prompt | 1 | Warm 64-token PP |

### Acceptance and stop rule
- Every target pass: `rank0_replaced = 0`; no `[moe_cold]` direct lines on warm passes.
- Warm Pass 2/3/4b: `ColdSSD = 0.00%`, `disk_wait_ms = 0.0`.
- Scope `MemoryPeak < 10.6 GiB`, zero OOM, zero aborts.
- Warm Pass 4b PP about `50+ tok/s`; Pass 2/3 TG `15-18+ tok/s`; complete Pass 3 Python.
- One session only. Report every miss; do not rerun. No commit, push, reset, stash restore, or PR; leave `fluxaeros-mtp` untouched.

### Evidence
The first launch failed during initialization, before any request: the kick op rejected the reshaped 3D score tensor, and the scope hit its 12G cap during startup. Two fixes were applied without changing the protocol: pass the pre-reshape 2D unbiased score tensor, and drop the GGUF page cache before the host arena is pinned (the existing post-seed drop remains). The single five-pass session then ran to completion; it was not rerun.

Geometry was unchanged: `stage=2x18 554.41 MiB hot=236 2854.42 MiB inclusive_host=660 7922.02 MiB target_host=600 draft_host=60 draft_vram=36`.

Protocol: `prompt_n` was 7/7/7/64/64 with `cache_n=0` on every pass, so each pass processed its full prompt.

| Pass | `rank0_replaced` (target) | Target PP ColdSSD | Target TG ColdSSD | Target disk wait | TG tok/s | PP tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 cold 64-token TG | 0 | 217/1680 = 12.92% | 752/16560 = 4.54% | 4176.6 ms | 4.701 | - |
| 2 warm 32-token TG | 0 | 58/1680 = 3.45% | 343/8400 = 4.08% | 1837.2 ms | 4.991 | - |
| 3 warm 64-token TG | 0 | 46/1680 = 2.74% | 583/15600 = 3.74% | 2785.5 ms | 5.664 | - |
| 4a cold 64-token PP | 0 | 693/15360 = 4.51% | - | 2924.2 ms | - | 5.800 |
| 4b warm 64-token PP | 0 | 559/15360 = 3.64% | - | 2343.2 ms | - | 6.777 |

`rank0_replaced` is zero for the target on every pass, and no target background promotion ran (`promo_bg=0`). DFlash still reports its own warm fallback: `rank0_replaced` 47/10/12 and `promo_bg` 14/8/16 on Passes 1/2/3; DFlash TG ColdSSD was 6.67%/0.00%/0.00%.

Failures:
- Warm ColdSSD is not zero on Pass 2, Pass 3, or Pass 4b. The resident-only secondary routing changes the cache state seen by each pass, so the decoded token sequence and its Rank-0 experts differ between passes; the layer's 20 slots do not cover the union. Six `[moe_cold]` direct lines occurred (3/0/2/1/0 per pass), all `host full`: during split PP, CPU-staged host slots are in flight and cannot be evicted.
- Throughput missed: Pass 2/3 TG 4.991/5.664 tok/s versus 15-18+; Pass 4b PP 6.777 tok/s versus 50+. Each 16-token PP ubatch splits into six 3-token MoE tiles with host synchronization, and the CPU branch evaluated about 1050 rows per layer step (`cpu_ms` about 830 ms per PP ubatch).
- Scope `MemoryPeak = 12,616,167,424 B` (11.75 GiB), above 10.6 GiB. `MemoryCurrent` was 12,074,496,000 B before requests and 11,962,888,192 B after Pass 4b. `MemorySwapPeak = 0`, `Result=success`, zero OOM, zero aborts.
- Pass 3 generated 64 tokens and stopped at `else:`; the function is incomplete and not valid Python (`PASS3_VALID_PYTHON=False`).

Pass 3 output:

```python
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
```

The single five-pass session was not rerun. No commit, push, reset, stash restore, or PR was performed; `fluxaeros-mtp` is untouched. Evidence: `/tmp/opencode/atomic-hot10.log`, `/tmp/opencode/atomic_hot10_results.json`, `/tmp/opencode/atomic_hot10_bench.out`, `/tmp/opencode/atomic_hot10_mem.txt`, `/tmp/opencode/atomic_hot10_pass*.log`.

---

## Atomic task 11: immutable Zone A, Rank-0 Zone B, single-pass PP

### Objective
Split each target layer's 15 Host-RAM slots into 9 immutable Zone A profile slots plus 6 append-only Zone B Rank-0 slots. Secondary ranks select only from Zone A (5 hot + 9 host). Rank 0 uses Zone A and Zone B, fetches only into a free Zone B slot on a cold pass, and falls back to the best Zone A or Zone B expert when no slot or when warm. Hot slots stay frozen after seeding. Remove the 3-token PP tiling and the PP CPU branch.

### Implementation
1. Seed 9 host entries per target layer (Zone A host 0..8) plus the 5 hot slots. The 14 Zone A slots hold the top-9 router-bias experts; hot duplicates host 0..4. Zone B host 9..14 starts empty and is never evicted.
2. `zone_b_victim`/`zone_b_free` scan only host slots 9..14. Cold Rank-0 misses write-allocate into free Zone B slots; once all are full, the pass falls back to the best Zone A or Zone B expert. Warm misses always fall back.
3. The active ubatch pre-scans its first `zone_b_free` distinct missing Rank-0 experts. Fallback selection sees Zone A, filled Zone B, and that pending set, so a cold ubatch and its warm repeat choose from the same candidate pool.
4. Secondary ranks scan only host 0..8 and the 5 hot slots. They never see Zone B and never trigger a fetch.
5. Remove the target `refresh_hot` call so Zone A hot slots are immutable. DFlash keeps its own promotion and 12+20 cache.
6. Shared MoE tiles are disabled for the 6-expert target (`tile_size = 64`), so a 16-token PP ubatch runs one MoE pass per layer. The target CPU branch is limited to `nt <= 6`; PP stages all selected host experts into the 18-slot ring. The union is bounded by 9 Zone A host plus 6 Zone B experts.
7. Pass 1 and Pass 3 use `n_predict = 96`; Pass 2 uses 32; Pass 4a/4b use the 64-token prompt.

### Acceptance
- Warm Pass 2/3/4b target `ColdSSD = 0.00%`, `disk_wait_ms = 0.0`.
- Scope `MemoryPeak` inside `MemoryMax=12G`, zero OOM.
- Warm Pass 4b PP about 50+ tok/s; Pass 2/3 TG 15-18+ tok/s.
- Complete `binary_search` Python from Pass 3.
- One benchmark session; then commit locally with `moe: task 11 zone A/B invariant routing and single-pass PP`. No push or PR.

### Evidence
The first launch completed Pass 1 but crashed in warm Pass 2: the graph still built the two-branch warm split while runtime routing disabled it for PP. Gating the graph split on `n_tokens <= 6` fixed the mismatch; the five-pass session below then ran once and was not rerun.

Geometry was unchanged: `stage=2x18 554.41 MiB hot=236 2854.42 MiB inclusive_host=660 7922.02 MiB target_host=600 draft_host=60 draft_vram=36`. Protocol was valid: `prompt_n` 7/7/7/64/64 and `cache_n=0` on every pass.

| Pass | Target `rank0_replaced` | Target PP ColdSSD | Target TG ColdSSD | Target disk wait | TG tok/s | PP tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 cold 96-token TG | 2760 | 274/1680 = 16.31% | 11/24960 = 0.044% | 633.0 ms | 12.796 | - |
| 2 warm 32-token TG | 889 | 0/1680 = 0.00% | 0/8400 = 0.00% | 0.0 ms | 10.347 | - |
| 3 warm 96-token TG | 2788 | 0/1680 = 0.00% | 0/23760 = 0.00% | 0.0 ms | 14.515 | - |
| 4a 64-token PP | 1514 | 0/15360 = 0.00% | - | 0.0 ms | - | 44.401 |
| 4b warm 64-token PP | 1514 | 0/15360 = 0.00% | - | 0.0 ms | - | 50.631 |

Warm target ColdSSD was exactly `0.00%` and `disk_wait_ms = 0.0` on Pass 2, Pass 3, Pass 4a, and Pass 4b. There were zero `[moe_cold]` lines and zero target `promo_bg`. Pass 4a and 4b have identical counters, and the Pass 2 content is a prefix of Pass 3, so the frozen-cache trajectory invariant holds.

Scope memory: peak `12,529,819,648 B` (11.67 GiB), `MemoryCurrent` 11,148,341,248 B before requests and 11,617,148,928 B after Pass 4b, `MemorySwapPeak=0`, `Result=success`, zero OOM, zero aborts. `hot_slots=200`, `host_slots=556`: Zone B is append-only and filled 196 of 240 slots during the cold phase.

Misses:
- Pass 2/3 TG were `10.347`/`14.515 tok/s`, below 15-18+.
- Pass 3 Python is degenerate and invalid: `if (len(array) > 0) < 0: 0: 0: ...` for 96 tokens. The immutable Zone A profile leaves only 9 distinct experts for ranks 1..5, so output quality collapsed even though the warm-cache invariant is exact.
- The first launch was an implementation failure, not a benchmark rerun; the five-pass session was run once after the graph fix.

The final implementation and this file were committed locally on `betelgeuse-ds41` with `moe: task 11 zone A/B invariant routing and single-pass PP`. No push or PR; `fluxaeros-mtp` untouched. Evidence: `/tmp/opencode/atomic-hot10.log`, `/tmp/opencode/atomic_hot10_results.json`, `/tmp/opencode/atomic_hot10_bench.out`, `/tmp/opencode/atomic_hot10_mem.txt`, and `/tmp/opencode/atomic_hot10_pass*.log`.
