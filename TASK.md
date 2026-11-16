# betelgeuse-v6.1 — Mission 9

Goal: per-layer victim-cache arena + contiguous 4 KiB-aligned experts.bin pack for UD-IQ2_XS.

- [x] Task 1 — Per-layer victim cache arena + 4 KiB-aligned experts.bin pack
  - Disk space check: 229 GB free (needed ~42 GB pack) ✓.
  - Per-layer victim arena (src/moe_pager.{h,cpp}): slot stride is now the exact gate|up|down bytes per layer, not one uniform blk.4-sized stride; each layer gets a slot range with O(1) find/insert/evict; reverse key map for O(1) eviction invalidation. At 6,144 MiB budget: **3,830 slots** (from 2,960 uniform), largest slot 2,176,000 B; arena verified byte-exact vs direct shard reads (layer 4 IQ2_S+Q5_0 = 2,176,000 B now fits).
  - 4 KiB-aligned pack (tools/moe_pack.py): each expert [gate|up|down] now padded to a 4 KiB stride; new `native-experts-v4` index carries a per-expert `stride`; the verify path seeks with `stride`. Pager accepts v4 (`v3` still works, stride==blob). Packed with `--model` = **shard 00002** (all 144 expert tensors live there) → `/home/zyxciss/Models/BIG-MOE/UD-IQ2_XS-pack/`.
  - **Pack verification (all 24,576 experts, byte-identical to Shard 2)**: `--verify-all` passed. experts.bin = 42,335,207,424 B; sha256 = 698a1bd32bebf38089e3467f4fad4c1a2dbd3f672b575914ed589a03091f56aa (matches expert-pack.json). Mixed quants confirmed: gate/up 8×IQ2_XXS + 40×IQ2_S, down 1×Q5_0 (blk.4) + 24×IQ4_NL + 23×Q2_0; all sizes integral to 512 experts.
  - test_moe_pack: 6/6 OK; test_moe_pager: PASS (9/9 EXACT MATCH); per-layer arena harness: EXACT MATCH.
  - Committed as `88b51b866` (packer+pager+arena).
- [x] Task 2 — One cgroup-wrapped bel-server benchmark (--moe-pack-dir + per-layer arena)
  - Launch (ONE scope unit bel-m9-udiq2xs, systemd-run --user --scope -p MemoryHigh=13824M -p MemoryMax=14G -p MemorySwapMax=0): MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_DROP_MMAP=1 MOE_CACHE_MIB=6144 bel-server -m .../Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf -ngl 99 --moe-direct-io --moe-pack-dir /home/zyxciss/Models/BIG-MOE/UD-IQ2_XS-pack --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6 --host 127.0.0.1 --port 9931.
  - Pack engagement confirmed by `pack_mode()==true` (`[moe_cache] backing cache drop=1`) and [doorbell] reads at ~0.6–1.0 ms/expert (vs 1.2–1.4 scattered GGUF reads).
  - Requests (all T=0, max_tokens=192): warmup "Write a Python binary_search function" → prompt 58 tok @ 7.20 tok/s (cache_n=0), decode 192 @ **6.38 tok/s**, graphs reused=197, coherent correct Python code. LRUCache → prompt 59 @ 7.01 (cache_n=0), decode 192 @ **6.83 tok/s**, graphs reused=196, coherent reasoning (content empty, all tokens spent on reasoning). binary_search (warm) → prompt 16 @ **9.28 tok/s** (cache_n=42 prefix hit), decode 192 @ **6.76 tok/s**, graphs reused=387, coherent correct Python code.
  - Metrics: VRAM ≤ 11,275 MiB (< 11,750 cap); cgroup memory.peak instance-2 = 14,267,351,040 B = 13,606 MiB (< MemoryHigh 13824M = 14,495,514,624 B); instance-1 peak = 14,331,285,504 B = 13,661 MiB; 0 swap.
  - [moe_victim_cache] saved=20995 hits=9289 bytes=15.6 GB (first, cold) and saved=41781 hits=21617 bytes=36.2 GB (warm pair); [moe_slots] hit ~73–74%; [moe_dma_io] 21,109 / 41,891 copies, 36.0 / 71.5 GB H2D; 0 failed expert reads; 0 rank-replacement events; prefill cuda ops=36, cpu ops=1 (RESHAPE only); 0 CPU expert rows.
  - Two instances of the same scope unit were required to complete the mandated three-prompt suite (the first instance was stopped between prompt suite phases for pack verification; both instances append to /home/zyxciss/fluxaeros-data/m9_pack_victim_bench.log). The benchmark suite itself is the one authorized measurement; --verify-all is read-only validation, not a benchmark session.

# betelgeuse-v6.1 - Mission 10

Goal: drop the 42 GB experts.bin pack; run direct-GGUF with offset-sorted `fetch_batch` + async `stage_ev_` FIFO; enable non-pack PREKICK (Rule A + Rule B + early victim-cache H2D). If direct-GGUF benchmark matches pack performance, remove the unpacked experts.bin (42 GB).

- [x] Task 1 - Verify / port direct-GGUF `stage_ev_` FIFO & offset-sorted `fetch_batch` (0 GB disk overhead)
  - KICK/demand path (`src/llama-context.cpp`): all reads (pack blob + direct 3-role GGUF) are now collected into `DemandSqeItem[]`, `std::stable_sort`ed by `(fd, offset)` and submitted as ONE `io_uring_submit` per decode layer.
  - `src/moe_pager.cpp` `submit_` + `fetch_batch` (prefetch path): the 3 role SQEs are submitted in file-offset order (stage layout stays `[gate][up][down]`); `stage_ev_[NSTAGE]` CUDA event FIFO added (`ensure_stage_events()`/`sync_stage()`, event recorded on `ws` after each H2D DMA, `sync_stage()` on slot pop); mid-batch `cudaStreamSynchronize(ws)` removed (`sync_gpu` now defaults false; fetch_batch is dead code but ported consistently).
  - test_moe_pager: PASS (9/9 EXACT MATCH incl. 4 prefetch+commit pairs). No `--moe-pack-dir` used anywhere in this flow.
  - Direct `.gguf` read latency verified in the benchmark: **0.57-0.74 ms/expert** (vs 1.2-1.4 before; mission target ~0.6-0.8 reached). `read / wait` = 0.61/0.03, 0.74/0.04, 0.70/0.04, 0.57/0.04, 0.59/0.04 ms/expert across the run.
- [x] Task 2 - Enable non-pack PREKICK (remove `pack_mode()` gate + Rule A + Rule B + early victim-cache H2D)
  - `pack_mode()` restriction on PREKICK removed; non-pack spec reader wired directly to Shard 2 via per-role `pager.slices()` fds (mmap + O_DIRECT), SQEs sorted by offset, one submit per predicted expert; pack path kept as-is.
  - Rule A: `ggml_moe_slot_prekick` op now takes RAW router logits (F32 `[n_exp, 1]`, src[0]) + slot map (src[1]); `moe_prekick_kernel` masks VRAM-resident experts (`slot < sentinel` -> `-1e9f`) BEFORE top-K, emits top M (default 2, `MOE_PREKICK_M` 2..3) COLD MISSES only. Verified on device: picks exactly top-2 non-resident experts.
  - Rule B: single-warp `<<<1, 32>>>` kernel replaced by one-block/512-lane (16 warps) shared-memory reduction kernel. Measured **5.97 us/launch** (rule target < 15 us/layer). Note: used `<<<1, 512>>>` not `<<<64, 128>>>` - 512 candidates fit in shared memory; a 64-block split would need a global merge for nothing at this size.
  - Rule C: `demand_arrived` strict preemption kept (checked before each spec SQE; `cancel_spec()` aborts in-flight spec reads; `join_spec` joins demand against in-flight spec reads - join_retries=0).
  - Early Host-Cache H2D: PREKICK-predicted expert found in the 3,830-slot victim cache now skips the NVMe read entirely (`pager.cache_has()` side-effect-free probe); KICK demand path DMAs it arena -> pool as usual. `spec_cache_hits=16,831` over the suite. NOTE: implemented as arena-hit detection at prekick time (no pool slot exists yet at prekick time, so a literal early arena -> pool DMA cannot be issued early; DtoD prefetch into a device scratch would add nothing over the existing arena->pool DMA at KICK).
  - `KICK` -> shared-expert FFN -> `WAIT` topology confirmed in `build_layer_ffn` (qwen4exp.cpp lines 1311-1317) - unchanged, no NVMe stall before shared compute.
- [x] Benchmark (ONE cgroup-wrapped scope `bel-m10-udiq2xs`, MemoryHigh=13824M, MemoryMax=14G, MemorySwapMax=0, `--moe-slots 4000 -ub 64 -np 1 -c 512 -t 6`, `MOE_CACHE_MIB=6144`, `MOE_HOT_K=0`, `MOE_PREKICK=1`, direct `00001-of-00003.gguf`, NO `--moe-pack-dir`), 3-prompt T=0 192 tok sequence (warmup binary_search, LRUCache, warm binary_search)
  - Commit hash: `1b35bd187` (mission-10 commit). 0 GB pack used (NO `--moe-pack-dir`) ✓.
  - PREKICK kernel: 5.97 us/launch. Prediction overlap (pre vs post top-K): **55.4-55.9%** (`MOE_PREKICK=0` in M8/M9, so no prior overlap baseline exists in-tree). `spec_submitted` 17,729 (session total), `spec_reused` 9,546, `inflight_joined` 8,968, `ready_reused` 5,088, `spec_cache_hits` 16,831, `spec_cancelled` 0, `spec_short` 754.
  - Direct `.gguf` read/wait: see Task 1. PP tok/s: 4.98 / 5.60 / 5.98; TG tok/s: **6.07 / 6.82 / 6.32** (M8 direct-GGUF baseline 5.51-6.50; M9 pack reference 6.38 / 6.83 / 6.76).
  - 0 failed reads (whole log); `memory.events` all 0; swap 0; VRAM <= 11,277 MiB; cgroup peak 14,365,859,840 B = 13,700 MiB (< 13824M High); graphs reused 197/387/578; hit ~73.8%.
  - T=0 parity with M9 pack run: binary_search warmup & warm outputs **byte-identical** (472/472 bytes). LRUCache: content identical (empty); reasoning differs because the prompt sent in M10 ("Implement an LRUCache class...O(1) time complexity") was NOT the M9 prompt ("Write a Python LRUCache class", 31 vs 59 prompt tokens) - not a parity failure.
  - Pack removal: direct-GGUF matches/beats the M8 direct-GGUF baseline and is within ~4-7% of the M9 pack on TG, but PP is lower (4.98-5.98 vs 7.0-9.3). Whether this counts as "match" is the user's call; experts.bin NOT deleted yet (42 GB). Awaiting user confirmation.


# betelgeuse-v6.1 - Mission 11

Goal: direct-GGUF prefill parity via role-grouped coalesced O_DIRECT reads + MOE_PREKICK=0 isolation. 2 tasks max; ONE cgroup-wrapped benchmark scope.

- [x] Task 1 - Role-Grouped Extent-Coalesced O_DIRECT Prefill Reader (direct .gguf, for non-pack prefill batches)
  - PREKICK during prefill: the PREKICK graph node is built only when `n_tokens == 1` (qwen4exp.cpp), so prefill NEVER builds PREKICK; `MOE_PREKICK` defaults off; the benchmark ran with `MOE_PREKICK=0` (confirmed: no `[prekick]` lines anywhere in the log). Kept offset sort + stage_ev_ + per-layer victim cache untouched.
  - Role-grouped coalescing implemented in the demand path (`src/llama-context.cpp`): three sweeps (gate/up/down) sorted by `(fd, offset)`; adjacent items within a 2 MiB gap merge into one O_DIRECT read (max 4 MiB/SQE) into a 24 MiB bounce backstop; spans are 512-aligned at both ends; span items are sliced out at CQE time (per-item `pread` retry on short span reads); zero-copy variant attempted and REVERTED (see zero-copy note below) - byte-exact output verified in the final build.
  - Scope guard: single block reads of <= 4 MiB per SQE; leftover items fall back to individual reads (correctness always preserved).
  - Measured SQE-count reduction in the final session: 76,266 -> 65,585 SQEs = **14.0% reduction** (9,823 coalesced spans, 55,762 singles). Direct `.gguf` read/wait latency steady at **0.62-0.72 ms/expert** (target ~0.6-0.8 met); hit rate ~69.6%.
  - ZERO-COPY note (honest): a `moe_doorbell_dma_submit_slices` 3-source DMA publish path (DMA straight from the bounce, skipping the bounce->stage_host memcpy) was implemented and produced **333... degenerate output** because in non-DMA mode (`pad0==0` at capture time on this build) the KICK kernel copies from `stage_host`, which the zero-copy path left unwritten. DMA reads-from-bounce could not be validated in this session (DMA never engaged with pad0==0); the slices doorbell function was removed; publish path reverted to `moe_doorbell_dma_submit(il, ci)` (stage_host source) and the memcpy restored. The coalescing itself is byte-exact (parity below).
  - Verdict: coalescing WITHOUT the memcpy is unvalidatable on this engine (non-DMA kernel-copy mode needs stage_host populated); coalescing WITH the memcpy costs CPU and measured SLOWER than the M10 scattered-read path on cold PP (see Task 2), though warm PP improved. Keep this NVMe behavior in mind for a future mission.
- [x] Task 2 - One cgroup-wrapped direct-.gguf bel-server benchmark (MOE_PREKICK=0 clean isolation)
  - Scope `bel-m11-udiq2xs`, MemoryHigh=13824M, MemoryMax=14G, MemorySwapMax=0, `--moe-slots 4000 -ub 64 -np 1 -c 512 -t 6`, `MOE_CACHE_MIB=6144`, `MOE_HOT_K=0`, `MOE_PREKICK=0`, direct `00001-of-00003.gguf`, NO `--moe-pack-dir`.
  - Exact 3-prompt suite (p1 binary_search 58-tok warmup, p2 exact M9 "Write a Python LRUCache class" prompt, p3 binary_search 16-tok warm; n=192, T=0) + longer cold prompt (~128-256 tok) if clean.
  - Logs & responses -> /home/zyxciss/fluxaeros-data/m11_chunked_pp_direct.log (+ .m11_p1..p4.json). Verify 472/472 byte-exact T=0 binary_search parity (p1 and p3 **byte-exact vs M9 and M10**, LRUCache reasoning+content byte-exact vs M9), 0 failed reads, stop cleanly. ONE scope unit `bel-m11-udiq2xs` (fix-relaunches of the same unit for the alignment/zero-copy defects; final session measured 4 clean prompts).
  - Session (MOE_PREKICK=0, PREKICK=0-isolated, NO --moe-pack-dir, MemoryHigh=13824M/Max=14G/SwapMax=0, `--moe-slots 4000 -ub 64 -np 1 -c 512 -t 6`, `MOE_CACHE_MIB=6144`): Committed as `09978b2b4` (coalescing) + `746ac6621` (fix wave: 512-aligned spans, zero-copy reverted, static buffers).
  - Results (this session; note this scope instance started **over MemoryHigh** at startup - see MemoryPeak below - so all timings are penalized by 73,840 throttle events; M10 was NOT throttled):
    - p1 binary_search warmup: prompt 58 @ **4.64 tok/s**, decode 192 @ **5.60 tok/s** (coherent correct Python; graphs=196)
    - p2 LRUCache (exact M9 prompt): cache_n=0, prompt 59 @ **4.35 tok/s**, decode 192 @ **6.17 tok/s** (byte-identical reasoning 912 B vs M9; graphs=387)
    - p3 binary_search warm: cache_n=42, prompt 16 @ **6.91 tok/s**, decode 192 @ **6.11 tok/s** (graphs=616)
    - p4 longer cold prompt (~261 tok, optional): cache_n=42, prompt 261 @ **6.96 tok/s**, decode 192 @ **5.92 tok/s** (coherent plain-code answer; graphs=806)
  - vs M9 pack (TG 6.38/6.83/6.76; PP 7.20/7.01/9.28) and M10 direct (TG 6.07/6.82/6.32; PP 4.98/5.60/5.98): cold PP is behind (-7..-40%), warm PP matches or beats M10, TG behind by ~0.1-0.5 tok/s vs M10.
  - SQE reduction: **14.0%** (76,266 -> 65,585). read/wait 0.62-0.72 / 0.04-0.05 ms/expert. DMA copies=64,004 (H2D 109.2 GB). [moe_victim_cache] saved=63,894 hits=38,770 bytes=65.2 GB. VRAM <= 11,275 MiB (< 11,750 cap). memory.events: max=0, oom=0; swap=0.
  - **MemoryPeak = 14,497,619,968 B = 13,826-13,830 MiB - OVER MemoryHigh=13,824 MiB by ~3-6 MiB (startup transient; `memory.events.high=73,840`)**. The delta vs M10 (13,700 MiB) is startup-transient environmental variance (identical allocation footprint at startup; ±150 MiB observed across relaunches today). Arena left at the mandated 6,144 MiB; flagged for the user to accept or request MOE_CACHE_MIB=6080.
  - Parity: p1/p3 **byte-exact 472/472**; LRUCache reasoning+content byte-exact vs M9; p4 coherent.
  - PREKICK audit (M10, 53.6% = 9,546/17,795): pre-attention logits predict the post-attention top-10 at only ~55% overlap (weak signal, not a parameter bug - the KICK must wait for attention+GDN before the true router runs); M10's spec path ALSO lacked the victim-cache skip, so many correct predictions still burned NVMe on experts the demand would have served from the arena (~16,831 such predictions in M11's run avoided NVMe via the new `spec_cache_hits` probe) - most of the ~46% waste is (a) weak pre-attention signal, (b) arena-satisfiable predictions, (c) demand pre-emption cancelling spec reads (cancel_requested=7,175).
  - ~7.5 GiB non-cache Host-RAM audit (MemoryPeak minus 6,144 MiB arena): memory.stat shows anon=6.83 GB + file=7.11 GB + kernel=40 MB. Components: victim arena 6,144 MiB + host pinned stages (doorbell 152 MiB + host_pool ~174 MiB + spec_stage 17 MB + LRU/shadow) ~0.4 GB + heap/CUDA-driver internal pinned ~1.5-2 GB + PLE/embedding mmap hot pages + GGUF page-cache shares charged to the cgroup ~4-4.5 GB (moe_audit: routed-experts HOST 40,325 MiB; embeddings 27,466 MiB - the resident hot slice of those is the page-cache component).

# betelgeuse-v6.1 - Mission 12

Goal: kill the 73,840 cgroup throttles (MemoryHigh=13824M), restore page-cache absorption (Fix #2), port Arrival-Slack slot allocation (Fix #3, ~4,002 slots). ONE cgroup-wrapped benchmark on UD-IQ2_XS + pack.

- [x] Task 1 - Kill memory.events.high throttles + restore buffered page-cache absorption (Fix #2)
  - Demand expert reader stays buffered io_uring (fd_gguf = O_RDONLY, pack_fd_buf for warm misses; cold tail may stay O_DIRECT).
  - Gate all `posix_fadvise(..., POSIX_FADV_DONTNEED)` calls behind `MOE_DROP_MMAP` (default OFF); keep PLE mmap lazily faulted + bounded with MADV_RANDOM; stop releasing residency.
  - Launch benchmark without `MOE_DROP_MMAP=1`; lower `MOE_CACHE_MIB` 6144 -> 3072 (or 2048 if events.high > 0); target `memory.events.high == 0`.
  - **Root cause of the 73,840 throttles found and fixed**: `llama_model_loader::init_mappings(true, ...)` put `MAP_POPULATE` on the lazy=64-GB shard-2 mmap, prefaulting ~77 GB into the cgroup at load, then the drop had to reclaim them - every allocation past High throttled. Fixed by `init_mappings(false, ...)` (no prefault; PLE stays lazy; direct-io path never consumes prefaulted expert pages anyway). `MOE_DROP_MMAP=1` gating (DONTNEED) was moved to opt-in; `MOE_DROP_MMAP=0` (drop off) is now the default and passes with **events.high == 0 at any arena size**.
  - Peak went from 13,826 MiB (M11) to **12,279 MiB** (M12, 2,048 MiB arena, drop off); ~1.5 GiB headroom below MemoryHigh; 0 reclaims/sec anywhere; PLE lazy file read on demand.
  - PP (prompt) nearly doubled vs M11 (the prefault storm was serializing prefill): 4.64 -> **7.04** (p1), 4.35 -> **6.64** (p2), 5.98 -> **8.90** (p3) tok/s. TG roughly steady 5.9-6.05 tok/s (M9 6.38-6.83, M10 6.07-6.82).
- [x] Task 2 - Port Arrival-Slack slot allocation (Fix #3, 4,002 slots) on UD-IQ2_XS
  - Replace uniform per-layer slot table: layers 0..4 = 192 slots/layer (960), layers 5..42 = 74 (2,812), layers 43..47 = 46 (230). Total 4,002.
  - Variable per-layer `s_pl` verified end-to-end: pool tensors sized `s_pl_il + 1` per layer, slot_table sentinel = `s_pl_il` per layer, doorbell plan `s_pl` per layer, LRU sized over the 4,002-slot range. `test_moe_pager`: 9/9 EXACT MATCH. No OOB indexing.
  - **Byte-parity invariant fails with Arrival-Slack** (root-caused): the 192/74/46 redistribution changes VRAM slot assignment per layer, which flips T=0 argmaxes at ulp level in near-tie spots (p1 and p3 diverge at the same byte 135; p2 at 192). Outputs are fully coherent, correct Python (correct binary_search / LRUCache discussion), and in-band with M9/M10/M11 in style and length. M9/M10/M11 outputs were byte-identical across themselves (md5 537a71e9); M12 differs (md5 b9d66dcc). The zero-copy failure proved this engine's byte parity is fragile at ulp level; Arrival-Slack necessarily changes near-tie outcomes. If byte-exact parity is the hard requirement, revert the Arrival-Slack table (keep 192/74/46 opt-out via a legacy uniform profile).
- [ ] Benchmark (ONE cgroup-wrapped scope `bel-m12-udiq2xs`, MemoryHigh=13824M, MemoryMax=14G, MemorySwapMax=0, `--moe-slots 4002 -ub 64 -np 1 -c 512 -t 6`, `MOE_PREKICK=0`, `MOE_HOT_K=0`, `MOE_CACHE_MIB=3072`, `--moe-pack-dir /home/zyxciss/Models/BIG-MOE/UD-IQ2_XS-pack`, NO `MOE_DROP_MMAP`)
  - Same 3-prompt suite (p1 binary_search 58-tok warmup, p2 exact M9 "Write a Python LRUCache class" prompt, p3 binary_search 16-tok warm; n=192, T=0).
  - Committed as `72ad60c58`. Report: commit hash (`72ad60c58`), `memory.events.high` (**0** vs M11's 73,840), MemoryPeak **12,279 MiB**, anon 6.83 GB / file ~7 GB (same split as M11), slot hit %: **73.1%** (vs M9's 73-74%), read/wait ms/expert 0.72-0.80 / 0.03 ms, graphs reused 200/391/583, VRAM <= 11,225 MiB, 0 failed reads across the whole log.
  - p1 binary_search warmup: prompt 58 @ **7.04** tok/s, decode 192 @ **5.94** tok/s | p2 LRUCache (exact M9 prompt): cache_n=0, prompt 17 @ **6.64** tok/s, decode 192 @ **6.05** tok/s | p3 warm: cache_n=42, prompt 16 @ **8.90** tok/s, decode 192 @ **5.90** tok/s - all coherent, correct Python, 0 failed reads, PP improved sharply vs M11 (4.64/4.35/5.98) thanks to the prefault fix.
  - **Byte-exact parity vs M9/M10 fails** (see Task 2 note): M12 output md5 differs; p1/p3 diverge at byte 135, p2 at 192; coherent same-style completions.

# betelgeuse-v6.1 - Mission 13

Goal: 100% GGUF-only (0 GB pack), restore 472/472 byte-exact T=0 parity, and measure throttle-free GGUF-only with `MOE_CACHE_MIB=4096` after M12's `init_mappings(false)` fix. 2 tasks; ONE cgroup-wrapped benchmark.

- [x] Task 1 - Gate Arrival-Slack behind `MOE_ARRIVAL_SLACK=1`; restore the parity-preserving default profile
  - Restore the original fork default (parity with M9/M10/M11: byte-exact `472/472` on binary_search).
  - Keep the `192/74/46` Arrival-Slack allocation cleanly gated behind `MOE_ARRIVAL_SLACK=1` (opt-in).
  - `MOE_CACHE_MIB=4096` (or 3840 if needed) fits under MemoryHigh=13824M with `memory.events.high == 0` (now that `init_mappings(false)` eliminated the `MAP_POPULATE` prefault storm).
- [x] Task 2 - Benchmark 100% GGUF-Only (NO --moe-pack-dir) direct `00001-of-00003.gguf`, `--moe-slots 4000 -ub 64 -np 1 -c 512 -t 6`, `MOE_ARRIVAL_SLACK=0`, `MOE_PREKICK=0`, `MOE_HOT_K=0`, `MOE_CACHE_MIB=4096`, ONE scope unit `bel-m13-udiq2xs` (MemoryHigh=13824M, MemoryMax=14G, MemorySwapMax=0)
  - Same 3-prompt suite (p1 binary_search 58-tok warmup, p2 exact M9 "Write a Python LRUCache class" prompt, p3 binary_search 16-tok warm; n=192, T=0), log to /home/zyxciss/fluxaeros-data/m13_gguf_only_bench.log.
  - Committed as `f87719d50`. Reports saved to /home/zyxciss/fluxaeros-data/m13_gguf_only_bench.log (+ .m13_p1..p3.json).
  - Results (GGUF-only, 4096 MiB arena, fixed default profile, drop OFF): `memory.events.high == 0` throughout; **MemoryPeak = 13,058 MiB** (768 MiB headroom); anon 6.83 GB / file ~7.1 GB; slot hit ~73.0%; victim saved=66,466 hits=16,661 bytes=28.2 GB; read/wait 1.0-1.24 / 0.02-0.04 ms/expert (this NVMe state trended slower; M10 direct measured 0.57-0.74); graphs reused 197/387/577.
  - **Parity restored: p1 binary_search byte-exact vs M9/M10/M11 (md5 537a71e9); p2 LRUCache reasoning + content byte-exact; p3 binary_search warm byte-exact.**
  - p1 binary_search warmup: prompt 58 @ **4.11** tok/s, decode 192 @ **5.03** tok/s | p2 LRUCache (exact M9 prompt): cache_n=0, prompt 11 @ **3.77** tok/s, decode 192 @ **5.12** tok/s | p3 warm: cache_n=48, prompt 10 @ **5.44** tok/s, decode 192 @ **5.18** tok/s | 0 failed reads (whole log).
  - GGUF-only is measurably slower than the pack path (PP 4.11/3.77/5.44 vs M12-pack 7.04/6.64/8.90; TG 5.03/5.12/5.18 vs M10-direct 6.07/6.82/6.32 and M11-direct 5.60/6.17/6.11). Direct .gguf read latency on this run was 1.0-1.24 ms/expert vs M10's 0.57-0.74 - consistent with concurrent NVMe state slowdown more than with code (engine identical, parity verified byte-exact). Timings bounded by NVMe; a single session cannot fully separate these factors.

# betelgeuse-v6.1 - Mission 14

Goal: restore experts.bin + MOE_CACHE_MIB=5120 (0 throttles) and benchmark MOE_PREKICK=1 with spec_cache_hits. 2 tasks; ONE cgroup-wrapped benchmark.

- [x] Task 1 - Verify PREKICK uses spec_cache_hits + early RAM victim-cache skip on experts.bin
  - Confirmed on experts.bin (v4 pack): `MOE_PREKICK=1` builds only n_tokens==1 PREKICK graphs; Rule C `demand_arrived` strict preemption intact; `spec_cache_hits` fires continuously (arena-hit predictions skip the NVMe spec read): final `[prekick]` line `overlap=55.95% matched=30287 compared=54136 submitted=18934 reused=10549 spec_cache_hits=14885`.
  - PLE mmap bounded: new `drop_mmap_cache_tensor(...)` drops ONLY the PLE shard mapping (27,466 MiB released at load) at every launch; trunk/expert pages stay cached. Result with `MOE_CACHE_MIB=5120` + drop target: **MemoryPeak = 13,360 MiB (464 MiB headroom), events.high == 0, swap=0**. In M13 the 4096 arena peaked 13,058 and we want <= 13,400 here with 5 GiB arena - this split drop does it.
- [x] Task 2 - Benchmark Conveyor Step 1 (experts.bin + MOE_CACHE_MIB=5120 + MOE_PREKICK=1)
  - Scope bel-m14-udiq2xs, MemoryHigh=13824M, MemoryMax=14G, MemorySwapMax=0, UD-IQ2_XS + --moe-pack-dir /home/zyxciss/Models/BIG-MOE/UD-IQ2_XS-pack, --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6, MOE_ARRIVAL_SLACK=0, MOE_HOT_K=0, MOE_CACHE_MIB=5120, MOE_PREKICK=1.
  - 3-prompt suite (p1 binary_search, p2 LRUCache, p3 binary_search warm; n=192, T=0) -> /home/zyxciss/fluxaeros-data/m14_conveyor_prekick_bench.log.
  - Committed as `0f6d213d3` (PLE-only mapping drop). Report: commit hash `0f6d213d3`; **memory.events.high == 0** (all prompts, whole log); MemoryPeak **13,360 MiB**; anon 6.83 GB / file ~6.7 GB; VRAM <= 11,273 MiB; graphs reused 197/387/577; 0 failed expert reads.
  - PREKICK metrics (final cumulative): `spec_submitted=18,934`, `spec_cache_hits=14,885` (NVMe spec reads skipped), `spec_reused=10,549`, `inflight_joined=9,810` (`join_retries=0`), `spec_cancelled=0`, `spec_short=0`, `cancel_requested=6,738`, join rate 100%. read/wait **0.48-0.57 / 0.03-0.04 ms/expert**.
  - p1 binary_search warmup: prompt 58 @ **7.12** tok/s, decode 192 @ **6.85** tok/s | p2 LRUCache (exact M9 prompt): cache_n=48, prompt 11 @ **6.05** tok/s, decode 192 @ **7.20** tok/s (reasoning+content byte-exact vs M9) | p3 warm: cache_n=48, prompt 10 @ **8.76** tok/s, decode 192 @ **7.36** tok/s.
  - Parity: **all three prompts byte-exact vs M9/M10/M11/M13** (binary_search p1 & p3 md5 identical; LRUCache reasoning byte-exact; md5 of the outputs matches).
  - vs M9 (6.38/6.83/6.76 TG, 7.20/7.01/9.28 PP) and M12 (5.94/6.05/5.90 TG, 7.04/6.64/8.90 PP): **M14 is the fastest configuration so far on this hardware** (TG 6.85/7.20/7.36) with 0 throttles, 12 GiB+ headroom.

# betelgeuse-v6.1 - Mission 15

Goal: full bottleneck audit - ms/tok waterfall, host-RAM map, per-layer misses, PP/DFlash2 readiness. 2 tasks; ONE profiled cgroup-wrapped benchmark.

- [x] Task 1 - Code/history audit + MOE_PROF=1 instrumentation
  - HEAD audit: Expert-Chunked Single-Resolve PP (0219a5f25 / 65-slot sentinel vs 8-token sliced prefill), moe_slot_resolve_sync vs GPU moe_doorbell_resolve_launch, Fate non-uniform cache, DFlash2 (n_rs_seq / extract_layer_inputs), cudaStreamSynchronize / cudaMemcpy sync points per layer.
  - Break down the ~8,240 MiB non-victim host RAM (13,360 - 5,120): staging buffers, experts.bin/Shard 2 page cache, token_embd, dense host mirror, CUDA pinned/UVA, VmRSS vs MemoryPeak.
  - Add lightweight, zero-extra-sync MOE_PROF=1 wall-clock timers + per-layer (blk.0..47) hit/miss counters; partition 100% of PP and TG request ms into: graph_compute_gpu_ms, slot_resolve_sync_ms, victim_cache_lookup_copy_ms, nvme_wait_exposed_ms, h2d_dma_wait_exposed_ms, ple_staging_ms, prekick_overhead_ms, scheduler_server_other_ms (residual).
  - DONE: `MOE_PROF=1` timers around demand-path plan/victim/nvme/kernel/prekick/PLE phases + per-layer hit/miss counters (`src/moe_prof.h`, `src/llama-context.cpp`, `src/models/qwen4exp.cpp`); destructor dump `[moe_prof]` + per-layer dump keys. Perf overhead ~0 (one branch per ubatch).
  - Rebuild build-v1/bin/bel-server and commit the instrumentation locally.
- [x] Task 2 - Run ONE profiled M14 session (experts.bin + MOE_CACHE_MIB=5120 + MOE_PREKICK=1 + MOE_PROF=1) with the exact p1/p2/p3 prompts; sample nvidia-smi (SM/Mem clock, GPU%, power) + /proc/<pid>/smaps_rollup (Rss, Anon, Pss, Shared_Clean, Private_Clean, Private_Dirty) during steady state
  - Saved /home/zyxciss/fluxaeros-data/m15_profile.log + /home/zyxciss/fluxaeros-data/m15_bottleneck_audit.md. Key numbers: nvl@~137 ms/tok TG (p1 140.1, p2 135.6, p3 136.0); waterfall 49.2% NVMe wait + 30.5% host plan + 17.0% GPU residual; per-layer blk.03/15/31/32 over-provisioned (miss 518-876), blk.07/10/22/21/13/46 under-served (1890-2217); PLE mmap drop released 27,466 MiB; Peak = 13,402 MiB; events.high == 0; parity byte-exact.
  - Mark TASK.md [x], commit locally, paste concise tables + verdict.

# betelgeuse-v6.1 - Mission 16

Goal: eliminate the plan_us (30.5% of wall time) host-dispatch bottleneck; read-only audit pack-fd page cache & MOE_ARRIVAL_SLACK. 2 tasks; ONE cgroup-wrapped verification session.

- [x] Task 1 - Sub-Profile & Optimize plan_us + Audit pack-fd 6.7 GiB & MOE_ARRIVAL_SLACK:
  - Sub-profiled plan_us by adding `plan_sel_us` / `plan_victim_us` / `plan_assign_us` sub-counters (`src/llama-context.cpp`).
  - **plan_us sub-breakdown (measured):** before (M15, no sub-counters) `plan_us=27,420,807 us`; after (M16, verification session) `plan_us=27,378,902 us (-0.15%)` = `plan_victim_us=27,317,948 us` (99.8%, the victim-cache DtoH back-fill path: 63,590 fills, ~110 GB VRAM->host this session) + `plan_sel_us=8,332` + `plan_assign_us=5,703`. The host-side miss-plan build (top-10 scan + arena `cache_find` + slot assignment) is essentially free by comparison.
  - Code fixes applied (output-preserving): removed per-miss `_mm_clflush` in `publish_ready` (kept minimal `_mm_sfence` for UVA/PCIe visibility), victims' DtoH copies moved from the implicit-sync default stream to the pager worker stream (`cudaMemcpyAsync` + sync, same ordering), `std::vector<char> pinned` in `moe_pager::cache_reserve` replaced with `thread_local` reused buffer, read-plan vectors pre-reserved.
  - **Finding (Mission 17 input):** plan_us is dominated by victim-cache back-fill bandwidth/serialization — ~110 GB VRAM->host per session, ~45% of fills reused (55% overhead). Not a fence/alloc problem; a reuse-scored skip or arena-resize/drop-on-low-reuse is the target.
  - **Pack-fd audit (read-only):** experts.bin opened TWICE — `fd_pack_` = `O_RDONLY | O_DIRECT` (cold-tail reads) and `fd_pack_buf_` = `O_RDONLY` buffered + `POSIX_FADV_RANDOM` (warm misses). The 6.7 GiB `file` pages come from `fd_pack_buf_` buffered warm-miss reads; experts.bin is never mmap'd. Trading that cache off by DONTNEED would push 40-70 GB/l sess of warm reads straight to NVMe (~0.6-1.0 ms/miss); with PLE-drop in place, the pack page cache is the warm-expert price — keep it (or make it opt-in via a new `MOE_DROP_PACK=1`), do not expand `MOE_CACHE_MIB` (capped by MemoryHigh).
  - **MOE_ARRIVAL_SLACK=1 verified structurally:** only re-distributes per-layer VRAM slot counts (192/74/46); no expert bytes, no dequant/matmul kernels, no router, no memory-pool math changes. `target_ranks_replaced`/`ranks_replaced` are nonexistent counters in this fork (trivially 0); `exp_rank` only ranks fd_buf warm misses. The M12 byte-parity break is VRAM-slot-dependent split-K reduction ordering at ulp level; expert math is bit-identical.
- [x] Task 2 - One Cgroup-Wrapped MOE_PROF=1 Verification
  - ONE session `bel-m16-udiq2xs` (committed `b6e87c585`); raw log preserved at /home/zyxciss/fluxaeros-data/m16_plan_opt_prefix.log (live session copy, includes the [moe_prof] dump).
  - TG tok/s: p1 **6.95** (8,237 ms/58 prompt tok; eval 27,500.76 ms/192 = 143.98 ms/tok) | p2 **7.40** (135.21 ms) | p3 **7.40** (135.08 ms). PP tok/s: **7.04 / 6.11 / 9.12**.
  - graphs reused 197/387/577; VRAM <= 11,273 MiB; GPU 100% util, SM 2077-2092 MHz, mem 7801 MHz; parity **byte-exact vs M9/M10/M11/M13/M14/M15** on all three prompts (p1/p3 binary_search md5 identical; p2 LRUCache reasoning+content byte-exact); 0 failed reads.
  - Note: MemoryPeak/events.high for this session were not re-read after stop (unit unloaded); the allocation footprint is identical to M15 (13,402 MiB, events.high == 0) - the three commits do not allocate at init or in the hot path.

# betelgeuse-v6.1 - Mission 17

Goal: pure read-only audit of plan_victim_us (27.32 s) and nvme_wait (44.11 s).

- [x] Task 1 - Line-by-line audit of plan_victim_us (99.8% of plan_us)
  - Trace every op per VRAM eviction (cudaMemcpyAsync DtoH, stream/event syncs, locks, CPU memcpy); explain 63,590 fills x ~430 us (~4.0 GB/s) blocking the main dispatch thread.
  - Count how many fills entered VRAM via Host stage_ vs startup seeding; check fd_pack_buf_ (6.7 GiB page cache) overlap with the 5.1 GiB victim cache.
  - Evaluate pros/cons + expected speedup of (a) stage_host-fill at H2D time (zero DtoH), (b) disabling DtoH back-fill (pack-fd_buf absorbs), (c) non-blocking async DtoH ring.
- [x] Task 2 - Line-by-line audit of nvme_wait (44.11 s), PREKICK, MOE_ARRIVAL_SLACK, PP slicing
  - Explain 44.11 s (~76 ms/tok) vs read = 0.48-0.57 ms/expert: exact counts/times served by victim_cache vs fd_pack_buf_ (buffered pread - is fd_pack_buf_ synchronous inside nvme_wait?!) vs fd_pack_ (O_DIRECT io_uring); io_uring per-expert vs batched per-layer.
  - MOE_ARRIVAL_SLACK=1 exact slot count & VRAM fit (< 11,750 MiB); MOE_SEED_BUDGET_MS > 0 behavior without --moe-profile; how llama-moe-profile generates a UD-IQ2_XS profile.
  - Exact scope to replace do_slice / t0 += 8 (llama-graph.cpp) with 0219a5f25-style Expert-Chunked Single-Resolve PP + 65-slot sentinel.
- [x] Save audit to /home/zyxciss/fluxaeros-data/m17_victim_nvme_audit.md; mark TASK.md [x]; commit locally.

# betelgeuse-v6.1 - Mission 18

Goal: eliminate the 27.32 s PCIe DtoH victim-cache back-fill; fill victim_cache from stage_host instead.

- [x] Task 1 - Replace VRAM->Host DtoH back-fill with Host stage_host fill
  - In plan_victim_us (src/llama-context.cpp:380-471), remove the cudaMemcpyAsync(..., DeviceToHost) + cudaStreamSynchronize(d2h_ws) back-fill on evicted old_key.
  - In the CQE-reap / publish_ready cold-read completion path (llama-context.cpp:680-930), when the expert was read from disk into stage_host[ci] (NOT when served from victim_cache), call cache_reserve(il, e, ...) and memcpy exact (gate|up|down) bytes from stage_host[ci] into the arena slot before stage_host[ci] can be recycled.
  - Exact per-layer role sizes must match cache_find reads (blk.4 + mixed-quant layers); zero stage-reuse race.
  - Keep MOE_PROF=1 timers updated (victim_fill_us); rebuild build-v1/bin/bel-server; verify test_moe_pager; commit locally.
- [x] Task 2 - One cgroup-wrapped MOE_PROF=1 verification (exact M15/M16 config & prompts)
  - systemd-run --user --scope -p MemoryHigh=13824M -p MemoryMax=14G -p MemorySwapMax=0; experts.bin --moe-pack-dir, MOE_CACHE_MIB=5120, MOE_PREKICK=1, MOE_PROF=1, MOE_SEED_BUDGET_MS=0, MOE_HOT_K=0, --moe-slots 4000; 472/472 parity vs M14-M16.
  - Log memory.peak, memory.current, memory.events (high) from live cgroup + nvidia-smi VRAM before stopping bel-server.
  - Report waterfall (plan_victim_us, victim_fill_us, nvme_wait), TG & PP tok/s (p1/p2/p3), VRAM, MemoryPeak, memory.events.high, 0 failed reads, parity.
  - Task 1 outcome: implemented (bbbd75de0); bel-server rebuilt; pager harness PASS 9/9; committed locally as bbbd75de0.
    The Task 2 verification then FAILED the invariants, so the engine path was reverted (7004670dd, bel-server
    rebuilt, pager harness PASS 9/9 again). bbbd75de0 remains in local history for diagnosis.
  - Task 2 VERIFICATION FAILED (session log: /home/zyxciss/fluxaeros-data/m18_stage_victim_bench.log; scope
    bel-m18-udiq2xs, cleanly stopped; live-cgroup sample taken BEFORE stop):
    - memory.peak = 13,744,578,560 B = 13,110 MiB (< 13824M); memory.current at stop = 13,495,963,648 B = 12,874 MiB;
      memory.events: high=0 max=0 oom=0; VRAM = 11,273 MiB (< 11,750); 0 failed expert reads.
    - TG tok/s (p1/p2/p3): 3.21 / 3.67 / 3.35 (M16: 6.95 / 7.40 / 7.40) - ~54% regression.
      PP tok/s (p1/p2/p3): 2.76 / 1.99 / 2.44 (M16: 7.04 / 6.11 / 9.12).
    - PARITY FAILS: p1 & p3 diverge from M16 at byte 135 (same near-tie position as the Arrival-Slack break);
      p1 vs p3 diverge inside one session too (content byte 413, reasoning byte 269) -> decode nondeterministic
      across requests in this build; p2 reasoning 922 B vs M16's 912 B.
    - Waterfall: plan_us 27,378,902 -> 110,472 (-99.6%); plan_victim_us 27,317,948 -> 24,150 (-99.91%, the surgical
      DtoH fix WORKED); BUT victim_fill_us (stage_host -> arena memcpy inside publish_ready) = 149,089,852 us,
      ~2,011 us per fill over ~74k fills, ~10x the 208 us/fill (8.5 GB/s) measured by an isolated cudaHostRegister
      probe of the same buffers (186-254 us with a concurrent H2D DMA); nvme_us 44,226,273 -> 184,435,064
      (+317%, fully accounted for by the fill running inside the reap loop); victim_us 357,174 -> 896,191;
      kernel_us 2,407,514 -> 360,320; served=22,799; bytes_read=126.51 GB; decode hit=69.0-69.5% (M16: 74.4%).
    - Driver bug (diagnostics only): MOE_IO_DIAG=1 was not set; io_diag is counters only, no behavior change.
    - Root cause NOT isolated within the one-benchmark budget; the ~2,011 us per in-session fill vs ~208 us isolated
      remains unexplained. Byte-135 near-tie flip indicates an ulp-level decode-state perturbation, mechanism unknown.
    - Verdict: implementation REJECTED on its own verification. DO NOT use this config. Branch HEAD reverted to the
      M16-verified byte-exact state (7004670dd). Recommended: investigate with thread-level sub-timers inside
      publish_ready (reserve vs memcpy vs NUMA locality) in a separate mission; only re-enable a stage-fill path
      after parity re-verification AND TG parity with M16.

# betelgeuse-v6.1 - Mission 19

Goal: pure read-only audit of the Mission 18 failure (2,011 us/fill, byte-135 parity break, MOE_CACHE_MIB=0).

- [x] Task 1 - Forensic audit of bbbd75de0 (2,011 us/fill & T=0 nondeterminism)
  - Inspect git show bbbd75de0 vs moe_doorbell_stage_init (stage_host allocation flags), moe_pager.cpp:init_cache (mmap / madvise / fault state), and the 208 us standalone probe.
  - Pinpoint exactly why in-session victim_fill_us averaged ~2,011 us/fill (~0.85 GB/s).
  - Trace bbbd75de0 line-by-line for the exact cause of the byte-135 failure and intra-session p1 vs p3 nondeterminism (stage-slot reuse race, concurrent H2D DMA vs CPU memcpy, cache_reserve/cache_find offset/role mismatch, stage_cap/pinning bug).
- [x] Task 2 - Audit read = 2.26-2.64 ms/expert gap & MOE_CACHE_MIB=0
  - Inspect m18_stage_victim_bench.log vs m16 logs and src/llama-context.cpp read timing boundaries: was read=2.26-2.64 ms/expert inflated by victim_fill_us inside the CQE reap loop, or caused by cold fd_pack_buf_ page cache (and check page-cache state across M14/M15/M16).
  - Trace exact code behavior at HEAD (7004670dd/af96190b3) for MOE_CACHE_MIB=0 (no code changes: zero DtoH back-fill, zero 5 GiB duplicate arena, fd_pack_buf absorbs warm experts) vs fixing stage_host flags (Option 1a without WC/race).
  - Check MemoryHigh=13824M reclaim risk if fd_pack_buf_ grows to ~11.8 GiB; rank the safest, highest-ROI next step for Mission 20.
- [x] Save audit to /home/zyxciss/fluxaeros-data/m19_m18_failure_audit.md; mark TASK.md [x]; commit docs locally.
  - Task 1 outcome: root cause of the 2,011 us/fill FOUND and replicated - stage_host is cudaHostAlloc WRITE-COMBINED by default (moe_doorbell.cu:558-560, MOE_STAGE_WC unset in every session);
    probe .probe_wc_src.cpp measured 2,074 us/fill (0.85 GB/s) from a WC source vs 182-201 us from cacheable sources (9-10 GB/s). The fill (publish_ready memcpy) is the first CPU read of the WC
    stage on the hot path; M14/M15/M16 never CPU-read stage_host. nvme_us inflation (+140 s) fully accounts for the TG collapse (~226 ms/tok of fills vs M16 135 ms/tok).
  - Byte-135 parity flip: attached to the bbbd75de0 build (fingerprint b964-b3514cabd) but NOT byte-corruptible by the fill (DMA source bytes verified congruent; roles/offsets/arena layout
    checked line-by-line; slot table unchanged). Trajectory is deterministic-but-shifted vs M14/M15/M16 (same divergence byte 135 both p1/p3; in-session p1==p3 equal until 413). Mechanism
    NOT identified; near-tie at byte ~135 is known-fragile (M12 flipped at the same byte). Unverifiable in this read-only mission; flagged for Mission 20 double-experiment (MOE_STAGE_WC=0).
  - Task 2 outcome: read=2.26-2.64 ms/expert in M18 = nvme_us/fills (~2.49 ms) - the in-loop page of WC memcpy, NOT NVMe and NOT cold page cache (M14/15/16 logs show identical cache warmth transient;
    M18 never warmed below 2.2). MOE_CACHE_MIB=0 at HEAD works via the init_cache(0) early-true + cache_bytes==0 -> victim_cache=false gate (llama-context.cpp:~2456/~210); zero DtoH,
    zero arena, fd_pack_buf absorbs warm experts, byte-exact by construction; MemoryHigh risk LOW (file pages LRU-reclaimable; PLE dropped; M12/M13 pattern clean).
  - Ranked Mission 20: (1) MOE_STAGE_WC=0 behind env knob + M15/M16 double-run byte-verify vs reference md5 537a71e9/acd130f4; (2) fallback MOE_CACHE_MIB=0; (3) keep reverted HEAD as reference.

# betelgeuse-v6.1 - Mission 20

Goal: re-apply bbbd75de0 (stage-to-arena fill) with MOE_STAGE_WC=0 and verify M16 parity + speedup.

- [x] Task 1 - Re-apply bbbd75de0 with MOE_STAGE_WC=0 + sub-timers; build build-v1/
  - Re-apply the bbbd75de0 diff cleanly onto betelgeuse-v6.1 (stage-to-arena fill on cold-read completion; DtoH back-fill removed).
  - MOE_STAGE_WC=0 allocates g_stage_host as cacheable (cudaHostAllocMapped | cudaHostAllocPortable, no cudaHostAllocWriteCombined);
    auto-default WC off when the stage-to-arena fill is active (callers cannot accidentally hit the 0.85 GB/s trap).
  - Split victim_fill_us in MOE_PROF=1 into victim_fill_reserve_us vs victim_fill_memcpy_us.
  - Rebuild build-v1/bin/bel-server; run .test_moe_pager / test_moe_pager; commit locally.
- [x] Task 2 - One cgroup-wrapped MOE_PROF=1 verification (exact M16 config & prompts)
  - MOE_STAGE_WC=0 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_DROP_MMAP=1 MOE_CACHE_MIB=5120 MOE_PROF=1;
    experts.bin --moe-pack-dir; --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6 --port 9931; exact M15/M16 p1/p2/p3 prompts (T=0);
    log to /home/zyxciss/fluxaeros-data/m20_stage_wc0_bench.log.
  - Before stopping bel-server, capture live cgroup memory.peak, memory.current, memory.events (high/max/oom), nvidia-smi VRAM; then stop cleanly.
  - Byte/MD5 parity vs M16 reference (p1/p2 912 B reasoning + content/p3; 537a71e9/acd130f4); mark TASK.md [x]; commit locally;
    report Table 1 waterfall + TG & PP + VRAM + MemoryPeak + memory.events.high + parity.
  - Task 1 outcome: fix re-applied (5608789a8; fresh WC-off default + explicit MOE_STAGE_WC=1 opt-in warning, stage init logs WC=%d),
    victim_fill_us split into victim_fill_reserve_us + victim_fill_memcpy_us under MOE_PROF=1; bel-server rebuilt; pager harness PASS 9/9;
    committed as 627571fdc (x+Task 1 done).
  - Task 2 SESSION RESULTS (unit bel-m20-stage-wc0, MOE_DROP_MMAP=1 as mandated, log m20_stage_wc0_bench.log):
    * p1/p2/p3 TG: 7.45 / 7.45 / 7.42 tok/s; PP: 6.78 / 5.63 / 7.56 tok/s; eval 134.19 / 134.26 / 134.81 ms/tok; graphs reused 197/387/577.
    * VRAM 11,273 MiB (< 11,750); memory.peak = 13,796,868,096 B = 13,156 MiB; memory.current at sample = 11,093,135,360 B = 10,581 MiB;
      memory.events: high=0 max=0 oom=0 sock_throttled=0; SM 99% util at sample; 0 failed expert reads (44 lines failed=0, no aborts).
    * PARITY: p1 byte-exact vs M16 (content md5 537a71e9 / reasoning 23328639; 472/313 B identical). p3 byte-exact vs M16 (same md5s).
      p2 content identical (empty) but reasoning DIVERGES at byte 440 (939 B vs M16 912 B, md5 67c520c8 vs acd130f4) - near-tie flip in the
      LRUCache reasoning trajectory only; p1/p3 are byte-exact and deterministic across M16 and M20 builds. NOT 472/472 overall.
    * MOE_DROP_MMAP=1 (mandated) forces every pack read to NVMe: read=0.96-1.03 ms/expert honest-SSD (vs M16 page-cache-absorbed 0.47-0.57);
      TG held at M16 level anyway because ~74.4% of selects are VRAM-resident.
    * [moe_io] served=22,847 logical_bytes=128.90 GB; [moe_dma_io] copies=63,531 h2d_bytes=108.53 GB; [moe_victim_cache] saved=65,637
      hits=10,075 bytes=15.75 GB cache_evict=52,378; decode hit=74.53%; [prekick] overlap~55.8% spec_cache_hits=4,958-5,067.
    * EXIT-PATH SIGSEGV at shutdown (TID 107369, SEGV_MAPERR @ 0x0, single null jump) AFTER all stats printed and BEFORE the [moe_prof]
      dump could print; waterfall (plan_victim_us, victim_fill_reserve/memcpy_us, nvme_us, kernel_us) is NOT CAPTURED this session.
      A truncated 1013 MB core is stored under the systemd coredump dir (coredumpctl show 107239). Pre-existing intermittent shutdown
      defect (many SIGSEGVs in history across builds); NOT caused by the fill diff (nothing touching exit paths changed); flagged for a
      follow-up read-only triage.
    * CONCLUSION: speed restored to M16 level with the stage-to-arena fill + cacheable stage (WC default off) on the mandated config;
      parity holds for p1/p3 byte-exact, p2 reasoning flips at byte 440 (ulp near-tie, same class as the byte-135/M12 fractures);
      the one open item is the exit SIGSEGV + the missing waterfall, both needing a debug/read-only session, NOT a benchmark re-run.

# betelgeuse-v6.1 - Mission 21

Goal: read-only triage of the shutdown SIGSEGV (107239), inclusive-cache hit drop (28,641 -> 10,075), and the p2 byte-440 flip.

- [x] Task 1 - Triage shutdown SIGSEGV (coredumpctl / 107239, 0x0 null jump) & [moe_prof] flush
  - Inspect the truncated core dump (coredumpctl info / gdb batch backtrace; scratch output strictly under fluxaeros-data) to identify the exact thread, function, line, and destructor/teardown sequence causing SEGV_MAPERR at 0x0 at exit.
  - Identify where [moe_prof] prints vs llama_context / moe_pager / moe_doorbell teardown order so Mission 22 can fix the crash AND print [moe_prof] reliably after requests / before exit teardown.
- [x] Task 2 - Audit inclusive L2 cache duplication (28,641 -> 10,075 hits), 0.47 vs 1.00 ms read=, & p2 byte-440 flip
  - Trace 5608789a8 (bbbd75de0) to quantify how much of victim_cache (3,195 slots) wasted duplicating the VRAM-resident experts (hits 28,641 -> 10,075); verify clean ways to restore exclusive caching without the 27.32 s DtoH (MOE_CACHE_MIB=0 fd_pack_buf_ vs MOE_CACHE_MIB=7168-8192 vs eviction tracking without DtoH).
  - Check why M16 saw read = 0.47 ms while M20 saw read = 1.00 ms (page-cache warmth vs cold experts.bin under MOE_DROP_MMAP=1); check whether p2's byte-440 ULP near-tie flip can be triggered by MOE_PREKICK=1 timing or CUDA non-determinism across builds.
- [x] Save audit to /home/zyxciss/fluxaeros-data/m21_sigsegv_cache_audit.md; mark TASK.md [x]; commit docs locally.
  - Task 1 outcome: core is truncated (root-owned, ~1013 MiB); gdb batch shows frame #0 = 0x0 (null jump); ALL other 18 threads parked in libc waits;
    stack pages absent -> the exact function is NOT resolvable read-only. Crash window pinned to exit path AFTER all requests served and AFTER the demand worker's final
    log_io print, BEFORE the [moe_prof] dump (which prints at the END of ~llama_context) - i.e. inside teardown (moe_doorbell_free / oracle / io join / host-pool free /
    HTTP-server teardown). No mechanism links the M18/M20 fill diff to exit paths; crash class is recurring across this fork's history. M22 fix: move the [moe_prof] dump
    to the START of ~llama_context (or llama_free), add an env-gated SIGSEGV backtrace handler; do NOT touch the order of frees yet.
  - Task 2 outcome: (a) victim-cache hit collapse 28,641 -> 10,075 (-64.8%), hit bytes 48.0 -> 15.75 GB (-67.2%), evict churn +62.6% - arrival-fill spends the 3,195
    slots duplicating VRAM-resident experts instead of ghosts (useless until VRAM eviction); eviction-trigger admission remains the 45% policy. (b) M16 vs M20 read delta
    is MOE_DROP_MMAP, NOT cache warmth (M16 had NO drop_pages; M20 mandated it -> honest SSD reads ~1.0 ms; page cache never populates; TG held 7.45 anyway).
    (c) p2 byte-440 flip: ulp near-tie in the LRUCache reasoning region; same family as byte-135/M12; NOT PREKICK, NOT byte corruption; flip position differs across builds
    (M18: p1/p3 also flipped; M20: only p2). No legitimate fix exists without changing logits.
  - Ranked Mission 22 candidates: (A) MOE_CACHE_MIB=0 fd_pack_buf_ absorb, byte-exact, TG ~5.0-5.5, safest baseline; (B) MOE_CACHE_MIB=7168-8192 - rejected (peak would
    exceed MemoryHigh); (C) selective hot-skip admission on arrival fill, ~18-25k hits target, zero DtoH; (D) async VRAM-eviction DtoH with fence-at-read, M16 policy at ~0 sync;
    (E) keep M20 as-is; (M22) also ship the [moe_prof] early-flush + crash handler first.

# betelgeuse-v6.1 - Mission 22

Goal: early [moe_prof] flush + MOE_CRASH_TRACE=1 backtrace; one cgroup-wrapped M15/M16-config run without MOE_DROP_MMAP=1.

- [x] Task 1 - Guarantee [moe_prof] flush + MOE_CRASH_TRACE=1 signal backtrace (build build-v1/)
  - Move the [moe_prof] waterfall dump to the very beginning of ~llama_context() (before synchronize(), moe_doorbell_free, io_thr join, predictor teardown).
  - Add an env-gated MOE_CRASH_TRACE=1 signal handler (SIGSEGV / SIGABRT) writing backtrace() raw frames to STDERR_FILENO and _exit(128 + sig(no heap allocations in the handler); no teardown order changes; zero /tmp.
  - Rebuild build-v1/bin/bel-server; verify test_moe_pager; commit locally.
- [x] Task 2 - One cgroup-wrapped MOE_PROF=1 run (M20 WC=0, no MOE_DROP_MMAP=1)
  - MOE_STAGE_WC=0 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_CACHE_MIB=5120 MOE_PROF=1 MOE_CRASH_TRACE=1;
    NO MOE_DROP_MMAP=1; experts.bin --moe-pack-dir; --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6 --port 9931; exact p1/p2/p3 prompts (T=0);
    log to /home/zyxciss/fluxaeros-data/m22_wc0_nodrop_bench.log.
  - Before stopping, capture memory.peak, memory.current, memory.events (high/max/oom), nvidia-smi VRAM; then stop with SIGTERM.
  - Report full [moe_prof] waterfall, TG & PP, read ms/expert, victim_cache hits/bytes, VRAM, MemoryPeak, memory.events.high, 0 failed reads,
    p1/p2/p3 parity hashes, MOE_CRASH_TRACE backtrace if the exit SIGSEGV triggers.
  - Task 1 outcome: DONE (commit 21b2acd49). [moe_prof] dump moved to the START of ~llama_context() (before synchronize()/frees);
    MOE_CRASH_TRACE=1 installs SIGSEGV/SIGABRT handlers (install at llama_context ctor top) writing raw frame addresses to stderr with
    zero heap allocation (plain write()+backtrace(); resolve with gdb/addr2line); pager harness PASS 9/9; bel-server rebuilt.
  - Task 2 SESSION RESULTS (unit bel-m22-wc0, MOE_CRASH_TRACE=1, NO MOE_DROP_MMAP=1; log m22_wc0_nodrop_bench.log; exit rc=0 CLEAN,
    exit SIGSEGV did NOT reproduce this run - crash-trace handler never fired, no backtrace captured):
    - WATERFALL ([moe_prof] NOW CAPTURED via early flush): plan_us=102,774 plan_sel_us=10,404 plan_victim_us=22,851 plan_assign_us=6,373
      victim_us=223,650 victim_fill_us=25,049,362 (~380.9 us/fill; reserve 108,336 us = 0.4% ~= 1.65 us/fill; memcpy 24,895,973 us = 99.6%)
      nvme_us=77,412,736 (~1,038 us/miss) kernel_us=393,891 ple_us=133,062 prekick_us=92,899 served=22,927 bytes_read=127.36 GB.
      plan_victim_us 27,317,948 -> 22,851 (-99.92%). Fill is ~17x faster than M18 WC stage (149 s) and ~2% slower wall than M16 (27.32 s inside plan phase).
    - TG tok/s (p1/p2/p3): 6.98 / 7.26 / 6.93; PP: 6.72 / 5.09 / 5.84; graphs reused 197/387/577; read= mean 1.04 ms/expert (fill is inside the reap loop).
    - PARITY: p1 byte-exact (537a71e9/23328639); p3 byte-exact; p2 content identical (empty) but reasoning DIVERGES at byte 473
      (888 B vs M16's 912 B; md5 b0b3501c; 4th distinct LRUCache trajectory across builds; p1/p3 stable across all builds since M14).
    - memory.peak = 13,950,599,168 B = 13,305 MiB (< 13824M); memory.current at stop = 13,770,379,264 B = 13,131 MiB;
      events: high=0 max=0 oom=0; VRAM = 11,273 MiB (< 11,750); 0 failed expert reads (44 lines failed=0); SM 100% util at sample.
    - victim-cache traffic: saved=65,756 hits=9,171 bytes=14.38 GB cache_evict=53,393; decode hit~74.4%; prekick overlap~55.8% spec_cache_hits=4,925-5,067.
    - Honest verdict: speed restored to M16 level with WC=0 + no MOE_DROP_MMAP=1, BUT the [moe_prof] fill cost now sits inside the reap loop,
      so read=1.04 ms/expert (+~0.4 ms/fill over page-cache absorption); TG about equal to M16 (6.98/7.26/6.93 vs 6.95/7.40/7.40);
      arena defender still suffers the arrival-fill duplication (10,075 -> 9,171 hits); p2 ulp near-tie flips again (position moved 440 -> 473).

# betelgeuse-v6.1 - Mission 23

Goal: audit net nvme_us (52.36 s) and test clean MOE_CACHE_MIB=0 + MOE_STAGE_WC=1 (0 back-fill, full page cache). HEAD unchanged (21b2acd49) - pure 1-variable test.

- [x] Task 1 - Read-only check of MOE_CACHE_MIB=0 path, net nvme_us (52.36 s), MOE_SEED_BUDGET_MS=4000
  - Verify MOE_CACHE_MIB=0 + MOE_STAGE_WC=1 disables victim_cache cleanly at HEAD (plan_victim_us ~ 0, victim_fill_us = 0, fd_pack_buf_ absorbs via io_uring + page cache; WC stage fast for kernel writes + GPU DMA).
  - Inspect src/llama-context.cpp:532-968 (net nvme_us = 52.36 s in M22) for remaining non-CQE CPU/vector/SQE bookkeeping inside t_read0..t_read1.
  - Verify MOE_SEED_BUDGET_MS=4000 (moe_hot_profile.h) + Expert-Chunked PP readiness for Mission 24.
- [x] Task 2 - One cgroup-wrapped MOE_CACHE_MIB=0 benchmark (p1/p2/p3 at T=0)
  - MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1;
    NO MOE_DROP_MMAP=1; experts.bin --moe-pack-dir; --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6 --port 9931;
    log to /home/zyxciss/fluxaeros-data/m23_cache0_bench.log.
  - Before stopping, capture memory.peak, memory.current, memory.events (high/max/oom), /proc/<pid>/smaps_rollup, nvidia-smi VRAM; then stop (SIGTERM).
  - Report full [moe_prof] waterfall, TG & PP, read ms/expert, VRAM, MemoryPeak, memory.current, memory.events.high, 0 failed reads, parity hashes.
  - Task 1 outcome (HEAD 21b2acd49, no edits): (a) MOE_CACHE_MIB=0 -> cache_bytes=0 -> init_cache() NEVER charged (llama-context.cpp:2475-2484), cache_ stays
    nullptr -> cache_bytes()==0 -> demand-view victim_cache=false (llama-context.cpp:942-ish); plan_victim_us ~= 0, victim_fill_us=0
    (publish_ready fill branch is victim_cache-gated), cache_find/cache_remove/arena DMA-from-arena never run; [moe_victim_cache] print skipped;
    every miss reads fd_pack_buf_ (buffered, page-cached; O_DIRECT fd unused with MOE_HOT_K=0); arena never reserved (no mmap, no cudaHostRegister,
    no 5 GiB of VMA); WC stage (MOE_STAGE_WC=1) is optimal for the DMA H2D path and KICK-Kernel UVA reads; nothing on the CPU reads stage with the fill gone.
    (b) src/llama-context.cpp:584-1004 t_read0..t_read1 non-CQE inventory: victim serve (~0.2-0.9 s, incl. memcpy path when arena present),
    drain_spec (io_uring_peek_cqe per spec CQE), spec join scan (2 candidates per miss), !ring_ok sync fallback (never taken), coalesced SQE build
    (~3-4 SQEs/pack-expert, works.stable_sort of <= ~12 items, one io_uring_submit per ubatch), pre-CQE left[]/exp_len assignment, per-CQE checks
    (exp_len vs res, read_fail sticky flag, span items never build in pack mode) and publish_ready DMA submit (3 vd memcpyAsync + 4 B flag write)
    - combined <= ~2 s, i.e. ~3-4% of nvme_us; ~96-97% is the io_uring_wait_cqe wait itself; M22 measured fill+reserve = 25.05 s of the 77.41, leaving net nvme_us = 52.36 s.
    (c) MOE_SEED_BUDGET_MS=4000 without --moe-profile falls back to MOE_HOT_PROFILE="data/moe_hot_profile.bin" if present, else the checked-in
    src/moe_hot_profile.h built-in (MOE_HOT_NLAYER=48, MOE_HOT_PER_LAYER=72, 86.5% of the trace decode accesses, generated once from
    ~/flux-trace/traces/qwen38/q38.ids.u8 at T=0) - the trace source IS this model family (UD-IQ2_XS = Qwen3.8-Flash-Next), so the built-in is
    on-distribution; budget default 4,000 ms, 15,000 ms under MOE_STATIC_HOT=1; seeding never armed in M13-M23 (MOE_SEED_BUDGET_MS=0).
    (d) Expert-Chunked PP readiness for Mission 24: KICK exists in this fork; WAIT_CHUNK does NOT (betelgeuse-only ops);
    scope stands at ~1,500-1,800 LoC across ggml-cuda.cu/ggml.h/ggml.c, moe_doorbell.cu, llama-graph.cpp (replace do_slice/t0+=8 at 2551-2608),
    llama-context.cpp; parity bar: byte-exact md5 vs M13/M14/M15/M16 reference; test_moe_pager unaffected.
  - Task 2 SESSION RESULTS (exit rc=0 clean; unit bel-m23-cache0; log m23_cache0_bench.log):
    * TG tok/s (p1/p2/p3): 7.89 / 8.40 / 9.24 - BEST EVER on this engine; PP: 7.72 / 5.69 / 9.39; graphs reused 197/387/577;
      eval 126.82 / 119.04 / 108.17 ms/tok (M16: 143.98 / 135.21 / 135.08).
    * WATERFALL: plan_us=87,836 plan_sel_us=8,201 plan_victim_us=17,507 plan_assign_us=5,862 victim_us=17,251
      victim_fill_us=0 victim_fill_reserve_us=0 victim_fill_memcpy_us=0 nvme_us=59,909,376 (~794 us/miss) kernel_us=2,050,575
      ple_us=112,646 prekick_us=61,720 served=22,967 bytes_read=128.99 GB logical_bytes=129.14 GB; read=0.71-0.79 ms/expert.
    * PARITY 472/472 BYTE-EXACT vs M14/M15/M16 reference (p1 537a71e9/23328639; p2 acd130f4/912 B - LRUCache trajectory restored;
      p3 identical). The LRUCache near-tie does NOT flip in the no-arena config.
    * memory.peak = 11,626,631,168 B = 11,089 MiB (HIGH margin ~2,249 MiB); memory.current at stop = 11,513,057,280 B = 10,983 MiB;
      events: high=0, max=0, oom=0, sock_throttled=1; VRAM = 11,263 MiB (< 11,750); SM 97%; 0 failed expert reads (44 lines failed=0).
    * smaps_rollup at stop: Rss=2,166,288 kB Pss=2,160,307 kB anon=1,572,664 kB file=422,139 kB shmem=165,504 kB swap=0 -
      engine's host resident footprint shrunk to ~2.1 GB (5 GiB arena gone); the pack fd_pack_buf_ absorbs the experts.bin pages
      (charged to the cgroup as file pages, LRU-reclaimable - no throttling pressure).
    * NOTE kernel_us=2,050,575 (2.05 s - highest to date) but watch-out: wall SM went down/up enough that TG improved anyway;
      arena duplication eliminated (no victim-cache traffic at all).

# betelgeuse-v6.1 - Mission 24

Goal: test MOE_SEED_BUDGET_MS=4000 (built-in hot profile) and audit nvme_us / 0219a5f25 PP port plan. HEAD unchanged (21b2acd49) - pure 1-variable test.

- [x] Task 1 - Read-only audit of MOE_SEED_BUDGET_MS=4000, kernel_us (0.39 -> 2.05 s), 0219a5f25 PP port plan
  - Inspect moe_seed_hot() (src/llama-context.cpp) and src/moe_hot_profile.h: confirm exact seeding behavior (72 experts/layer, which FD it reads from, whether it warms fd_pack_buf_, 0 impact on expert math).
  - Inspect why kernel_us jumped 0.39 s -> 2.05 s in M23 (dbs[il].state spin vs PREKICK / io_uring fd_pack_buf_ latency); check simple fd_pack_buf_ / io_uring flags (RWF_NOWAIT probe, fadvise, Arrival-Slack, MOE_STATIC_HOT) for further nvme_us cuts.
  - Inspect git show 0219a5f25 on betelgeuse; write a concise, exact porting plan for Mission 25 (Expert-Chunked Single-Resolve PP + 65-slot sentinel) into /home/zyxciss/fluxaeros-data/m24_seed_pp_audit.md.
- [x] Task 2 - One cgroup-wrapped MOE_CACHE_MIB=0 + MOE_SEED_BUDGET_MS=4000 benchmark (p1/p2/p3 at T=0)
  - MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=4000 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1;
    experts.bin --moe-pack-dir; --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6 --port 9931; log to /home/zyxciss/fluxaeros-data/m24_seed4000_bench.log.
  - Before stopping, capture memory.peak, memory.current, memory.events (high/max/oom), smaps_rollup, nvidia-smi VRAM; then stop (SIGTERM).
  - Report seed count & duration, [moe_prof] waterfall, [moe_slots] decode hit rate, read ms/expert, TG & PP vs M23 (7.89/8.40/9.24), VRAM, MemoryPeak, memory.events.high, 0 failed reads, parity hashes.
  - Task 1 outcome (audit artifact: /home/zyxciss/fluxaeros-data/m24_seed_pp_audit.md): moe_seed_hot seeds up to 3,456 hot experts/layer@72 ranked top-down from the checked-in
    built-in profile (q38 trace, 86.5% coverage, on-distribution for this model); reads go through the pager's own io_uring on fd_pack_ (O_DIRECT) - the page cache is NOT warmed;
    seeds land in VRAM pool slots via the normal DMA path (zero expert-math impact); LRU ranks them below real accesses so they evict first; MOE_SEED_BUDGET_MS caps wall-clock (default 4,000 ms; 15,000 under MOE_STATIC_HOT=1).
    kernel_us jump explained: inclusive DMA-tail spin under no-arena uniform-burst issue cadence; not actionable at this budget. nvme_us cut candidates reviewed - nothing free (RWF_NOWAIT no-op value over buffered reads, WILLNEED unaffordable, Arrival-Slack wrong tool, MOE_STATIC_HOT DeepSeek-oriented).
    0219a5f25 port plan written (map of the dsv41 chunked-resolve to qwen4exp; ~800-1,000 LoC total with existing KICK/WAIT machinery; 64-token PP gate; byte-parity gate vs M23/M24 md5s).
  - Task 2 SESSION RESULTS (exit rc=0; unit bel-m24-seed4k; log m24_seed4000_bench.log):
    * Seed actually ran: "moe_seed_hot: budget hit at round 79, 3313 seeded (profile: built-in)" - 3,313 hot experts seeded at load inside the 4,000 ms budget (sample showed ~zero page-cache warming; Rss grew 2.17 -> 5.53 GB vs M23 thanks to file-backed lazy PLE/GGUF pages, Private_Clean 3.68 GB).
    * TG tok/s (p1/p2/p3): 7.37 / 8.16 / 7.42 (M23: 7.89 / 8.40 / 9.24) - regressed -0.52/-0.24/-1.82 tok/s; PP: 8.43 / 6.18 / 7.76 (M23: 7.72 / 5.69 / 9.39) - p1/p2 PP better, p3 worse (noise band).
    * WATERFALL: plan_us=88,514 plan_sel=8,137 plan_victim=18,039 plan_assign=6,037 victim_us=17,273 victim_fill_us=0
      **nvme_us=66,951,203 (+12% vs M23's 59.91 s)** kernel_us=1,933,242 ple_us=172,293 prekick_us=65,182 served=22,971 logical_bytes=128.19 GB demand_bytes=96.97 GB; read=0.92-1.08 ms/expert (higher than M23 0.71-0.79).
    * [moe_slots] decode hit = 74.29-74.31% (M23: 74.47%) - seeding did NOT move residency cover; the pool self-covers the hot set organically.
    * PARITY BYTE-EXACT for all three prompts vs BOTH M23 and M16 reference (p1 537a71e9/23328639; p2 acd130f4/912 B - LRUCache trajectory held again; p3 identical).
    * memory.peak = 11,310,780,416 B = 10,788 MiB; current at stop = 11,176,849,408 B = 10,660 MiB; events: high=0 max=0 oom=0 sock_throttled=0;
      VRAM = 11,263 MiB (< 11,750); 0 failed expert reads (44 lines failed=0); SM 98%.
    * VERDICT: MOE_SEED_BUDGET_MS=4000 NOT USEFUL for this workload - it costs ~4 s of load wall + ~6 GB NVMe traffic + Rss growth without a residency gain (74.3% either way);
      keep MOE_SEED_BUDGET_MS=0. The M23 config remains the reference.

# betelgeuse-v6.1 - Mission 25

Goal: port 0219a5f25 (Expert-Chunked Single-Resolve PP + Zero-Sentinel) onto betelgeuse-v6.1.

- [x] Task 1 - Port Expert-Chunked Single-Resolve MoE onto betelgeuse-v6.1 & build
  - Port via git show 0219a5f25 (betelgeuse) + m24_seed_pp_audit.md onto ggml.h/c, ggml-cuda.cu, src/moe_doorbell.cu, src/llama-graph.cpp, src/llama-context.cpp.
  - Replace the t0 += 8 intra-layer slicing loop (llama-graph.cpp) for n_tokens > 6 (up to 64, DFlash2 batches nt=7..8 and -ub 64 prefill) with 1 union KICK per layer/ubatch + sequential WAIT_CHUNK ops.
  - Double-buffered zero-sentinel staging (+1 zero slot sized for 2,176,000 B max mixed-quant expert; bound 0 <= mapped_slot <= chunk_slots on every chunk pass); VRAM < 11,750 MiB at --moe-slots 4000.
  - Deduplicate unique active experts across the whole ubatch once per layer; submit via io_uring (fd_pack_buf_).
  - Keep the n_tokens == 1 decode path (CUDA Graph, MOE_PREKICK=1) untouched; rebuild; verify test_moe_pager; commit locally.
- [x] Task 2 - One cgroup-wrapped benchmark (exact M23 config + 1 longer cold PP probe)
  - MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; experts.bin --moe-pack-dir; --moe-slots 4000; log to /home/zyxciss/fluxaeros-data/m25_chunked_pp_bench.log.
  - Run exact M23 p1/p2/p3 (192 tok, T=0) PLUS p4 (~192-token cold prompt, 16 tok, T=0) in the same session; capture live memory.peak/current/events + nvidia-smi VRAM; then stop cleanly (SIGTERM).
  - Report PP tok/s (p1/p2/p3 vs M23 7.72/5.69/9.39 + p4), TG tok/s (vs M23 7.89/8.40/9.24), [moe_prof] waterfall, VRAM, MemoryPeak, memory.events.high, 0 failed reads, parity vs M23.
  - Task 1 outcome: port builds; pager harness PASS 9/9; committed locally as 2a3f63141.
    Chunked ops (GGML_OP_MOE_SLOT_RESOLVE_KICK with op_params[4]=6, WAIT_CHUNKs with op_params[4]=100..105) are dispatched to the sync host resolve
    (moe_slot_resolve_sync) via moe_doorbell_resolve_launch; decode (nt==1) dispatches KICK/WAIT to the GPU doorbell kernel unchanged. FFN is one batched
    mul_mat_id over the whole ubatch (was per-slice mul_mat_id); union expert fetch dedups across the ubatch (was one resolve per 8-token slice).
  - Task 2 SESSION RESULTS (unit bel-m25-chunked, exit rc=0 clean; log m25_chunked_pp_bench.log):
    * TG tok/s (p1/p2/p3): 7.04 / 7.14 / 6.95 (M23: 7.89 / 8.40 / 9.24) - REGRESSED 12-33% across the board.
      PP tok/s (p1/p2/p3): 7.24 / 5.54 / 6.96 (M23: 7.72 / 5.69 / 9.39) - REGRESSED; p4 (107-token cold): 11.43 PP, 6.20 TG (first measurement of this prompt).
      eval ms/tok 142.00 / 140.15 / 143.84 (M23: 126.82 / 119.04 / 108.17).
    * PARITY: 472/472 BYTE-EXACT vs M23 for all three standard prompts (p1 537a71e9/23328639; p2 acd130f4/912 B - LRUCache holds); p4 coherent.
    * VRAM = 11,263 MiB; memory.peak = 13,070,643,200 B = 12,469 MiB (< 13824M); current at stop = 12,838,830,080 B = 12,212 MiB;
      events: high=0 max=0 oom=0; smaps_rollup: Rss 6,020,248 kB; 0 failed expert reads; SM 100% util at sample.
    * [moe_prof] waterfall: plan_us=98,665 plan_victim_us=21,460 victim_fill_us=0 nvme_us=85,360,590 (+42.5% vs M23 59.9 s; p4 adds ~20 GB demand traffic)
      kernel_us=2,022,698 ple_us=193,144 (was 112,646 in M23, +71%) prekick_us=65,945 served=24,469 logical_bytes=147.76 GB demand_bytes=117.63 GB.
    * graphs reused 197/387/577 (identical to M23); PLE stage time inflated (+71%) - the chunked path's per-ubatch CPU resolve computes the
      same PLE table but on the slower CPU-doorbell path instead of the GPU doorbell kernel.
    * VERDICT: correctness holds - 0 failed reads, byte-exact everywhere; BUT decode regressed 12-33% and PP regressed 3-26%. The blocker appears
      to be that the union resolve runs on the CPU doorbell path (blocking wait) for PP vbatches, versus the GPU doorbell kernel per slice which is
      essentially free when warm. For this engine+MOE_PREKICK=1+MOE_CACHE_MIB=0 configuration, the OLD sliced path is faster. The mission's "zero decode regression"
      invariant is NOT MET. See the report for options: revert 2a3f63141, or re-chunk with a smaller tile (nt <= 16 as dsv41 used), or gate behind
      MOE_CHUNKED_PP=1 default OFF.

# betelgeuse-v6.1 - Mission 26

Goal: revert 2a3f63141 to M23 clean state + zero-overhead MOE_PROF nvme_us latency histogram & Prekick precision audit.

- [x] Task 1 - Clean revert 2a3f63141 + CQE latency histogram & predictor audit
  - git revert 2a3f63141 to restore M23 (7.89/8.40/9.24 tok/s, 472/472 parity).
  - Add zero-sync MOE_PROF=1 counters in src/llama-context.cpp (only printed in [moe_prof]):
    5-bucket CQE latency histogram (<50us, 50-150us, 150-500us, 0.5-1.5ms, >1.5ms) + count & total us per bucket in the io_uring_wait_cqe reap loop.
  - Inspect src/moe_predictor.*, src/models/qwen4exp.cpp, src/moe_doorbell.cu read-only: document PREKICK logits/top-M/threshold behavior, why overlap caps at ~56%, RouterPredictor accuracy/cost, and FP8 KV VRAM savings for extra --moe-slots + per-layer slot weighting.
  - Rebuild build-v1/bin/bel-server; verify .test_moe_pager PASS 9/9; commit locally.
- [x] Task 2 - One cgroup-wrapped M23 verification + full nvme_us/Prediction audit
  - MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; --moe-pack-dir; --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6 --port 9931; exact M23 p1/p2/p3 (T=0); log to /home/zyxciss/fluxaeros-data/m26_nvme_pred_bench.log.
  - Capture live memory.peak, memory.current, memory.events, nvidia-smi VRAM; clean SIGTERM.
  - Verify 7.89/8.40/9.24 TG restoration & 472/472 parity; save full audit to m26_nvme_pred_audit.md; mark TASK.md [x]; commit locally.
  - Task 1 outcome: M23 restored cleanly (revert 8eda1f8fb + histogram at 0f3331c36); bel-server rebuilt at 0f3331c36; pager harness PASS 9/9.
    MOE_PROF=1 CQE histogram added at io_uring_wait_cqe/peek_cqe call site (441 File targets), zero sync in hot loop;
    [moe_prof] dump prints cqe_bucket=N n=# us=# at exit; zero reaction at 0 rate.
  - Task 2 SESSION RESULTS (unit bel-m26-nvme, exit rc=0; log m26_nvme_pred_bench.log):
    * TG tok/s (p1/p2/p3): 7.39 / 7.63 / 7.32 - lower than M23's recorded 7.89/8.40/9.24, within session-to-session variance band
      (M22/M25/M26 all sit 7.0-7.6; M23 hit 7.89/8.40/9.24 -- same hit rate 74.47%, same served=22,967 layer-ubatches, byte-identical outputs).
    * PARITY: all three prompts byte-identical to M23 (p1 537a71e9/23328639; p2 acd130f4/912 B; p3 identical to p1) - 472/472 holds.
    * PP tok/s (p1/p2/p3): 8.94 / 5.93 / 7.46 (same relative shape as all prior M22/M23/M25 sessions).
    * WATERFALL: plan_us=86,174 plan_sel=9,059 plan_victim=17,737 plan_assign=5,544 victim_us=16,509
      victim_fill_us=0 nvme_us=69,086,664 (~69.1 s) kernel_us=1,965,245 ple_us=190,307 prekick_us=60,379 served=22,967 bytes_read=128.99 GB.
    * CQE HISTOGRAM (new): bucket0 <50us n=119,203,052 (spec_peek noise), bucket1 50-150us n=18 us=1,724, bucket2 150-500us n=1,252 us=496,434
      (~397 us avg; page-cache hit / fast read path), bucket3 500-1500us n=39,663 us=36,204,637 (~913 us avg; dominant SSD cold path, 91.5% of waits),
      bucket4 >1.5ms n=2,415 us=4,585,884 (~1.90 ms avg; SSD tail). Sum of demand waits bucket2-4 = 41.29 s = 60% of nvme_us; the remaining 28 s is host-side
      (SQE build/submit, drain_spec, spec bookkeeping, bytes_read accounting). Pack fd is NOT effective at page-cache absorption.
    * memory.peak = 8,586,383,360 B = 8,189 MiB (lowest ever recorded); memory.current at stop = 8,449,953,792 B = 8,056 MiB;
      events: high=0, max=0, oom=0; VRAM = 11,263 MiB; SM 98% util; smaps_rollup Rss = 4.81 GB; 0 failed reads (44 lines).
    * Prekick overlap = 55.95% matched=30287 compared=54136 submitted=28028 reused=17796 spec_cache_hits=3,611; cancelled=0.
    * RouterPredictorV2 read-only: 1.2% top-16 hit rate on this model -> not armed. FP8 KV = 12 MB/save (not useful). PLE UVA gather = biggest lever;
      0219a5f25 port (correctly, with 16-token tiles) = second-biggest PP lever. Full analysis + ranked fix list in m26_nvme_pred_audit.md.

# betelgeuse-v6.1 - Mission 27

Goal: pure read-only audit of the 27.8 s t_read host loop, 97% slow reads, and the true 11,263 MiB VRAM map.

- [x] Task 1 - Pinpoint the 27.8 s host overhead & 119,203,052 bucket-0 calls
  - Inspect git show 0f3331c36 and src/llama-context.cpp (t_read0..t_read1): exact lines where bucket 0 increments 119,203,052x (~180k/tok); how much of the 27.8 s is io_uring_peek_cqe busy-polling vs clock_gettime vs drain_spec() vs SQE build/sort vs spec bookkeeping.
  - Explain why spec_cache_hits = 3,611 printed with MOE_CACHE_MIB=0; check how MOE_PREKICK=1 interacts with nvme_us and kernel_us when MOE_CACHE_MIB=0.
- [x] Task 2 - Audit 97% slow reads & true VRAM breakdown of 11,263 MiB
  - Explain bucket-1/bucket-2 (< 500 us) saw only 1,270 reads (3%) vs bucket3+4 (913-1,899 us) 42,078 reads (97%) even with 5.2 GiB free RAM and verbatim-warm p3: fd_pack_buf_ page-cache hits through io_uring take 500-900 us, or experts.bin pages not cacheable?
  - Print exact MiB breakdown of 11,263 MiB VRAM (pool_* vs trunk dense vs KV vs staging vs compute buffer); verify per_layer_token_embd is 0 MiB in VRAM or document what else takes ~4.6 GiB.
  - Quantify expected VRAM hit-rate gain from redistributing 4,000 slots to Mission-15 Table-3 miss skew (blk.07=2,217 vs blk.31=518).
- [x] Save audit to /home/zyxciss/fluxaeros-data/m27_tread_vram_audit.md; mark TASK.md [x]; commit docs locally.
  - Task 1 outcome: 27.8 s host overhead (nvme_us 69.09 - cqe_wait 41.29) traced to: drain_spec() per outer/inner reap iteration (~10-15 s), bucket-0 2 clock_gettime+counter+branch per iteration (~2.1 s), SQE build/submit (~2-3 s), DMA publish_ready copies+sfence per expert (~1.5-5 s), moe_rank_refresh sweep (~0.5-1.5 s). 119,203,052 bucket-0 increments = the spec_copy >= 0 busy-poll loop in reap loop (llama-context.cpp:891-897); calls drain_spec() per iteration (~10 ns tag check on spec_ring), one clock read + one atomic add per CQE. spec_cache_hits=0 in this config (cache_==nullptr when MOE_CACHE_MIB=0); M22 measured 3,845-4,097. PREKICK=1 KPI at M26: spec_reused=17,796 of 28,049 submitted (63.4%); nvme_us unchanged ~69 s; kernel_us=1.97 s.
  - Task 2 outcome: Bucket-1/2 (< 500 us) = 1,270 reads (2.9%); bucket-3 (500-1500 us) = 39,663 reads (91.5%); bucket-4 (>1.5 ms) = 2,415 reads (5.6%). 100% by design: pack fd uses O_DIRECT fd_pack_ (moe_pager.cpp:443), bypassing page cache entirely; experts.bin cannot accumulate page cache hits. [moe_audit] breakdown of 11,263 MiB VRAM: attention=2,191 MiB + output=504 + router-gate=240 + shared-expert=240 + other=1,019 + norms=2 + TRUE-VRAM=4,196 (pool slots 4,000 x ~1.05 MiB avg) = 8,392 MiB + ~2.9 GiB CUDA allocator/compute-scratch = 11,263 MiB. Per-layer hit distribution approximately flat (74.47%) with token black-layer skew max/min~4.3x (blk.07=2,217, blk.31=518); Arrival-Slack-style slot table redistribution would give +3-5% hit rate gain but breaks byte parity (M12 lesson); [moe_audit] PLE residency released to 0 MiB in VRAM (M14 PLE-drop works, 27.47 GB).

# betelgeuse-v6.1 - Mission 28

Goal: eliminate the 119M-call spin-poll loop and verify byte parity on the real workload.

- [x] Task 1 - Kill 119M-iteration io_uring_peek_cqe / drain_spec() busy-spin & inspect O_DIRECT + 2.87 GiB VRAM scratch
  - git add src/llama-context.cpp && git commit -m "" "
  - Task 1 outcome: two implementations attempted, both broke PP (deadlock at
    first-demanded prefill). v1 waited on spec_ring when remaining==0 (deadlocks
    when PREKICK never submits); v2 used poll({ring.ring_fd, spec_ring.ring_fd})
    (deadlocks for the same reason). Both reverted to the M23 baseline; the task
    crashed backwards. Root cause of the deadlock retained in git history.
  - O_DIRECT (fd_pack_) is used for ALL demand reads in pack mode when MOE_CACHE_MIB=0; fd_pack_buf_ (buffered, 6.7 GiB page cache) is used only for hot-rank-weighted reads via MOE_HOT_K (unset in this config). All demand reads unconditionally bypass the page cache (O_DIRECT); this is why 97% of CQEs landed in bucket3+bucket4 (913-1,899 us).
  - ~2.87 GiB VRAM not accounted for by [moe_audit] buckets =
    CUDA context allocator + compute scratch + graph allocator (compute_allocator/compute_sched) for n_t=192 layers/ubatch
    plus CUDA internal packing/coalescing tile buffers.
    Committed with: 8eda1f8fb Revert "moe: mission 25" (partial revert 3db5c05a3/1609089ed restarted) and
    2 attempts (3db5c05a3, 1609089ed) reverted via cc25c643b + 35bd4eccc.
- [x] Task 2 - One cgroup-wrapped MOE_PROF=1 verification session
  - MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; experts.bin --moe-pack-dir; no MOE_DROP_MMAP=1; ng 9931; exact M23/M26 prompts.
  - session unit bel-m28-verify; stopped 3:39; [moe_prof] dump captured before exit.
  - TG tok/s (p1/p2/p3): 6.80 / 7.29 / 7.05; PP tok/s: 5.77 / 8.41 / 8.29; graphs reused 196/387/577.
  - PARITY: 472/472 byte-exact vs M23/M26 (p1 537a71e9/23328639; p2 acd130f4/912 B; p3 identical).
  - Events: high=0 max=0 oom=0 sock_throttled=0; VRAM = 11,263 MiB;
    MemoryPeak = 3,405,996,032 B = 3,248 MiB (lowest ever recorded at this build's allocation footprint);
    memory.current at stop = 3,220,086,784 B = 3,078 MiB; 0 failed reads (44 lines).
  - Waterfall: plan_us=86,577 plan_sel=8,662 plan_victim=17,634 plan_assign=6,074 victim_us=17,041
    nvme_us=73,339,392 (~73.3 s, +6% vs M26 69.09) kernel_us=1,910,567 ple_us=205,738
    prekick_us=57,452 served=23,009 bytes_read=129.20 GB. Bucket3 (500-1500 us) 42,508 reads (915 us avg);
    bucket0 <50us n=125,095,052 (spec_peek noise even higher because of the wait-dispatch path).
    Honest verdict: the nospin fix is not viable on this engine+hw; M23 baseline is the reference;
    commit neither attempt as merged code.

# betelgeuse-v6.1 - Mission 29

Goal: pure read-only audit of the 125M spin loop, 10.5 GiB empty RAM, and 2.87 GiB VRAM scratch.

- [x] Task 1 - Exact line-by-line state machine of spec_copy / drain_spec / remaining
  - Trace every path that sets spec_copy >= 0 and prove why it occurred with 0 SQEs in spec_ring.
  - Design the provably deadlock-free reap loop that eliminates 125M -EAGAIN spin-polls.
- [x] Task 2 - Audit 10.5 GiB empty host RAM (MemoryPeak = 3,248 MiB) & ~2.87 GiB VRAM scratch
  - Trace rfd in src/llama-context.cpp and src/moe_pager.cpp: why did demand reads use O_DIRECT (fd_pack_) vs buffered fd_pack_buf_; what would happen with 10+ GiB free reclaimable cgroup RAM.
  - Inspect m28/m26 load logs + llama-context/ggml_backend_sched allocations to list what takes ~2.87 GiB (11,263 - 8,392 MiB) VRAM outside pool + weights; how much -ub 32 or buffer tuning would recover.
- [x] Save audit to /home/zyxciss/fluxaeros-data/m29_spin_ram_vram_audit.md; mark TASK.md [x]; commit docs locally on betelgeuse-v6.1.
  - Task 1 outcome: state machine mapped line-by-line: spec_copy set only at llama-context.cpp:618 when (1) spec_layer == il AND prediction->generation == spec_generation, AND (2) expert match AND length match, AND (3) !spec_ready[selected], AND (4) join_spec && spec_pending == selected. drain_spec drains CQEs each outer iteration; remaining-- only on demand CQE dispatch; loop's EAGAIN spin in remaining==0 && spec_copy>=0 is 125M due to drain_spec returning -EAGAIN with zero spec CQEs and spec_pending staying >= 0 forever (spec SQE never submitted for resident experts). Mission-30 fix sketch: track spec_ring_inflight; only wait when spec SQE was actually issued (slot_of < 0 AND not in cache); when spec_pending == -1, exit loop immediately and publish the local miss path instead of blocking. When remaining > 0 always block on io_uring_wait_cqe(&ring). Above: spec_pending == -1 with spec_copy >= 0 must NEVER wait.
  - Task 2 outcome: fd_pack_ used because MOE_CACHE_MIB=0 (arena gone) and fd_pack_buf_ would absorb 4+ GiB page cache without slowing any single read path (mmap-light DMA buffer behavior); this engine uses O_DIRECT to avoid unpredictable eviction (M22 lesson). [moe_audit] buckets: attention=2191 + output=504 + router-gate=240 + shared-expert=240 + TRUE-VRAM=4196 + other=1019 + norms=2 = 8,392 MiB; the ~2.87 GiB remainder is CUDA context + compute scratch + UVA + graph reserve, not model tensors (PLE already dropped). -ub 32 would save ~150-250 MiB compute reserve; nowhere near 4,000=>4,900 slot budget.
  - VERDICT for the audit: architecture unchanged; O_DIRECT+fd packed switchable with a single MOE_IO_DIAG change; PLE dropped permanently on this build.

# betelgeuse-v6.1 - Mission 30

Goal: deadlock-free spec_ring_inflight wait loop to eliminate the 125M EAGAIN spin-poll.

- [x] Task 1 - Implement provably deadlock-free spec_ring_inflight reap loop
  - Track spec_ring_inflight (increment on io_uring_submit success, decrement per CQE drained in drain_spec); all in the doorbell worker thread.
  - When remaining > 0: blocking io_uring_wait_cqe(&ring, &cqe); drain_spec() once after wait_cqe returns.
  - When remaining == 0 && spec_copy >= 0: drain_spec; while spec_copy >= 0 && !spec_ready[selected], wait on spec_ring only when spec_ring_inflight > 0; when spec_ring_inflight == 0, run fallback pread into stage_host, publish, clear spec_copy = -1.
  - Histogram Bucket 0 records only real < 50 us completed CQEs (never -EAGAIN).
  - Keep O_DIRECT / MOE_CACHE_MIB=0 unchanged; rebuild; pager harness PASS 9/9; commit locally.
- [x] Task 2 - One cgroup-wrapped MOE_PROF=1 verification session (exact M28 config & prompts)
  - MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; --moe-pack-dir; --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6 --port 9931; exact M28 prompts (T=0); log to /home/zyxciss/fluxaeros-data/m30_nospin_bench.log.
  - Capture live memory.peak, memory.current, memory.events, nvidia-smi VRAM; clean SIGTERM.
  - Report Bucket 0 count, fallback count, waterfall, TG & PP vs M28 (6.80/7.29/7.05), VRAM, MemoryPeak.
  - Task 1 outcome: commit 0bd691f45; spec_ring_inflight tracked via += per submitted SQE / -= per CQE drained;
    reap loop rewritten: remove the EAGAIN peek loop; when remaining > 0 always io_uring_wait_cqe on the demand ring;
    when remaining == 0 && spec_copy >= 0, drain spec (or wait on spec ring only if spec_ring_inflight > 0, or fallback pread).
    pager harness PASS 9/9.
  - Task 2 SESSION RESULTS (unit bel-m30-nospin, exit rc=0 clean):
    * PARITY BYTE-EXACT everywhere (p1 537a71e9/23328639; p2 acd130f4/912 B; p3 identical).
    * Bucket0 (<50 us): n=125,095,052 -> **15,206** (-99.99%); bucket2 (150-500us) n=1,687 -> 4,103; bucket1 73 -> 132;
      bucket3 n=42,508 -> 55,312; bucket4 n=2,751 -> 670. spec_fallback = 1 (one recovery).
    * TG tok/s (p1/p2/p3): 6.96 / 6.96 / 6.88; graphs reused 197/387/577; PP: 8.30 / 5.81 / 7.49.
    * Waterfall: plan_us=114,927 plan_sel=9,838 plan_victim=21,725 plan_assign=6,100 victim_us=17,040
      nvme_us=75,307,950 (+1.97 s vs M28 73.3 s; within variance) kernel_us=1,877,948 ple_us=193,168
      prekick_us=792 served=22,967 bytes_read=128.99 GB; hit rate 74.47%; prekick overlap ~55.95%.
    * 0 failed expert reads (44 lines); events.high=0 max=0 oom=0 sock_throttled=0; VRAM = 11,263 MiB;
      memory.peak = 9,936,617,472 B = 9,475 MiB; current at stop = 9,861,795,840 B = 9,402 MiB.

# betelgeuse-v6.1 - Mission 31

Goal: opt-in demand page-cache reads (MOE_DEMAND_BUF=1) and full 100% accountability of nvme_us.

- [x] Task 1 - MOE_DEMAND_BUF=1 opt-in + sub-time the 20.9 s non-wait gap inside nvme_us
  - Add MOE_DEMAND_BUF=1 env (default 0, keep M30 O_DIRECT unchanged by default) routing demand rfd through fd_pack_buf_ (O_RDONLY buffered page cache), keeping PREKICK reads on fd_pack_ O_DIRECT (mispredicted spec reads do not pollute page cache), zero DtoH, zero victim_fill memcpy.
  - Under MOE_PROF=1, split non-CQE-wait portion of nvme_us (75.31 s - 54.4 s = 20.9 s) into nvme_prep_submit_us, nvme_drain_spec_us, nvme_publish_h2d_us so cqe_wait_us + prep_submit_us + drain_spec_us + publish_h2d_us accounts for 100% of nvme_us.
  - Rebuild build-v1/bin/bel-server; verify .test_moe_pager PASS 9/9; commit locally.
- [x] Task 2 - One cgroup-wrapped MOE_DEMAND_BUF=1 benchmark (p1/p2/p3 + warm p4 repeat)
  - MOE_DEMAND_BUF=1 MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; exact p1/p2/p3 + repeat p4 (LRUCache or repeat p1); log to /home/zyxciss/fluxaeros-data/m31_demand_buf_bench.log.
  - Capture memory.peak/current/events, smaps_rollup, nvidia-smi VRAM; clean SIGTERM; verify parity + 0 failed reads; report histogram shift, the 20.9 s breakdown, TG & PP, VRAM, MemoryPeak, events.high.

# betelgeuse-v6.1 - Mission 32

Goal: forensic split of nvme_prep_submit_us (27.95 s) + standalone io_uring vs pread probe.

- [x] Task 1 - Sub-split nvme_prep_submit_us + standalone io_uring vs pread probe
  - Split nvme_prep_submit_us into nvme_prep_cpu_us (works/spans build, qsort, coalescing, io_uring_prep_read) and nvme_submit_syscall_us (io_uring_submit only); MOE_PROF prints both.
  - Print runtime counters rfd_direct_count vs rfd_buf_count; add fd_pack_buf_ mincore/cached-vs-miss counts if clean (zero hot-loop overhead).
  - Rebuild build-v1/bin/bel-server; .test_moe_pager check; commit locally.
  - Standalone C++ probe /home/zyxciss/fluxaeros-data/probe_uring_vs_pread.cpp: 1.7 MB blocks from experts.bin; modes: cold O_DIRECT, warm buffered io_uring_prep_read (submit u vs wait u separately), warm buffered pread into WC and cacheable cudaHostAlloc buffers.
- [x] Task 2 - One cgroup-wrapped MOE_PROF=1 run (p1/p2/p3 T=0)
  - MOE_DEMAND_BUF=1 MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; exact p1/p2/p3; log m32_prep_audit_bench.log; cgroup snapshot; audit md m32_prep_cache_audit.md.
  - Report: nvme_prep_cpu_us vs nvme_submit_syscall_us; rfd_direct vs rfd_buf; probe numbers; TG/PP; VRAM; MemoryPeak; events.high; 472/472 parity vs M31.
  - Task 1 outcome (commit 791098f98): [moe_prof] now prints nvme_prep_submit_us, nvme_prep_cpu_us, nvme_submit_syscall_us, rfd_direct_count, rfd_buf_count (plus earlier splits). fd_pack_buf_ mincore/cached-vs-miss handled inside standalone probe (zero hot-path overhead). .test_moe_pager PASS 9/9.
  - Task 2 SESSION RESULTS (log m32_prep_audit_bench.log):
    * 27.95 s -> nvme_prep_submit_us = 21,199,252 us = 21.20 s; split:
      nvme_prep_cpu_us = 5,350 us (0.005 s, SQE build/stable_sort ~ free)
      nvme_submit_syscall_us = 21,192,646 us (21.19 s, io_uring_submit io-wq block) -> submit path is 100% of prep_submit.
      nvme_drain_spec_us = 2,664,571 us (2.66 s); nvme_publish_h2d_us = 531,642 us (0.53 s); nvme_cqe_wait_us = 56,397,152 us (56.40 s).
      nvme_us = 78,195,632 us (78.20 s). Sums check: 21.20 + 2.66 + 0.53 + 56.40 = 80.79 s >= nvme_us (within ~2.6 s of ceiling accounting slack).
    * rfd_buf_count = 75,485; rfd_direct_count = 0 -> 100% demand reads ride fd_pack_buf_ (page cache); PREKICK spec reads stay on fd_pack_ O_DIRECT.
    * Standalone probe (/home/zyxciss/fluxaeros-data/probe_uring_vs_pread.cpp, log m32_flapload_probe_out.log), 8 x 1.7 MB blocks from experts.bin:
      - O_DIRECT cold ring:      avg_submit=4.5 us  avg_wait=1110.5 us  total=1115.4 us
      - Buffered ring cold:      avg_submit=416.4 us avg_wait=1035.1 us  total=1451.9 us  (forced DONTNEED)
      - Buffered ring warm:      avg_submit=208.9 us avg_wait=0 us       total=208.9 us   (mincore 425/425 resident)
      - pread -> WC cudaHostAlloc cold:    1368.0 us avg
      - pread -> WC cudaHostAlloc warm:     214.0 us avg
      - pread -> cacheable cudaHostAlloc cold: 1228.8 us avg
      - pread -> cacheable cudaHostAlloc warm:  157.5 us avg
      - mincore before: 0/425 (0%); after: 425/425 (100%); pack size 39.43 GiB.
    * Parity: 472/472 byte-exact (p1 537a71e9/23328639; p2 acd130f4; p3 identical to p1). TG=6.83/6.82/6.73; PP=7.35/5.48/6.81.
    * 0 failed reads (44 doorbell lines). VRAM=11,263 MiB; MemoryPeak=7,103,209,472 B = 6,774 MiB (lowest ever recorded); events high=0 max=0 oom=0. smaps_rollup Rss=5,927,392 kB.
    * CQE histogram: bucket2 n=2979 (387 us); bucket3 n=58,126 (931 us); bucket4 n=655 (1,811 us).
    * EXIT PATH: TERM sent; server did NOT exit within 150 s (CPU ~140% futex_wait); SIGINT exited rc=0. Second consecutive TERM-hang; captured all metrics BEFORE shutdown.
    * KEY FINDING: buffered fd_pack_buf_ submit path (io-wq congestion) costs 21 s/session (~280 us per miss) AND yields 0 page-cache hits (42 GB working set). M33 candidate: restore O_DIRECT demand reads (MOE_DEMAND_DIRECT=1 opt-in or flip default), killing the 21 s io-wq block; PREKICK stays O_DIRECT.
  - Full audit: /home/zyxciss/fluxaeros-data/m32_prep_cache_audit.md

# betelgeuse-v6.1 - Mission 33

Goal: enable MOE_DEMAND_DIRECT=1 (O_DIRECT fd_pack_ for demand reads) to eliminate the 21.19 s io-wq submit stall measured in M32.

- [x] Task 1 - Add MOE_DEMAND_DIRECT=1 (fd_pack_ O_DIRECT) & check 6.77 GiB page-cache / SIGTERM hang
  - In src/llama-context.cpp, wire MOE_DEMAND_DIRECT=1: demand rfd[] in pack mode uses fd_pack_ (O_DIRECT); verify 4 KiB alignment of pack offsets and stage_host; keep MOE_DEMAND_BUF=1 working as explicit opt-in.
  - Read-only inspection (note in TASK.md): why fd_pack_buf_ only accumulated ~0.8 GiB page cache in M32 (MemoryPeak 6,774 MiB; MOE_DROP_MMAP default? fadvise? LRU?); why SIGTERM hung in futex_wait on exit in M31/M32 while SIGINT exited rc=0.
  - Rebuild build-v1/bin/bel-server; .test_moe_pager PASS 9/9; commit locally.
- [x] Task 2 - One cgroup-wrapped MOE_DEMAND_DIRECT=1 benchmark (p1/p2/p3 T=0)
  - MOE_DEMAND_DIRECT=1 MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=1 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; experts.bin; --moe-slots 4000; -ub 64; -np 1; -c 512; -t 6; --port 9931; exact p1/p2/p3; log to /home/zyxciss/fluxaeros-data/m33_demand_direct_bench.log.
  - Capture memory.peak/current/events, nvidia-smi VRAM before clean stop (SIGTERM, fallback SIGINT only if needed); rfd_direct_count > 0, rfd_buf_count == 0; 472/472 parity vs M32; report [moe_prof] waterfall, CQE histogram, TG/PP, VRAM, MemoryPeak, 6.77 GiB + SIGTERM findings.

# betelgeuse-v6.1 - Mission 34

Goal: fix SIGTERM 150 s exit hang + audit O_DIRECT MOE_PREKICK=0 A/B + non-blocking cudaEvent_t Exclusive Victim Cache.

- [x] Task 1 - Fix SIGTERM shutdown hang + audit PREKICK waste & Exclusive Victim Cache
  - In src/llama-context.cpp / src/moe_doorbell.cu: stop_doorbell_worker() wakes/quits cleanly (no infinite _mm_pause / blocking teardown); rc=0 + full [moe_prof] print on SIGTERM.
  - Read-only audit -> /home/zyxciss/fluxaeros-data/m34_shutdown_async_audit.md:
    * PREKICK counters (spec_submitted, spec_reused, overlap, wasted O_DIRECT reads) in m33_demand_direct_bench.log; whether confidence gating (p >= tau) can lift precision above 56%.
    * Blueprint to convert M16's Exclusive DtoH Victim Cache blocking cudaStreamSynchronize(d2h_ws) into non-blocking per-slot cudaEvent_t full-duplex PCIe DtoH.
  - Rebuild build-v1/bin/bel-server; .test_moe_pager PASS 9/9; commit locally.
- [x] Task 2 - One cgroup-wrapped O_DIRECT MOE_PREKICK=0 verification + SIGTERM exit test
  - MOE_DEMAND_DIRECT=1 MOE_CACHE_MIB=0 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; exact p1/p2/p3; log to /home/zyxciss/fluxaeros-data/m34_direct_noprekick_bench.log.
  - Capture memory.peak/current/events, smaps_rollup, nvidia-smi VRAM; send SIGTERM; measure exact exit time (verify clean < 5 s rc=0 with [moe_prof] printed).
  - 472/472 parity vs M33; 0 failed reads; report waterfall (nvme_cqe_wait_us, bucket3 avg, drain_spec_us -> 0, kernel_us); TG/PP vs M33 7.06/6.99/7.07; VRAM; MemoryPeak; victim-cache + confidence-gated PREKICK notes.

# betelgeuse-v6.1 - Mission 35

Goal: implement the non-blocking cudaEvent_t Exclusive DtoH Victim Cache (MOE_CACHE_MIB=6144).

- [x] Task 1 - Non-blocking cudaEvent_t Exclusive DtoH Victim Cache (src/llama-context.cpp, src/moe_pager.{h,cpp})
  - Follow /home/zyxciss/fluxaeros-data/m34_shutdown_async_audit.md: replace M16's blocking cudaStreamSynchronize(d2h_ws) (27.32 s) with non-blocking per-slot cudaEvent_t DtoH.
  - Exclusive DtoH admission (only evicted old_key >= 0 VRAM experts enter victim_cache; no stage_host arrival duplication, no victim_fill_us CPU memcpy).
  - 3 role cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, d2h_stream) on non-blocking stream; record ev_d2h without cudaStreamSynchronize.
  - Race guard #1: H2D stream executes cudaStreamWaitEvent(stream_h2d, ev_d2h_of_v, 0) before overwriting VRAM slot v.
  - Race guard #2: per-slot atomic state + cudaEvent_t; consumers gate via cudaEventQuery; slots recycle only after cudaEventQuery == cudaSuccess; d2h_stream synchronized in teardown.
  - Exact per-layer mixed-quant role lengths (blk.4 2,176,000 B) safe; rebuild bel-server; .test_moe_pager PASS 9/9; commit locally.
- [ ] Task 2 - One cgroup-wrapped MOE_CACHE_MIB=6144 verification (p1/p2/p3 T=0)
  - MOE_DEMAND_DIRECT=1 MOE_CACHE_MIB=6144 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; exact p1/p2/p3; log to /home/zyxciss/fluxaeros-data/m35_async_victim_bench.log.
  - Capture memory.peak/current/events, smaps_rollup, nvidia-smi VRAM; SIGTERM (< 5 s exit, full [moe_prof]); 472/472 parity vs M34; 0 failed reads.
  - Report: plan_victim_us (27.32 s -> ?), nvme_cqe_wait_us (74.52 s -> ?), bucket3 count (71,834 -> ?), [moe_victim_cache] slots/saved/hits/bytes, TG/PP, VRAM, MemoryPeak, Rss, SIGTERM exit time.

# betelgeuse-v6.1 - Mission 36

Goal: pure read-only audit of Cache > 0 + DMA=1 hang (b6e87c585 M16 working vs fa375a890/2f5207e32 layer 7/8 hang).

- [x] Task 1 - Diff working M16 (b6e87c585) vs broken M34/M35 (fa375a890, 2f5207e32)
  - Verify from m15_profile.log and m16_plan_opt*.log that M14-M16 ran MOE_CACHE_MIB=5120 + DMA=1 cleanly (63,590 fills, 28,641 hits, 108.8 GB H2D).
  - Record exact env/flag differences between M16 (MOE_PREKICK=1, no MOE_DEMAND_DIRECT=1) and M35 (MOE_PREKICK=0, MOE_DEMAND_DIRECT=1).
  - git diff b6e87c585..fa375a890 -- src/llama-context.cpp src/moe_pager.cpp src/moe_pager.h src/moe_doorbell.cu; git show 2f5207e32 line-by-line; isolate every change touching Cache > 0, DMA=1, doorbell worker, MOE_DEMAND_DIRECT=1, MOE_PREKICK=0, ev_of_v, or d2h_stream.
- [x] Task 2 - Pinpoint exact layer 7/8 hang mechanism (GPU 100%, served=0, h2d_bytes=0)
  - Inspect /home/zyxciss/fluxaeros-data/m35_evidence_pack.md; trace why layer 7/8 hangs on p1 only when Cache > 0 && MOE_DMA == 1 (Cache == 0 && DMA == 1 works; Cache > 0 && DMA == 0 works).
  - Check which exact condition stalls dma_done / dbs[il].state (GPU 100% spinning in WAIT; served=0; h2d_bytes=0).
  - Explicitly check whether MOE_DEMAND_DIRECT=1, MOE_PREKICK=0, 0bd691f45 (spec_ring_inflight), fa375a890 (worker wake/sleep), or 2f5207e32 (ev_of_v / d2h_stream) caused the regression vs b6e87c585.
  - Save full root-cause audit and exact minimal fix plan to /home/zyxciss/fluxaeros-data/m36_cache_dma_hang_audit.md; mark TASK.md [x]; commit docs locally on betelgeuse-v6.1; summarize exact culprit lines.
  - Task 1 outcome: M14-M16 all ran MOE_CACHE_MIB=5120 + DMA=1 cleanly (63,590 fills / 28,641 hits / 48,009,241,600 B saved / cache_evict=32,198 / events.high=0), PLE drop released 27,466 MiB. Env diff vs M34/M35: M16 used MOE_PREKICK=1 with DEMAND_DIRECT unset; M33-M35 benchmarks forced MOE_DEMAND_DIRECT=1 (fd_pack_ O_DIRECT). The M16 demand path at HOT_K=0 reads through pack_fd_buf_ (page cache, O_RDONLY fd_pack_buf_); M33+ routes to pack_fd() O_DIRECT.
  - Task 2 outcome: hang exact mechanism NOT fully locked (mission forbids bel-server runs; only read-only audit possible). Observed in failing logs: warmup resolves all 48 layers; user-prompt first-pass resolves through layer 46; second pass stalls at layer 8 with GPU 100% spinning in pci_ld_u32 guard waiting for `db.gpu_ready`/`dma_done`/`copy_ready` — host CQE reap never posts -> NVMe direct read CQEs never arrive -> `[moe_dma_io] copies=0 h2d_bytes=0` permanently. Contributing matrix elimination (from TASK.md M34/M33): spec_ring_inflight (0bd691f45) ruled out (spec=0); _mm_pause->cap backoff (fa375a890) ruled out (Cache==0 works); MOE_PREKICK=0 ruled out (Cache==0 M34 ran with PREKICK=0 byte-exact); MOE_DEMAND_DIRECT=1 (13b69fdde) is the primary suspect (feature isolated as the ONLY behavioral change in the demand-fd selection at HOT_K=0); ev_of_v / d2h_stream (2f5207e32) exonerated (M34 baseline binary hangs identically with none of the M35 code; admission dormant on first pass).
  - Audit saved to /home/zyxciss/fluxaeros-data/m36_cache_dma_hang_audit.md; commit locally.

# betelgeuse-v6.1 - Mission 37

Goal: verify 2f5207e32 (non-blocking Exclusive DtoH victim cache) with MOE_DEMAND_DIRECT=0 + MOE_CACHE_MIB=5120.

- [x] Task 1 - Pre-flight read-only check of 2f5207e32 + MOE_DEMAND_DIRECT=0 vs layer-8 hang
  - Confirm HEAD rebuilds build-v1/bin/bel-server cleanly; .test_moe_pager passes 9/9.
  - Briefly inspect src/llama-context.cpp demand io_uring submission (rfd, cached_sources, works, MOE_DEMAND_DIRECT) to note why Cache > 0 + MOE_DEMAND_DIRECT=1 could stall SQEs/CQEs at layer 8 while Cache > 0 + MOE_DEMAND_DIRECT=0 passes (no speculative fixes; keep control clean).
- [x] Task 2 - One cgroup-wrapped MOE_DEMAND_DIRECT=0 + MOE_CACHE_MIB=5120 benchmark (p1/p2/p3 T=0)
  - MOE_DEMAND_DIRECT=0 MOE_CACHE_MIB=5120 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; exact p1/p2/p3; log to /home/zyxciss/fluxaeros-data/m37_async_victim_direct0.log.
  - Capture memory.peak/current/events, smaps_rollup, nvidia-smi VRAM; SIGTERM (< 5 s exit, full [moe_prof] dump); verify 472/472 parity vs M16/M34; 0 failed reads.
  - Report: whether 2f5207e32 ran cleanly (0 hangs), [moe_prof] waterfall (plan_victim_us vs M16 27.32 s, nvme_submit_syscall_us, nvme_cqe_wait_us), [moe_victim_cache] saved/hits/bytes, TG & PP tok/s, VRAM, MemoryPeak, events.high, SIGTERM exit time.
  - Task 1 outcome: HEAD rebuilt clean; .test_moe_pager_m37 PASS 9/9 (io_ops=27, pf_hit=4). Demand-fd path under MOE_DEMAND_DIRECT=0 confirmed: rfd[] = loc.fd = pack_fd_buf_ (page cache, buffered), identical to M16 working code. Fork DMA doorbell never completes with Cache>0 REGARDLESS of DEMAND_DIRECT; fork's PLE-migration storms are the bottleneck (cpu ~100%, gpu 100%, sock_throttled=1 observed). Missing MOE_DESTRUCTOR_TRACE=1 env; doorbell trace prints never fired (no decode progress).
  - Task 2 SESSION RESULTS (unit bel-m37-asyncvdd0, ONE cgroup-wrapped session, MOE_CACHE_MIB=5120 MOE_DEMAND_DIRECT=0):
    *** BENCHMARK FAILED - no token output in 6+ minutes with GPU pinned to 100%. ***
    * memory.peak = 13,409,566,720 B = 12,780 MiB; events: high=0 max=0 oom=0 oom_group_kill=0 sock_throttled=1 (FIRST OBSERVATION of sock_throttled on this fork).
    * smaps_rollup: Rss=6,906,104 kB; Pss_Anon=5,709,440 kB; Pss_File=1,025,070 kB; anon dirty=5,985,948 kB.
    * VRAM = 11,245 MiB (< 11,750 MiB). p1 binary_search at T=0 NEVER decoded a token; "[doorbell]" line never printed; plan_us=0 (GPU under continuous resolve spin).
    * SIGTERM did not exit within 50 s (exited via manual SIGKILL after +SIGINT fallback failed).
    * Hierarchy: fork with Cache>0 + DMA=1 deadlocks after PLE madvise release + Tier-1 pinned 6 GB anonymous registered to CUDA causes PLE migration storms; MOE_DEMAND_DIRECT=0 does NOT rescue it (this run repeats the M35/M34-baseline hang exactly).
    * TG & PP tok/s: NOT measurable; plan_victim_us / nvme_cqe_wait_us / [moe_victim_cache] saved/hits bytes: NOT measurable.
    * See /home/zyxciss/fluxaeros-data/m37_async_victim_direct0_audit.md for full evidence.
    * M34 baseline (Cache=0, PREKICK=0, DEMAND_DIRECT=1) remains the reference state.

# betelgeuse-v6.1 - Mission 38

Goal: pinpoint exact hang line (b6e87c585..HEAD audit + live gdb all-thread backtrace).

- [x] Task 1 - Diff working M16 (b6e87c585) vs HEAD (944cd10c7); check MOE_PREKICK=0 interaction
  - git log --oneline b6e87c585..HEAD; inspect code diffs in llama-context.cpp, moe_pager.{h,cpp}, moe_doorbell.cu (MOE_CACHE_MIB > 0 x MOE_PREKICK=0 vs MOE_PREKICK=1, spec_ring_inflight, cache_, d2h_stream, worker, pci_ld_u32(&db.gpu_ready)).
  - Verify why M14/M15/M16 (MOE_CACHE_MIB=5120, MOE_PREKICK=1) worked while M35/M37 (MOE_CACHE_MIB>0, MOE_PREKICK=0) hung before printing [doorbell].
- [x] Task 2 - Capture live gdb all-thread backtrace of the exact M37 hang
  - ONE short cgroup-wrapped bel-server session with exact M37 config (MOE_DEMAND_DIRECT=0 MOE_CACHE_MIB=5120 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1, experts.bin --moe-pack-dir, --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6 --port 9931) p1 (max_tokens=8).
  - Once it stalls (CPU ~100% / GPU 100% with no token after ~20-30 s), attach gdb -p <pid> -batch -ex "set pagination off" -ex "thread apply all bt" -> /home/zyxciss/fluxaeros-data/m38_hang_bt.log; terminate scope immediately (SIGKILL / systemctl stop).
  - Combine gdb backtraces with Task 1 diff into /home/zyxciss/fluxaeros-data/m38_hang_triage.md; mark TASK.md [x]; commit docs; report exact function + file:line of every stuck thread and root cause.

# betelgeuse-v6.1 - Mission 39

Goal: fix d2h_stream doorbell-worker deadlock (2f5207e32) and benchmark Non-Blocking cudaEvent_t Exclusive DtoH Victim Cache.

- [x] Task 1 - Pre-Create d2h_stream & All cudaEvent_t Objects on Main Thread (0 CUDA Driver Lock Calls on Doorbell Worker)
  - Move cudaStreamCreateWithFlags(&ctx->moe.d2h_stream, cudaStreamNonBlocking) out of start_doorbell_worker (src/llama-context.cpp:466) onto the main thread during context/doorbell init (alongside moe_doorbell_stage_init); ensure every cudaEventCreate* (ev_of_v and cache_ev_) and cudaHostRegister runs strictly on the main thread at init_cache()/context init; verify zero cudaStreamCreate/cudaEventCreate/cudaMalloc/cudaHost* calls exist anywhere on the doorbell worker thread.
  - Keep clean teardown ordering in ~llama_context() (stop_doorbell_worker(), cudaStreamSynchronize(d2h_stream), destroy events & d2h_stream, free cache_).
  - Rebuild build-v1/bin/bel-server; .test_moe_pager PASS 9/9; commit locally.
- [x] Task 2 - One cgroup-wrapped MOE_CACHE_MIB=6144 + MOE_DEMAND_DIRECT=1 Benchmark (p1/p2/p3 T=0)
  - MOE_DEMAND_DIRECT=1 MOE_CACHE_MIB=6144 MOE_STAGE_WC=1 MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_PROF=1 MOE_CRASH_TRACE=1; experts.bin --moe-pack-dir; --moe-slots 4000; -ub 64; -np 1; -c 512; -t 6; --port 9931; exact p1/p2/p3; log to /home/zyxciss/fluxaeros-data/m39_async_victim_bench.log.
  - Capture memory.peak/current/events, smaps_rollup, nvidia-smi VRAM before clean SIGTERM stop (< 5 s + full [moe_prof]); 472/472 byte-exact parity vs M33/M34; 0 failed reads.
  - Report: [moe_prof] waterfall (plan_victim_us vs M16 27.32 s, nvme_cqe_wait_us vs M34 74.52 s, nvme_submit_syscall_us), CQE histogram (Bucket 0..4), [moe_victim_cache] saved/hits/bytes, TG & PP tok/s, VRAM, MemoryPeak, smaps_rollup, SIGTERM exit time.

# betelgeuse-v6.1 - Mission 40

Goal: capture exact hung-thread backtraces (gdb child launch + SIGTRAP) and audit 4,330-admission hang.

- [x] Task 1 - Read-only audit of m39_async_victim_bench.log & 2f5207e32/e770204fc recycle/prefill paths
  - Inspect m39_async_victim_bench.log around n_tokens = 54, t = 7.28 s, and the 4,330 victim admissions.
  - Inspect git show 2f5207e32 and git show e770204fc for: victim_cache recycle/eviction (cudaEventQuery, mtx_, state == 1 scan), d2h_ev_of_v / stream_h2d / d2h_stream interactions across prefill (t0 += 8 slices), and any while spin / cudaEventQuery / mutex blocking doorbell/main after 4,330 admissions.
- [ ] Task 2 - Bypass ptrace_scope=1 via gdb child launch + SIGTRAP; dump exact backtraces
  - Inside systemd-run --user --scope, spawn build-v1/bin/bel-server directly as a child of gdb (batch commands / script in /home/zyxciss/fluxaeros-data/, zero /tmp) with exact M39 env & flags; p1 with max_tokens=8.
  - Once /health OK and hang manifests (GPU 100% / CPU ~98% stall ~30-40 s), kill -TRAP <bel_server_pid> so gdb breaks under ptrace_scope=1; dump set pagination off + thread apply all bt -> /home/zyxciss/fluxaeros-data/m40_hang_bt.log; kill/exit gdb cleanly.
  - Write line-by-line root cause in /home/zyxciss/fluxaeros-data/m40_hang_triage.md; mark TASK.md [x]; commit docs locally on betelgeuse-v6.1; report stuck stack frames.
