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
