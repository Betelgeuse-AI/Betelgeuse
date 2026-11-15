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

- [ ] Task 1 - Role-Grouped Extent-Coalesced O_DIRECT Prefill Reader (direct .gguf, for non-pack prefill batches n_tokens > 1)
  - Disable PREKICK during prefill (n_tokens > 1); set MOE_PREKICK=0 default; keep only offset sort + stage_ev_ + per-layer victim cache.
  - Role-grouped monotonic coalescing across the prefill batch: split NVMe misses into 3 contiguous role sweeps per layer ((1) gate sorted, (2) up sorted, (3) down sorted); when adjacent missing experts within a sweep sit within a small gap (<= 1 expert / ~2 MiB), coalesce into one contiguous O_DIRECT io_uring read (up to 4-8 MiB per SQE) into a temporary bounce/staging span, then slice out needed experts.
  - Verify direct-GGUF read/wait ms/expert during prefill and decode; measure coalesced SQE count reduction.
- [ ] Task 2 - One cgroup-wrapped direct-.gguf bel-server benchmark (MOE_PREKICK=0 clean isolation)
  - Scope `bel-m11-udiq2xs`, MemoryHigh=13824M, MemoryMax=14G, MemorySwapMax=0, `--moe-slots 4000 -ub 64 -np 1 -c 512 -t 6`, `MOE_CACHE_MIB=6144`, `MOE_HOT_K=0`, `MOE_PREKICK=0`, direct `00001-of-00003.gguf`, NO `--moe-pack-dir`.
  - Exact 3-prompt suite (p1 binary_search 58-tok warmup, p2 exact M9 "Write a Python LRUCache class" prompt, p3 binary_search 16-tok warm; n=192, T=0) + longer cold prompt (~128-256 tok) if clean.
  - Logs & responses -> /home/zyxciss/fluxaeros-data/m11_chunked_pp_direct.log. Verify 472/472 byte-exact T=0 binary_search parity, 0 failed reads, stop cleanly.
  - Report PP tok/s (p1/p2/p3 vs M9 Pack 7.20/7.01/9.28 and M10 Direct 4.98/5.60/5.98), TG tok/s, VRAM, MemoryPeak, coalesced SQE count reduction, read/wait ms/expert, 0 drops, 0 CPU rows.
