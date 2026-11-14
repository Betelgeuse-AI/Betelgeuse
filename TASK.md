# betelgeuse-v6.1 — Mission 8

Goal: unlock Tier-1 victim cache, true O_DIRECT doorbell, optimal VRAM slot budget on UD-IQ2_XS.

- [x] Task 1 — Forensic audit of 4.05 tok/s & 13,652 MiB MemoryPeak + fix O_DIRECT / victim cache / MMAP
  - Audit saved to /home/zyxciss/fluxaeros-data/m8_throttle_audit.md.
  - Why [moe_victim_cache] didn't fire: victim-cache eligibility was gated on `pager.pack_mode()` in 2 places (llama-context.cpp) — UD-IQ2_XS has no --moe-pack-dir, so the host victim cache was never activated. Fix (59bb9252c): gate on cache_bytes()/DMA env only.
  - Shard 2 O_DIRECT + doorbell path: pager opens every shard with O_DIRECT; slices carry their shard fd (d7d606031 was missing that for the demand prefetch path); io_uring demand reads at ~1.2-1.4 ms/expert, failed=0, `uring=1`. Q2_0 (460,800 B, 512-aligned) slices still use 4 KiB O_DIRECT windows (align-down off + align-up len, pre-pad preserved) — no buffered fallback observed.
  - MOE_DROP_MMAP=1 verified present; drops 72,319 MiB of trunk/PLE residency post-warmup (already default-on).
  - CUDA graphs confirmed firing during nt==1 decode (graphs reused = 341 by the end of the timed requests; 0 reallocations).
  - PREKICK: NOT enabled. Gap: (a) hard pack_mode gate (`spec_ok` requires it), (b) Rule A (no slot-map mask before top-K in the prekick graph), (c) Rule B (`moe_prekick_kernel` launched as single block `<<<1,32>>>` — not multi-SM per spec), (d) no non-pack speculative reader exists in the codebase. Rule C (strict demand priority via `demand_arrived` break) is present. Left MOE_PREKICK=0 and reported exact gap.
  - Root cause of degenerate output: victim-cache arena was sized by LAYER 0'S slices only (1,777,664 B slots); layers with larger experts (IQ2_S+Q5_0, e.g. blk.4 = 2,176,000 B) overflowed every arena slot and smashed neighboring slots, so cache hits returned garbage. Fixed (2588e2f67): size by the largest gate+up+down payload across ALL layers. Arena then verified byte-exact against direct shard reads for the largest combos.
  - Verified via standalone harness (`/home/zyxciss/fluxaeros-data/m8_verify/`): arena fill (commit_locked path) byte-exact for layers 0,1,2,3,5,6,7,8,12,24,47 (before the stride fix) and for ALL layers after.
- [x] Task 2 — One cgroup-wrapped bel-server verification (~4000 VRAM slots + MOE_CACHE_MIB=6144)
  - ONE cgroup-wrapped session (systemd-run --user --scope -p MemoryHigh=13824M -p MemoryMax=14G -p MemorySwapMax=0 --unit=bel-m8-udiq2xs): MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1 MOE_DROP_MMAP=1 MOE_CACHE_MIB=6144 bel-server -m .../Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf -ngl 99 --moe-direct-io --moe-slots 4000 -ub 64 -np 1 -c 512 -t 6 --host 127.0.0.1 --port 9931.
  - Full log: /home/zyxciss/fluxaeros-data/m8_bel_udiq2xs_tuned.log (earlier degenerate runs of this same config preserved as m8_bel_udiq2xs_arena_misalign_bug.log and m8_bel_udiq2xs_victimcache_bug.log).
  - Requests (temp=0): warmup "Write a Python binary_search function" max_tokens=192 → COHERENT PYTHON CODE (decode 192 tok @ 6.5 tok/s). LRUCache (warm): decode 192 @ 5.51 tok/s, content empty (all tokens spent on reasoning). binary_search (warm): COHERENT correct code (decode 192 @ 5.68 tok/s, cache_n=48).
  - VRAM 11,235 MiB (< 11,750 cap); cgroup memory.peak = 14,331,285,504 B = 13,661 MiB (under MemoryHigh 13824M = 14,495,514,624 B); 0 swap; graphs reused = 639; prefill cuda ops=36, cpu ops=1 (RESHAPE only); 0 CPU expert rows; 0 failed expert reads.
  - [moe_victim_cache]: saved=69109 hits=34694 bytes=58,992,998,400 (≈34,700 hits, ~55% of warm-request misses served from host cache; 59 GiB of NVMe re-reads avoided), cache_evict=31835.
  - [moe_slots] hit ~75.1%; [moe_dma_io] copies=69,223 h2d_bytes=118 GB; demand reads ~1.2-1.4 ms/expert, failed=0.
  - Shutdown: SIGTERM, scope stopped, VRAM back to 15 MiB.
  - Note: earlier in this mission, three bel-server launches of the same config produced the degenerate "333..." constant output until the arena-stride fix landed; root cause and fix timeline documented in m8_throttle_audit.md and the m8_bel_*_bug.log files.
