# Phase-aware residency

Branch: `feat/phase-aware-residency`
Base: `e1538e9954a191d352cae0697b5908b0e477c913`
Model files: existing local checkpoint only. No model files were downloaded.

## Completed

- [x] Preserve the mmap-backed CPU expert path and the separate `feat/selective-expert-staging` experiment.
- [x] Audit allocation ownership, lifetimes, exact expert sizes, prefill workspace sizes, K/V growth, and graph invalidation in `ALLOCATION-AUDIT.md`.
- [x] Record current GPU and host memory snapshots on the RTX 3060.
- [x] Add a shared VMM backing-handle cache between prefill workspace, decode expert pools, and growable K/V. Drain unused cached handles before ordinary CUDA allocations.
- [x] Preserve existing expert residents during post-prefill refill; fill only newly available slots.
- [x] Rank new slots with actual prompt routes, static route counts, decayed decode routes, and measured per-layer CPU ring-to-ack cost per MiB. Prioritize layers 6-17.
- [x] Add runtime logging for mapped workspace, transition time, reused VMM bytes, pool sizes, and CPU service costs.
- [x] Add a fixed `INK2_RING_EXTRA` control for an apples-to-apples 4K/8K chunk comparison; keep the current default behavior.
- [x] Build and run the direct-checkpoint server with Q8 K / Q4 V, flash attention, five CPU threads, and prompt caching enabled.
- [x] Benchmark coherent source-review prompts at about 1K, 4K, 8K, and 16K, with 512 generated tokens and three repeats per context for both chunk sizes.
- [x] Save timing responses, the 4K server trace, an 8K 16K route capture, and RAM/VRAM samples under `raw/`.
- [x] Confirm zero CPU misses in looped layers 6-17 in the captured decode traces.
- [x] Stop the stale server the user identified; later start and stop a fresh best512-profile server for the requested coherent 16K rerun.

## Follow-up required

- [ ] Establish exact output/logit parity against the base runtime. Greedy output hashes differed across repeated identical prompts, so exact numerical equivalence is not proven.
- [ ] Exercise an actual cached conversation continuation. Prompt caching remained enabled, but repeated full prompts reported zero cached tokens.
- [ ] Run a long-context stability check at 32K or above. This round reached a 16K prompt and 20,480 K/V cells only.
- [ ] Preserve and compare an 8K full-matrix server trace if per-run CPU miss counts are needed; the saved 8K server log is a one-run 16K route capture.
- [ ] Make a before/after speed comparison against the unmodified base binary if the project needs a direct feature speedup claim. Current throughput numbers compare 4K and 8K chunks on the phase-aware build.
- [ ] Reproduce the 40.17 tok/s decode reference on the coherent 16K prompt; the fresh best512-profile run measured 28.99 tok/s median.
- [ ] Create separate commits after the user provides commit subjects. Repository instructions prohibit AI-written commit messages; do not commit or push without that input.
