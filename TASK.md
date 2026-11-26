# Inkling-2-Mini direct-checkpoint validation

Date: 2026-11-24
Branch: `betelgeuse-v6.1`

## Goal and test rules

Recreate the warm 16,384-prompt + 512-output result with a coherent prompt, confirm the direct
checkpoint path and request options work after rebuild, keep the looped core on the GPU, and measure
a safe pinned-host expert pool on the RTX 3060 12 GB system.

- Keep prompt caching enabled for every production-style measured request (`cache_prompt: true`).
- Use `--cache-type-k q8_0 --cache-type-v q4_0 -fa on` for the measured requests.
- Use the local checkpoint; do not download model files.
- Keep logs and prompt artifacts under `<artifact-dir>/`; do not use `/tmp`.
- Do not push. The paired benchmark session for this task is complete; do not repeat it as a speed run.

## Completed

- [x] Replaced the previous task ledger with this current mission.
- [x] Built `bel-server`, `bel-cli`, and `bel-inkling2` from the current source.
- [x] Restored the embedded in-process protocol queue, and confirmed rebuilds do not route requests
  through process stdin/stdout.
- [x] Passed direct request options `ignore_eos` and `cache_prompt` from the server. Direct mode
  accepts `ignore_eos` rather than returning the earlier 501 error.
- [x] Mapped server flags `--cache-type-k q8_0 --cache-type-v q4_0` to direct-engine K8/V4; startup
  reports `k8v4 KV and tiled CUDA attention`. `-fa on` was included in every measured command.
- [x] Kept a 16,384-token coherent server-documentation briefing as the measured prompt. Its SHA-256
  is `ae012289b87cc6909e812fa5e8d818ebb824870a20ddcd555f83c901d91f47a4`; prompt text, IDs, and
  metadata are under `<artifact-dir>/inkling-m5/`.
- [x] Kept `cache_prompt: true` for all four measured requests. The reported cache count was zero:
  after generation, the same standalone prompt was shorter than the saved prompt-plus-generation
  history, so it did not reuse a prefix. The warm row is same-process full prefill, not a cache hit.
- [x] Ran one mmap-backed baseline and one bounded-pool candidate, each with a first and warm
  request, using 16,384 prompt tokens, 512 generated tokens, context 16,896, greedy sampling,
  `ignore_eos: true`, 5 CPU threads, and the `best512` route profile.
- [x] Warm mmap-backed result: 909.87 pp/s, 27.94 tok/s, 36.294 s combined. First request:
  853.69 pp/s, 26.27 tok/s, 38.647 s.
- [x] Warm 511 MiB pinned-pool result: 900.52 pp/s, 27.26 tok/s, 36.940 s combined. First request:
  766.86 pp/s, 25.52 tok/s, 41.385 s.
- [x] Audited the old `857.35 pp/s / 40.17 tok/s` prompt IDs. They detokenize to incoherent word
  fragments; this is not a representative coherent-prompt baseline. The audit note and new prompt
  results are in `<artifact-dir>/inkling-m5/`.
- [x] Confirmed all 1,536 loop-core experts (physical layers 6-17) are resident in VRAM. Current
  placement also holds 394/1,024 outer NVFP4 and 91/512 MXFP8 experts. No extra loop-core floor is
  needed; a prior floor experiment was removed because it hurt prefill without improving decode.
- [x] Added a bounded host pin pool that copies route-ranked nonresident experts from the
  safetensors mapping into `cudaHostAlloc` memory. It selected 68 experts / 511 MiB; the other
  5,987 MiB stayed mmap-backed. The pool is capped by cgroup headroom with a 768 MiB reserve.
- [x] Verified the baseline and candidate stayed below `MemoryHigh=14G` and `MemoryMax=15G`, with
  swap disabled and zero cgroup high, max, or OOM events. Peak sampled VRAM was 11,814 MiB and
  11,832 MiB, respectively.
- [x] Recorded process and cgroup memory peaks, request JSON, output hashes, and the server log in
  `<artifact-dir>/inkling-m5/`. The warm output hash matches between the two expert
  source modes; first-request hashes also match each other. First and warm hashes differ, so this
  does not establish cold-to-warm output identity.
- [x] Completed the read-only DSpark verify audit: current CPU verify batches unique nonresident
  experts across target rows and does not stage the full missing set. CPU + speculation + graph
  remains disabled because captured CPU verify was not shown equivalent; current graph capture and
  verify row sizes do not line up. No verify implementation was changed here.

## Not achieved / remaining

- [ ] Reproduce 40.17 tok/s with a coherent 16K prompt. The measured coherent warm result was
  27.94 tok/s. The old word-fragment prompt is not a valid natural-prompt comparison.
- [ ] Show a decode or prefill gain from the pinned-host pool. The one warm candidate request was
  slightly slower than the mmap-backed warm request; only one warm sample per build was taken, so
  the variance is not bounded.
- [ ] Pin all nonresident experts. The tested safe budget allowed 511 MiB; copying the full
  5,987 MiB source is not supported by the observed 15 GB memory envelope. Keep the cold remainder
  mmap-backed unless a new memory plan demonstrates safe headroom.
- [ ] Establish output identity between first and warm requests. Their hashes differ, although
  corresponding baseline and pinned-pool request hashes match. Save generated text and investigate
  near-tie/router variation before claiming exact quality parity.

## Result files

- `tools/inkling2/benchmarks/v6.1-integration-2026-11-23/RESULTS.md`
- `tools/inkling2/benchmarks/v6.1-integration-2026-11-23/RUN-DIFFERENCE.md`
- `<artifact-dir>/inkling-m5/`
