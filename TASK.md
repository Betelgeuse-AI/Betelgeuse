# betelgeuse-v6.1 — Mission 3

Working branch: `betelgeuse-v6.1` (was reset from 09e69a047 onto **v1.0.0** = `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`).
`fluxaeros-mtp` (09e69a047) and `betelgeuse` (4d8a379ce) untouched. Net reference patch saved at `/home/zyxciss/fluxaeros-data/fluxaeros_net_diff.patch` (10,189 lines, 515,607 B).

- [x] Task 1 — Port fork-only MoE modules, ggml/, src/ core, and qwen4exp.cpp onto v1.0.0
  - 29 fork-added files copied wholesale (moe_pager/doorbell/predictor, llama-moe-profile, docs, prompts, data, tests, tools); rollback coverage already consolidated upstream into `tests/llama-context/cases-rs.cpp` (fork's stale `tests/test-recurrent-state-rollback.cpp` intentionally not re-added).
  - ggml/ ported: `TOP_K_MASK` + `GGML_OP_MOE_SLOT_RESOLVE{,_KICK,_WAIT,_PREKICK}` retained in suffix position; `GGML_OP_COUNT = 103`; CPU switch aborts for graph-only ops; CUDA graphs only at `nt == 1` (v1.0.0 chunked CUDA graph path used).
  - src/ ported with upstream kept intact: #30565, upstream `llama-moe-cache` (`--moe-cache-mib`/`--moe-cache-cpu-share`), #30927 `hc_stream_mean` + #31660, native `--ple`/`--shard`; fork keeps NVMe pager, Tier-1 host victim cache, `moe_slot_resolve_sync`, KICK/WAIT/PREKICK.
  - `src/models/qwen4exp.cpp`: v1.0.0 base + only MOE_PLE_STAGING and moe_direct_io/prekick/KICK/WAIT hooks.
- [x] Task 2 — Port common/ + tools/server minimal delta & build build-v1/ (sm_86 CUDA)
  - Fork's `--moe-slots`/`--moe-profile`/`--moe-pack-dir`/`--moe-chunk-max`/`--moe-direct-io`/`--cache-warmup`/`--auto-warm` flags applied; v1.0.0 DFlash2 and `--ple` intact; drafts never inherit MoE settings; `tools/server/server-context.cpp` warmup-path delta applied only.
  - `build-v1/` configured out-of-tree (`cmake -B build-v1 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc ...`); `llama-server`, `llama-cli`, `test-llama-context` all build clean (incremental rebuilds: 0 warnings; only pre-existing fork `-Wcast-qual` idioms + NCCL-not-found note remain).
  - Port committed on betelgeuse-v6.1 as `0473b9c0273537094788d3a1c77048b862a4f88f`.
