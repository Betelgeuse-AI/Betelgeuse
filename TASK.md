# betelgeuse-v6.1 — Mission 4

Working branch: `betelgeuse-v6.1` (119f7a015 = v1.0.0 port + bel re-implementation + fix).
`fluxaeros-mtp`, `betelgeuse`, UD-IQ2_XS download, and `build/` all untouched.

- [x] Task 1 — Re-implement bel-server & bel-cli on v1.0.0 + build build-v1/ (verified)
  - bel-server/bel-cli from betelgeuse inspected read-only (tools/cli/*, tools/server/server.cpp + CMakeLists.txt, common/betelgeuse-splash.h, assets/generate_terminal_logo.py). Ported onto v1.0.0 only:
    - `tools/server/CMakeLists.txt`: OUTPUT_NAME bel-server + add_custom_target (target name unchanged); default port 9931 (v1.0.0 default).
    - `tools/cli/CMakeLists.txt`: OUTPUT_NAME bel-cli + add_custom_target.
    - `common/betelgeuse-splash.h` ported verbatim (self-contained inline banner); used by bel-cli + bel-server entries; no llama_moe_finish_pass / moe-stopwatch / hardcoded DeepSeek paths; no terminal-logo generation (not self-contained; splash header's inline banner used instead).
    - bel-cli TUI (cli-context/cli-client/cli-server/cli-ui.h/cli-markdown.h) adapted from betelgeuse onto v1.0.0 APIs (dump_safe kept); hero banner made honest (bel-cli v1.0.0, no false machine/version claims).
    - test-cli-markdown.cpp ported; passes ("test-cli-markdown: all assertions passed").
    - One INP_TOKENS crash fix: staged-embedding path left the token leaf unconsumed (dropped from the graph); fixed with `ggml_build_forward_expand(gf, inp->tokens)` (+mixed). Committed as `061ee24dcfb0efdd46a89ac9f25cf7deca546f8c`.
  - Built out-of-tree in `build-v1/` (sm_86, same CMake flags as reference `build/`, plus `CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc`). `bel-server`, `bel-cli`, `test-llama-context` built; incremental rebuilds clean (0 warnings; only pre-existing fork `-Wcast-qual` idioms + NCCL-not-found note remain).
  - Verified: `build-v1/bin/bel-server --help` and `build-v1/bin/bel-cli --help` both exit 0 and print the fork flags (`--moe-direct-io`, `--moe-slots`, `--moe-pack-dir`, `--moe-chunk-max`, `--auto-warm`, `--cache-warmup`, `--ple, --shard`, `--spec-type`); `ldd build-v1/bin/bel-server` resolves to build-v1 libs only.
  - Bel port commit: `1a17076707b878f0389fc615941f87292557c050`; staged-embedding fix commit: `061ee24dcfb0efdd46a89ac9f25cf7deca546f8c`.
- [x] Task 2 — Model check + ONE cgroup-wrapped bel-server baseline (no spec decode)
  - Environment: aria2c pid 59045 still downloading UD-IQ2_XS (NOT touched). Alpha-Orionis-XS = reassembled logical GGUF split: Dense.gguf (bytes [0,3770978720)) + experts.bin (bytes [3770978720,39252693408)) + PLE.gguf (shard 1, split.no=1). Reassembled byte-exact as `Alpha-Orionis-XS-assembled/Qwen3.8-Flash-Next-00001-of-00002.gguf` (39252693408 B; PLE auto-discovered via split pattern). Originals untouched.
  - Launch: `systemd-run --user --scope --quiet -p MemoryHigh=13824M -p MemoryMax=14G -p MemorySwapMax=0 --unit=bel-m4-baseline3 build-v1/bin/bel-server -m .../Alpha-Orionis-XS-assembled/Qwen3.8-Flash-Next-00001-of-00002.gguf -ngl 99 --moe-direct-io --moe-slots 5000 -ub 64 -np 1 -c 512 -t 6 --host 127.0.0.1 --port 9931` (env: MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0 MOE_IO_DIAG=1).
  - Model loaded; MoE audit TRUE-VRAM 3264 MiB; PLE table (27466 MiB) and token_embd (322 MiB) host-aliased; hit rate 99.5%; ~941 page-ins for session.
  - VRAM 10497 MiB / 12288 (1413 free); cgroup MemoryPeak 13697007616 B = 13061 MiB (~12.76 GiB < 13824M); end-of-run current 10918 MiB.
  - 0 page failures, 0 ranks replaced, 0 CPU fallback expert rows; single prefill CPU op = RESHAPE (trivial).
  - Warmup prompt (2 sentences): 63 prompt tokens @ 896.6 ms (70.3 tok/s), 48 tg @ 1164.3 ms (40.4 tok/s), cache_n=0.
    Timed gen ("hash map internal", 96 tok): cache_n=42, 22 @ 328.4 ms (67.0 tok/s), 96 @ 2280.6 ms (41.7 tok/s).
    Coding prompt ("implement binary_search...", 128 tok): cache_n=42, 39 @ 552.6 ms (70.6 tok/s), 128 @ 3062.1 ms (41.5 tok/s).
    Probe ("Name one primary color", 16 tok): cache_n=42, 18 @ 330.7 ms (54.4 tok/s), 16 @ 363.9 ms (41.2 tok/s).
  - Degeneracy: IQ2_XS 2-bit GSQ-RCO quant emits constant '3' for every prompt (temperature=0); engine verified end-to-end; degeneracy is a quant artifact (identical with MOE_PLE_STAGING=0 and with bel-cli).
  - Clean shutdown via SIGTERM in ~2 s; VRAM released (15 MiB).
  - Full baseline log: `/home/zyxciss/fluxaeros-data/m4_bel_server_baseline.log` (includes failed-attempt log history).
