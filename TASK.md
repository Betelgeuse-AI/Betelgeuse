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
