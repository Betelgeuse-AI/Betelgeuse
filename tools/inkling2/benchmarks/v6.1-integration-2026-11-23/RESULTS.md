# Betelgeuse v6.1 Inkling integration: restored direct engine

Date: 2026-11-23
Branch: `betelgeuse-v6.1`
GPU: NVIDIA RTX 3060 12 GiB
System RAM: 15 GiB

The official safetensors directory selects the direct checkpoint engine linked into `bel-server`.
`bel-cli` uses the same path by starting its local `bel-server`. Neither path converts the checkpoint
to GGUF, uses Marlin, or starts a second model process. This replaces the earlier 63.65 pp/s /
26.34 tok/s result in this directory.

## Regression and fix

The earlier `bel-server` measurement used a 250-token prompt with `-c 16384 -t 5`, default engine
environment, K8/V4 KV, no route profile, and an 8192-token chunk:

| Prompt processing | Decode | Expert residency |
| ---: | ---: | ---: |
| 63.65 pp/s | 26.34 tok/s | 1,618 NVFP4 + 220 MXFP8 (58%) |

Three causes were found:

1. The embedded engine source had lost its protocol I/O. The serve loop read `std::cin` and wrote
   `stdout` instead of the in-process `ink2::embedded_server` queue, so a rebuild from source could
   not serve `bel-server` requests. Restored `ink2_emit` / `ink2_read_line` helpers and the
   `ink2::engine_main` wrapper in `ink2_engine.cu`.
2. `init_inkling2_backend` used the engine defaults: `INK2_OFF_MARGIN_MIB=128`,
   `INK2_MX_POOL_FRACTION=0.20`, no `--expert-frequencies`, `--kv k8v4`, and `--chunk 8192` for
   contexts up to 64K. That produced the 58% residency and the slow short-prompt decode.
3. The route profile was not passed to the engine at all, so loop layers 6-17 used index order.

The direct path now sets, before starting the embedded engine:
`STRATA_NUMA_POOL=0`, `INK2_CPU_ACT=int8`, `INK2_CPU_OPT=29`, `INK2_OFF_MARGIN_MIB=64`,
`INK2_MX_POOL_FRACTION=0.08`, `--kv i8r`, `--chunk 4096`, `--graph 1`, `--tc 1`, `--spec 0`,
`--vram-reserve-mib 0`, `--cpu-threads 5`, `--kv-grow 1 --kv-grow-step 4096`, and a route profile.
Explicit environment overrides and `INK2_ROUTE_PROFILE` are honored. `-t 5` is the default when
`--threads` is not set. At 16K the restored placement is 1,911 NVFP4 + 90 MXFP8 (59%, 8.66 GiB);
at 1K it is 1,919 + 90 (60%).

Route profiles are installed beside the checkpoint:
`inkling2-routes-256k-decode.freq` for all contexts except 16K prompt + 512 output, and
`inkling2-routes-best512.freq` when `n_ctx` is 16,385-20,480 (the `-c 16896` 16K case). The
`inkling2-routes-profiled.freq` file remains as a fallback. The 256K decode profile is what lifts
16K decode from 35.0 to 39.2 tok/s in the standalone engine; the `best512` profile is best in the
server at a 16,384-token prompt.

## `bel-server` result table

Each row is one greedy `/v1/completions` request with 512 generated tokens. "Warm" is the second
request on the same engine. Target values are the requested table. Combined time is prompt ms +
decode ms, excluding load.

| Prompt + output | Target pp/s | Warm pp/s | Cold pp/s | Target tok/s | Warm tok/s | Cold tok/s | Target s | Warm s | Cold s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 + 512 | 465 | 454.7 | 246.8 | 40.0 | 37.66 | 37.70 | 15.0 | 15.82 | 17.70 |
| 2,048 + 512 | 635 | 654.7 | 336.5 | 39.8 | 39.62 | 39.66 | 16.1 | 16.02 | 18.97 |
| 4,096 + 512 | 725 | 695.5 | 542.3 | 39.5 | 40.87 | 40.85 | 18.6 | 18.39 | 20.06 |
| 8,192 + 512 | 730 | 903.8 | 657.2 | 39.0 | 40.45 | 40.36 | 24.4 | 21.70 | 25.13 |
| 16,384 + 512 | 690 | 890.1 | 764.6 | 38.0 | 40.29 | 37.86 | 37.1 | 31.09 | 34.92 |
| 32,256 + 512 | 655-670 | 851.7 | 781.0 | 35-36 | 36.45 | 36.42 | 64-65 | 51.90 | 55.33 |

The 16K and 32K rows use the automatic profile selection: `best512` at `-c 16896`, 256K decode
profile at `-c 32768`. The other rows use the 256K decode profile.

## Comparison

- PR #6520 `strata-ink2` plain 16K: 691.5 pp/s / 37.9 tok/s. Restored `bel-server` 16K warm:
  890.1 pp/s / 40.29 tok/s; cold: 764.6 pp/s / 37.86 tok/s. Both prefill values beat 691.5, and
  warm decode beats 37.9. Standalone `bel-inkling2` at 16K prompt + 512 with `best512`: 767.6 /
  39.6; with the 256K profile: 781.7 / 38.4 (repeat 780.9 / 38.6).
- Prior integrated Betelgeuse reference at 16K total: 789.9 pp/s / 35.4 tok/s with i8r. Restored
  standalone 16K total (15,872 prompt + 512): 822.3 / 39.2. Restored server warm 16K:
  890.1 / 40.29.
- Native Q4/Marlin `bel-server` (76.8 pp/s / 17.48 tok/s) is a different llama.cpp path and is not
  a baseline for the direct engine.

## Standalone direct engine, one-shot

Same checkpoint, environment, and settings, but driven by `bel-inkling2 --tokens-file` rather than
the HTTP server. One run per row; 512 generated tokens.

| Prompt tokens | Route profile | Prefill | Decode |
| ---: | --- | ---: | ---: |
| 1,024 | 256K decode | 224.0 pp/s | 37.7 tok/s |
| 2,048 | 256K decode | 365.4 pp/s | 39.7 tok/s |
| 4,096 | 256K decode | 657.6 pp/s | 41.5 tok/s |
| 8,192 | 256K decode | 736.5 pp/s | 40.4 tok/s |
| 16,384 | best512 | 767.6 pp/s | 39.6 tok/s |
| 16,384 | 256K decode | 781.3 pp/s | 38.5 tok/s |
| 32,256 | 256K decode | 799.5 pp/s | 36.5 tok/s |

The standalone 16K total case (15,872 prompt) reached 822.3 pp/s / 39.2 tok/s. A profile recorded
from that exact decode overfit and fell to 29.4 tok/s; it was not adopted.

## `bel-cli`

Rebuilt `bel-cli` and verified a local-server smoke test:
`bel-cli -m /home/zyxciss/Models/Inkling-2-Mini -c 1536 -t 5 -p "Hello, my name is" -n 8
--no-display-prompt -st`. It loaded the direct checkpoint in-process (1,919 NVFP4 + 90 MXFP8,
11.32 GiB VRAM) and generated at 30.2 tok/s on the 7-token prompt. The short prompt is not
comparable to the 512-token table above. `bel-cli` inherits the server fixes because it calls
`llama_server()` in a local thread.

## Legacy GGUF + Marlin `bel-server` (not comparable)

These measurements are from the older GGUF execution path, built with CUDA,
`LLAMA_INKLING_MARLIN=ON`, and the compact relative-attention path. The model was
`build-inkling-probe/inkling-mini-nvfp4.gguf` with packed experts from
`build-inkling-probe/marlin-host-ranked`.

Configuration: context 32K, batch and ubatch 128, 5 threads, one sequence, all possible layers on
GPU, fit disabled, flash attention enabled, Q4 K/V cache. Marlin placement used 8960 MiB GPU,
8192 MiB host, full host coverage, 80% protected experts, 2233 MiB W8 pool, and transfer overlap.

| Request | Prompt processing | Decode |
| --- | ---: | ---: |
| Short, 8 prompt tokens + 512 output | 33.61 pp/s | 27.14 tok/s |
| 16,384 prompt tokens + 512 output | 87.38 pp/s | 18.02 tok/s |

## Limits

- Short-context results have the largest run-to-run variance. An earlier warm pass measured 338.3
  and 510.7 pp/s at 1K and 2K; the table uses the later `run.sh` reproducer results. 1K prefill is
  close to the 465 pp/s target, but 1K decode (37.7 tok/s) remains below 40.0. 2K warm prefill
  (654.7) exceeds 635. A profile recorded from a 1K decode overfit and reduced decode to 32.2
  tok/s, so the 256K decode profile was kept.
- Cold 16K decode is 37.86 tok/s (best512) and 36.13 tok/s with the 256K profile. The target table
  behavior is the warm row at 40.29 tok/s. A second request on the same engine is representative
  of steady state.
- Direct checkpoint mode supports one active text request and temperature, top-k, top-p, and
  min-p sampling. Image input, sleep/reload, external draft models, and parallel requests are not
  supported. Visible `<|end_message|>` cleanup still needs a model-aware parser fix.
- All rates are single runs, not medians.

## Artifacts

- Raw benchmark logs and measurements: `raw/`
- Route profiles beside the checkpoint: `inkling2-routes-best512.freq`,
  `inkling2-routes-256k-decode.freq`
- Prompts: `~/Docs/strata-inkling2-rtx3060/evidence/prefill-prefix-16384.ids`
- Strata comparison: `~/fluxaeros-data/strata-inkling2-eval-2026-11-23/REPORT.md`

## Mission 1 - 16K warm pair and loop-core floor

Date: 2026-11-24. This is one paired comparison session. The preserved server binary and its
matching `libllama-server-impl.so` ran first; the rebuilt source ran second. Both used the same
16,384-token prompt IDs, context 16,896, greedy sampling, i8r KV, 4,096 prompt chunk, five CPU
threads, `best512` route profile, and 512 output tokens. The second request per server is the warm
row. Full commands, JSON, server logs, memory samples, and prompt IDs are in
`/home/zyxciss/fluxaeros-data/inkling-m1/`.

| Build | Request | Prefill | Decode | Combined |
| --- | --- | ---: | ---: | ---: |
| Preserved | cold | 778.74 pp/s | 37.42 tok/s | 34.695 s |
| Preserved | warm | 857.35 pp/s | 40.17 tok/s | 31.831 s |
| Core-floor | cold | 617.45 pp/s | 37.98 tok/s | 39.991 s |
| Core-floor | warm | 505.20 pp/s | 40.35 tok/s | 45.096 s |

Every request generated 512 tokens. All four generated-text SHA-256 values match. The preserved
warm result is 3.7% below the supplied 890.1 pp/s, 40.29 tok/s, 31.09 s result. The candidate
matches decode within 0.2 tok/s but loses 41% warm prefill throughput and takes 13.27 s longer than
the preserved warm request. This is not a performance pass.

The measured candidate temporarily reserved at least `NLOOP * NE = 1,536` NVFP4 slots for lazy
direct-checkpoint offload. This floor was removed after the paired run: it changed warm decode by
only +0.18 tok/s while reducing warm prefill by 41%.

The preserved build began with 1,984 NVFP4 + 94 MXFP8 slots and fell to 1,276 + 64 during the 16K
prompt. The candidate began with 1,911 + 90 and retained 1,546 + 64, including all 1,536 core
experts plus 10 outer NVFP4 experts. `INK2_MISS_DIAG=1` counted zero decode route misses on layers
6-17 in both candidate requests. Outer-layer miss totals were 8,878 and 8,808. Decode expert
staging remained zero; nonresident experts ran through the CPU worker path. The preserved build
refilled to 1,861 NVFP4 slots before decode; since `expert_placement_order()` puts all 1,536 core
experts first, source placement implies the baseline also had the full core resident at decode.
The baseline had no miss counter, so this last statement is inferred from the refill code and slot
count rather than directly counted.

Peak sampled VRAM was 11,832 MiB for both builds. Process `VmHWM` was 12,266 MiB preserved and
12,585 MiB candidate. Candidate reported 6,076 MiB of nonresident expert source. The process has an
8 MiB `RLIMIT_MEMLOCK`; `VmLck` and `VmPin` were zero, so the mmap-backed source was not pinned. Do
not describe this state as pinned host RAM.

### DSpark audit for Mission 2

The current embedded engine has a fixed `INK2_SPEC_BLOCK` draft window and no
`--spec-draft-n-max` argument. `spec_continue()` verifies target rows in two-row groups. In the CPU
miss path, `moe_b()` selects `k_cache_cpu_batch()` and queues each unique routed miss with all its
rows; `expert_parts()` returns before staging for CPU verification, and `k_cpu_wait()` joins the
worker outputs. The worker reads the file-backed `ExpertParts` mmap. Thus the old 214.4 MB/token
forced-run log describes an older path; current source does not stage the whole missing expert set
for CPU verification. Without CPU verification, `PF.verify_route_misses` stages routed misses only.

The CLI still rejects CPU experts with speculation and graph capture because captured CPU verify
has not been shown equivalent to the uncaptured path. `forward_prefill()` captures only `T == DBLK`,
while the current verifier submits at most two rows per call. A recorded DBLK2 graph1 check diverged
from graph0, so graph capture is not cleared for use. `draft_sample_method` is absent from current
source. The direct DSpark feature taps are visits 5, 23, and 41 plus final normalized output. Older
forced runs favored one draft but remained slower than plain; the later CPU-hybrid DBLK2 record is
30.2 tok/s with 0.51 accepted drafts per verify versus 37.1 tok/s plain, with exact 128-token
output and no staging. No DSpark run was made for this mission.

The core-floor experiment achieved zero loop-core misses and output parity, but it did not improve
decode and caused a large prefill regression, so the floor was removed. Host pinning remains unavailable
under the current 8 MiB hard memlock limit. See the replaced `TASK.md` for the current checkbox
ledger and open work.
