# Phase-aware residency results

- Branch: `feat/phase-aware-residency`
- Base runtime: `e1538e9954a191d352cae0697b5908b0e477c913`
- GPU: RTX 3060, 12,288 MiB
- Host RAM: 16,151,204 kB
- Model: local `thinkingmachines/Inkling-2-Mini` checkpoint; no model files were downloaded.

## Method

The prompt is a CUDA runtime review task with source excerpts from the fixed base revision. It contains no repeated filler. Each prompt ends with a complete review question and assistant turn. Actual token counts were 1,022, 4,094, 8,192, and 16,338. Prompt text and IDs are under `prompts/`.

Each request generated 512 tokens with temperature 0, top-k 0, top-p 1, min-p 0, and `ignore_eos=true`. Prompt caching stayed enabled (`cache_prompt=true`). Both chunk runs used context 32,768, five CPU threads, AVX2 int8 CPU activations, MXFP8 pool fraction 0.08, 64 MiB offload margin, 512 MiB pinned host expert pool, and `INK2_RING_EXTRA=8192`. K/V used Q8 K and Q4 V with flash attention enabled.

The prior prefix-cut prompts and their raw results remain in `prompts/invalid-prefix-*` and `raw/`. They are excluded from the tables because the prefixes stopped inside the user message and caused source continuation instead of a review answer.

## Throughput

Three runs per context. The table reports the median across the three runs. `Prompt ms` is the server's prompt evaluation time and serves as TTFT here.

| Actual prompt tokens | 4K chunk pp/s | 4K chunk tok/s | 4K prompt ms | 8K chunk pp/s | 8K chunk tok/s | 8K prompt ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,022 | 420.4 | 29.86 | 2,431 | 435.3 | 30.26 | 2,348 |
| 4,094 | 801.8 | 30.69 | 5,106 | 857.9 | 31.05 | 4,772 |
| 8,192 | 860.8 | 29.69 | 9,517 | 1,010.5 | 28.76 | 8,107 |
| 16,338 | 890.2 | 29.95 | 18,354 | 1,050.4 | 29.51 | 15,554 |

Across the warm second and third repeats, the 8K chunk improved prefill by about 1%, 5%, 20%, and 18% at the four prompt sizes. Decode stayed near 30 tok/s; the 8K chunk did not provide a repeatable decode gain. A separate cold 16K run on a fresh 8K server measured 804 pp/s, 27.30 tok/s, and 20.32 s prompt time.

### Fresh best512-profile rerun

After restarting the server with an 8K chunk and context 16,896, automatic profile selection chose `inkling2-routes-best512.freq`. K/V stayed Q8 K / Q4 V with flash attention. On the same 16,338-token coherent prompt, two 512-token requests measured 764.7 / 28.49 and 1,052.7 / 29.49 pp/s / tok/s. Their median was 908.7 pp/s and 28.99 tok/s (18.44 s median prompt time). `cache_prompt` stayed enabled; the server reported zero cached tokens. The two generated text hashes differed.

This fresh run clears the requested 857.35 pp/s prefill reference, but it does not reproduce the 40.17 tok/s decode reference. The server was stopped after the run.

### Fresh pre-merge repeat

A fresh process repeated the same two-request workload from the pending phase-aware worktree. It used an 8K chunk, context 16,896, the best512 route profile, five CPU workers, K8/V4, flash attention, `--spec 0`, and `cache_prompt=true`.

| Request | Prompt ms | Decode ms | Prefill pp/s | Decode tok/s | Reused prompt tokens |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17,870 | 20,792 | 914.3 | 24.58 | 0 |
| 2 | 16,359 | 16,736 | 998.7 | 30.53 | 0 |
| Median | 17,114.5 | 18,764 | 956.5 | 27.55 | 0 |

Both requests generated 512 tokens. The two output hashes differed. Against the recorded 908.7 pp/s and 28.99 tok/s median, this pair was 5.3% faster in prefill and 5.0% slower in decode. Prompt caching was enabled, but the server reported no reused tokens. Raw responses are in `raw/merge-check-chunk-8192.jsonl`; the post-merge branch verification is a separate check.

## Reclaimed pages and residency

At startup, the engine placed 1,875 NVFP4 + 88 MXFP8 experts for the 8K run and 1,876 + 88 for the 4K run. All 1,536 experts in looped layers 6-17 were resident. A 16K request grew K/V to 20,480 cells, or 0.27 GiB.

| Chunk | Prompt size | Mapped prefill workspace | VMM handles remapped at handoff | Newly filled slots | Final 16K decode pool | Handoff time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 1K | 922 MiB | 922 MiB | 172 NVFP4 + 24 MXFP8 | - | 0.16-0.24 s |
| 4K | 4K-8K | 2,706 MiB | 2,702-2,708 MiB | 589-590 NVFP4 + 24 MXFP8 | - | 0.47-0.58 s |
| 4K | 16K | 2,706 MiB per chunk | about 2,704 MiB | 589-590 NVFP4 + 24 MXFP8 | 1,826-1,837 NVFP4 + 88 MXFP8 | 0.47-0.48 s |
| 8K | 1K | 922 MiB | 922 MiB | 172 NVFP4 + 24 MXFP8 | - | 0.16-0.21 s |
| 8K | 4K | 2,780 MiB | about 2,780 MiB | 607 NVFP4 + 24 MXFP8 | - | about 0.48 s |
| 8K | 8K | 5,166 MiB | 5,164-5,168 MiB | 1,166 NVFP4 + 24 MXFP8 | - | about 0.90-1.25 s |
| 8K | 16K | 5,166 MiB per chunk | 5,164-5,168 MiB | 1,165-1,166 NVFP4 + 24 MXFP8 | 1,836 NVFP4 + 88 MXFP8 | 0.90-1.17 s |

The 8K workspace supports faster batched prefill, but at 16K both chunk sizes ended with almost the same decode pool after K/V growth. The 8K run therefore buys prompt speed, not more final decode residency, in this memory configuration. The mapped pages are reusable physical VMM handles; the small cached remainder was about 20-22 MiB.

## CPU misses and memory

Every captured trace had zero CPU misses in layers 6-17. For the 4K-chunk runs, total CPU expert misses per generated token, taking the median of three traces, were 44.21 at 1K, 42.69 at 4K, 45.71 at 8K, and 45.03 at 16K. One cold 8K-chunk 16K route-capture run had 54.15 misses/token and zero loop-core misses.

The worker reports ring-to-ack service measurements during real inference. In warmed 4K runs, admission logs were generally about 47-50 us/MiB for NVFP4 and 49-52 us/MiB for MXFP8. Cold first-touch measurements were much higher, so admission cost is sensitive to file-backed page faults and warm state.

| Memory sample | 4K chunk | 8K chunk |
| --- | ---: | ---: |
| Peak `bel-server` RSS | 12,439 MiB | 12,137 MiB |
| Peak `bel-server` VRAM | 11,834 MiB | 11,834 MiB |
| Peak total GPU use, including display | 11,857 MiB | 11,857 MiB |
| Minimum `MemAvailable` during sampled run | 12,132 MiB | 12,046 MiB |
| File-backed nonresident expert source | 6,242 MiB | 6,246 MiB |
| Pinned host expert copy | 511 MiB | 511 MiB |

No full anonymous copy or full checkpoint pin was allocated. The 4K and 8K process RSS peaks include ordinary engine and mmap residency; the full VMM prompt workspace was on the GPU.

## Correctness and limits

- The server built from the phase-aware source ran every prompt to 512 generated tokens without allocation failure. At 16K, K/V grew through 20,480 cells and the decode pool was refilled.
- Prompt caching was enabled for every request, but `cache_n` remained zero because these repeated full prompts did not reuse a prior prefix. Conversation continuation was not tested.
- Repeated greedy responses were coherent but their text hashes differed across repeats. Exact output or logit parity with the base runtime is not established; this remains a correctness item.
- The measured prompt ceiling is 16K with a 32K configured context. No 32K or 256K run was made, so longer-context stability is not established by this experiment.
- These results compare 4K and 8K chunks on the phase-aware build. They do not measure a before/after speedup against the base binary.

## Reproduction artifacts

- Prompt builder: `prepare_prompts.py`
- Benchmark runner: `run_server_bench.py`
- Server launch: `start_server.sh`
- Memory sampler: `sample_memory.py`
- Full timing responses: `raw/phase-aware-coherent-chunk-4096.jsonl`, `raw/phase-aware-coherent-chunk-8192.jsonl`
- Captured server details: `raw/server-phase-aware-coherent-chunk-4096.log`, `raw/server-phase-aware-coherent-chunk-8192.log`
- Fresh best512-profile rerun: `raw/fresh-best512-chunk-8192.jsonl`, `raw/server-fresh-best512.log`
- Memory samples: `raw/memory-phase-aware-coherent-chunk-4096.csv`, `raw/memory-phase-aware-coherent-chunk-8192.csv`
- Cold 8K route capture: `raw/phase-aware-coherent-routecheck-chunk-8192.jsonl`
