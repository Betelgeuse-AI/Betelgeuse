# Betelgeuse v0.7.0 (Alpha-Orionis)

### Native Inkling 2 Mini support. Built for memory-constrained hardware.

This release introduces a substantially redesigned inference engine for running Thinking Machines Lab's Inkling 2 Mini on consumer GPUs using the original mixed NVFP4, MXFP8 and BF16 checkpoint.

Developed and optimised on a single RTX 3060 12 GB with just 16 GB of single-channel DDR4 system memory.

No additional weight quantisation. No expert skipping. No speculative decoding in the reported benchmarks.

## Performance

Test hardware:

- NVIDIA RTX 3060 12 GB
- Intel Core i5-12400
- 16 GB single-channel DDR4
- Approximately 360 GB/s theoretical GPU memory bandwidth
- Linux, CUDA SM86
- Original Inkling 2 Mini checkpoint

| Workload | Generation |
|---|---:|
| LRU cache with TTL | 44.51 tok/s |
| Directed cycle detection | 45.52 tok/s |
| Race-condition reasoning | 47.11 tok/s |
| Token-bucket rate limiter | 44.33 tok/s |
| Serializability reasoning | 45.27 tok/s |
| Median | 45.27 tok/s |
| Mean | 45.35 tok/s |
| Coherent 16K source review | 42.03 tok/s |

All five short workloads use 512 generated tokens with ordinary greedy decoding through bel-server.

### Prompt processing

| Measurement | Performance |
|---|---:|
| Cold uncached 16K prefill | 818.9 tok/s |
| Warm uncached 16K prefill | 1,232.8 tok/s |
| Warm 16K phase transition | 203.4 ms |
| Initial GPU-fetch preparation | Approximately 11 seconds |

Warm uncached prefill processes the complete prompt again. It does not reuse previously calculated prompt KV states.

## What's new

### Native mixed-precision inference

- Direct execution of the original NVFP4, MXFP8 and BF16 checkpoint.
- Packed expert weights retained in GPU memory.
- Ampere-specific low-bit GEMV kernels.
- Packed FP16 tensor-core prefill.
- Optimised BF16 projection kernels.
- GPU-side greedy sampling.
- Full recurrent execution and original expert selection.

### Architecture-aware expert residency

- Preferential residency for recurrent-core experts.
- GPU execution of nonresident experts through a bounded expert cache.
- Decayed-LFU eviction.
- Overlapped PCIe transfers and resident expert computation.
- Staged expert transfers overlapping gate/up and down projections.
- Context-aware NVFP4 and MXFP8 pool allocation.
- CPU fallback under memory constraints.

### Dual-phase inference

Prefill and decode have different memory and computation requirements.

Betelgeuse now dynamically reconfigures its resources between these phases inside the same server process.

Prefill uses larger tensor-core workloads and temporary workspace. After prompt processing, the workspace is reclaimed and reused for expert residency and decode.

The model, KV cache and recurrent states remain intact throughout the transition.

### Attention and long-context support

- Fused grouped-query attention.
- Optimised sliding-window attention.
- Dynamic KV growth.
- Memory-aware expert pool resizing.
- CUDA graph reconstruction following pool changes.
- K8/V4 KV-cache support.

Earlier native development builds completed populated contexts approaching 262K tokens. Performance decreases at longer contexts as KV storage competes with expert residency. Maximum-context results for the final release should be treated separately from those historical measurements.

### Unified server

The native Inkling engine is integrated directly into bel-server.

This release prioritises a consistent server interface, shared engine configuration, streaming and multi-request operation rather than a separate experimental runtime.

## Current limitations

- The first request can incur a substantial one-time GPU-fetch arena preparation cost.
- Performance depends on expert locality, available VRAM and system-memory bandwidth.
- Long-context generation remains slower than short-context generation.
- Validation is concentrated on Linux and NVIDIA Ampere hardware.
- Original checkpoint weights are preserved, but different execution and KV-cache precisions can produce numerical differences.

## What's next

- Intel integrated GPU expert execution using shared system memory.
- Learned expert routing prediction and asynchronous prefetching.
- DSpark speculative decoding.
- Training-free multi-head latent attention retrofits.
- Chunk-indexed sparse attention.
- Further long-context memory and bandwidth optimisation.

## Research preview: latent attention

An independent experimental branch has demonstrated approximately 3.06x global KV-cache compression without gradient-based retraining.

At 128K context, global KV storage was reduced from 1.59 GiB to 0.52 GiB using a rank-512 FP8 latent representation.

The combined latent and sparse attention prototype remains experimental due to unresolved retrieval-quality problems and is not included in this release.

## Acknowledgements

Betelgeuse builds upon the llama.cpp ecosystem and native Inkling engine work contributed to Strata by baptisterajaut.

Thanks to the developers of llama.cpp, Strata, Thinking Machines Lab and the wider local inference community.

Repository: https://github.com/Betelgeuse-AI/Betelgeuse
