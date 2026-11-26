# v0.7.0-alpha-orionis release smoke validation

Build: `betelgeuse-v6.1`, integration tip `db1f05d1b` plus the prefill host-pin lifetime fix and
release documentation. CMake: `GGML_CUDA=ON`, `GGML_CUDA_ARCHITECTURES=86`,
`CMAKE_CUDA_ARCHITECTURES=86`, `LLAMA_BUILD_INKLING2_ENGINE=ON`, `LLAMA_BUILD_SERVER=ON`.
Compiler GCC 16.1.1, CUDA 13.3, driver 610.43.02, RTX 3060 12 GB.

All requests used greedy decoding (`temperature=0`) through `bel-server` with a local Inkling 2 Mini
checkpoint directory. The published 45.27 tok/s median and 42.03 tok/s coherent 16K figure come
from the 512-token validation set; these smoke checks verify serving behaviour, not peak speed.

## Default profile, cold request

- `cache_prompt=true`, 22 prompt tokens, 16 generated tokens.
- GPU-fetch arena prepared after prefill: 8383 MiB pinned for evictable outer experts.
- Phase transition 12.82 s (one-time arena preparation), decode 41.44 tok/s.
- Response completed normally with `finish_reason=length`.

## Repeated request

- Identical prompt submitted again in the same process.
- Completed normally with a 29.8 ms phase transition and 46.88 tok/s decode.
- No crash, no stale graph or pool state.

## Warm continuation

- Conversation with user, assistant and a new user message submitted.
- 38 prompt tokens, 16 generated tokens, 29.8 ms phase transition, 46.73 tok/s decode.
- Completed normally.

## Streaming

- `stream=true`, 16 generated tokens.
- 33 SSE data chunks plus `data: [DONE]`, 49.50 tok/s decode.
- Completed normally.

## GPU-fetch default and override

- With `INK2_GPU_FETCH=0` but no `INK2_ALLOW_TUNING_OVERRIDE`, the server forced GPU fetch on and the
  arena preparation log appeared. Decode was 38.46 tok/s on an 8-token request.
- With `INK2_ALLOW_TUNING_OVERRIDE=1 INK2_GPU_FETCH=0 INK2_PIN_HOST_MIB=512`, the server respected
  the override. It pinned 509 MiB across 110 route-ranked experts and did not activate GPU fetch.

## Prefill host-pin lifetime and KV growth

- The override run submitted a 4816-token prompt with `-c 8192`.
- KV grew from 4096 to 8192 cells; the phase transition was 427.6 ms and the decode graph was
  recaptured and ready.
- The prefill host pin pool survived the prefill-to-decode boundary. Without the one-line lifetime
  fix, `gpu_fetch_release()` freed the pool and left dangling expert pointers.
- The request completed normally: 542.71 tok/s prefill, 37.78 tok/s decode.

## CUDA graph and pool behaviour

- Every smoke transition logged `graph ready` and `cached VMM 0 MiB`.
- Graph reallocations were zero in the short-context requests.
- Pool refill and sliding-window restore/resize logs appeared on warm requests without errors.

## Packaged binary check

The binary archive was extracted to a clean directory and run through its `run` launcher:
`./run bel-cli --help`, `./run bel-server --help`, `./run bel-inkling2 --help` and a short live
server request. See `validation/package-smoke.json` in the archive.
