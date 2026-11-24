# Why the controlled run and the shell loop differed

Date: 2026-11-24

## Commands compared

The controlled run used `run-comparison.sh` under a 14 GB cgroup with swap disabled, explicit
`INK2_*` values, `env -u INK2_SPEC_DRAFT_N_MAX -u INK2_SPEC_FORCE -u INK2_SPEC_DIAG -u
INK2_ROUTE_PROFILE`, `--no-webui`, and the archived `prompt-16384.ids` prompt with 512 generated
tokens. It measured 905.84 pp/s / 40.31 tok/s on run 2 (cold run 1 was lower).

The shell loop used `build/bin/bel-server` without those unsets or cgroup, repeated
"The quick brown fox jumps over the lazy dog. " 21 to 1365 times (253 to 16,381 tokens), and
generated 256 tokens. Its 16K rows were about 28.1-28.3 tok/s.

## Findings

1. `server-context.cpp` had started reading `INK2_SPEC_DRAFT_N_MAX` from the environment and
   passing `--spec 1 --spec-draft-n-max N` to the engine. The controlled run unsets that variable;
   a shell that had run the DSpark experiments does not.
   - Before the fix, with `INK2_SPEC_DRAFT_N_MAX=2` on the repeated 16K prompt: 27.4 and 25.9
     tok/s at 256 / 512 generated tokens.
   - Plain control on the same prompt: 30.1 and 29.6 tok/s.
   - The server now forces plain decode and ignores that variable.
2. Prompt content is the larger difference. After the fix, with all stale tuning variables set
   to bad values and a bad `INK2_ROUTE_PROFILE`:
   - repeated 16K prompt, 512 tokens: 28.2 and 29.9 tok/s;
   - archived 16K IDs, 512 tokens: 40.0 and 40.24 tok/s.
   The route profile sweep on the repeated prompt gave only 29.4-30.9 tok/s for `profiled`,
   `best512`, and `256k-decode-k8v4`, so this is a routing/content effect, not a profile choice.
3. Generation length (256 versus 512) changed the repeated-prompt rate by less than 1 tok/s.
4. `--no-webui` and the cgroup are not the cause of the decode gap. The controlled cgroup peak was
   13.05 GB with zero high/max/OOM events; the sampled VRAM peak was 11,832 MiB.
5. The archived 16K token IDs used for the earlier 40 tok/s result detokenize to incoherent word
   fragments. That result is not a natural-prompt baseline. The coherent replacement run and its
   input audit are recorded in `RESULTS.md` under Mission 5 and in
   `/home/zyxciss/fluxaeros-data/inkling-m5/prompt-audit.txt`.

## Fix applied

- Removed the environment-driven DSpark activation in `init_inkling2_backend`.
- Added explicit `--spec 0` to the engine arguments.
- Forced the tuned engine environment unless `INK2_ALLOW_TUNING_OVERRIDE` is set.
- Ignored stale `INK2_ROUTE_PROFILE` unless `INK2_ALLOW_TUNING_OVERRIDE` is set.
- Unset `INK2_SPEC_DRAFT_N_MAX`, `INK2_SPEC_FORCE`, `INK2_SPEC_DIAG`, and `INK2_NUMA_POOL`
  before starting the embedded engine.
- Rebuilt `bel-server` and `bel-cli`. A startup log with the stale variables set now shows
  1,911 NVFP4 + 90 MXFP8 slots, int8 CPU experts, automatic `best512` profile, and no DSpark line.
- Verified temperature/top-k/top-p/min-p/seed: seeds 42 and 43 at temperature 0.8 produced
  different completions with no unsupported-parameter error.

## Scope of the performance target

The 60-77 tok/s quoted for Strata is from an RTX 3090 (all experts resident) or an RTX 3090
limited to 12 GB with 2x Xeon Gold 6252 and 22 CPU threads per socket. This RTX 3060 has an
i5-12400 with 5 effective CPU workers and about 59% expert residency; about 40 tok/s at 16K is
the measured ceiling here with the archived prompt. No code change can make the same hardware
reach a dual-Xeon or 24 GB resident-engine figure.

## Historical KV settings and chunk size

On the Strata engine, `k8v4` with chunk 4096 and the 256K route profile measured 692.8 pp/s /
37.9 tok/s at 16K, while chunk 8192 with MX fraction 0.30 raised prefill to 847.5 pp/s but
collapsed decode to 20.2 tok/s because the post-refill pool held only 56% of the experts. The
Earlier Betelgeuse results used i8r and chunk 4096. The coherent Mission 5 run explicitly used
`--cache-type-k q8_0 --cache-type-v q4_0 -fa on`, which the direct server maps to K8/V4 and tiled
CUDA attention. Prompt caching stayed enabled; its reported cache count was zero because the repeated
standalone prompt was shorter than the previous prompt-plus-generation history.
