# Inkling-2-Mini CUDA engine

This target integrates the Betelgeuse Inkling-2-Mini CUDA execution path. CUDA builds include it when
`LLAMA_BUILD_INKLING2_ENGINE=ON`. It reads the official safetensors checkpoint, keeps the BF16 dense weights and packed
NVFP4/MXFP8 experts in their source formats, and builds the matching reference executable. No GGUF
conversion or Marlin path is used.

Configure with CUDA enabled. Set `LLAMA_BUILD_INKLING2_ENGINE=OFF` to omit this model path:

```sh
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_INKLING2_ENGINE=ON
cmake --build build --target llama-ink2 llama-ink2ref llama-server bel-cli -j
```

`bel-inkling2` is the standalone runtime and `llama-ink2ref` is the reference executable. Both read
the safetensors directory directly.

## Standalone engine

For a fixed-token run, pass `--model` with the checkpoint directory, `--tokens-file` with token IDs,
and `--n` with the generated token count. The dependable settings are i8r KV, a 4096-token prompt
chunk, tensor-core dense projections, CUDA graph decode, 5 CPU expert threads, KV growth in 4096
steps, and the tuned offload environment:

```sh
INK2_NUMA_POOL=0 \
INK2_CPU_ACT=int8 \
INK2_CPU_OPT=29 \
INK2_OFF_MARGIN_MIB=64 \
INK2_MX_POOL_FRACTION=0.08 \
build/bin/bel-inkling2 \
  --model /home/zyxciss/Models/Inkling-2-Mini \
  --max-context 16896 --chunk 4096 --kv i8r --tc 1 --graph 1 --spec 0 \
  --vram-reserve-mib 0 --cpu-threads 5 --kv-grow 1 --kv-grow-step 4096 \
  --tokens-file PREFIX.ids --n 512 \
  --expert-frequencies /home/zyxciss/Models/Inkling-2-Mini/inkling2-routes-best512.freq
```

Use `inkling2-routes-256k-decode.freq` instead for 32K and longer contexts. `bel-inkling2 --serve`
exposes the embedded line protocol used by the server.

## `bel-server` and `bel-cli`

`bel-server -m <checkpoint directory>` auto-selects the direct embedded engine when the directory
contains the safetensors shards. `bel-cli -m <checkpoint directory>` does the same by starting
`llama_server()` in a local thread. There is no `bel-inkling2` child process.

```sh
build/bin/bel-server -m /home/zyxciss/Models/Inkling-2-Mini -c 16896 -t 5 \
  --host 127.0.0.1 --port 8080

build/bin/bel-cli -m /home/zyxciss/Models/Inkling-2-Mini -c 1536 -t 5 \
  -p "Hello, my name is" -n 8 --no-display-prompt -st
```

The server sets the same tuned environment as the standalone command, defaults to 5 CPU threads,
uses i8r KV with a 4096-token chunk, and picks a route profile beside the checkpoint:
`inkling2-routes-best512.freq` for a 16K prompt plus 512 output (`-c 16896`) and
`inkling2-routes-256k-decode.freq` otherwise. Set `INK2_ROUTE_PROFILE` to override the choice, or
set any of the `INK2_*` variables explicitly to override the defaults. Direct
mode supports one active text request; no images, sleep/reload, draft models, or parallel requests.

The embedded engine uses `ink2::embedded_server` for its line protocol, so a rebuild keeps
`bel-server` and `bel-cli` working. The checkpoint reader is shared with the reference executable.

## Results

- Server and direct-engine measurements: `benchmarks/v6.1-integration-2026-11-23/RESULTS.md`
- Paired standalone runs: `benchmarks/paired-4k-8k/`
- Fused GQA decode attention: `benchmarks/gqa-decode-2026-11-23/`
