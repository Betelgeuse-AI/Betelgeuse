#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
data_root=${INK2_DATA_ROOT:-$HOME/fluxaeros-data/inkling2-dspark-2026-11-23}
jobs=${BUILD_JOBS:-2}

mkdir -p "$data_root/builds"

for block in 2 3 5 8; do
    build_dir="$data_root/builds/cmake-dblk$block"
    cmake -S "$repo_root" -B "$build_dir" \
        -DGGML_CUDA=ON \
        -DLLAMA_BUILD_INKLING2_ENGINE=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_CUDA_FLAGS="-DINK2_SPEC_BLOCK=$block" \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$data_root/builds/dblk$block/bin" \
        >"$data_root/builds/config-native-fp8-dblk$block.log" 2>&1
    cmake --build "$build_dir" --target llama-ink2 -j"$jobs" \
        >"$data_root/builds/build-native-fp8-dblk$block.log" 2>&1
done
