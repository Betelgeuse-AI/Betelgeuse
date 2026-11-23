#!/usr/bin/env bash
set -euo pipefail

strata_bin=${STRATA_BIN:-/home/zyxciss/strata-inkling2-build/strata-ink2}
betelgeuse_bin=${BETELGEUSE_BIN:-build/bin/bel-inkling2}
model=${INK2_MODEL:-/home/zyxciss/Models/Inkling-2-Mini}
routes=${INK2_ROUTE_PROFILE:-/home/zyxciss/Docs/strata-inkling2-rtx3060/evidence/inkling2-routes-profiled.freq}
prefix_4k=${INK2_PREFIX_4K:-/home/zyxciss/Docs/strata-inkling2-rtx3060/evidence/betelgeuse-4k-8k/ink2-prefix-3584.ids}
prefix_8k=${INK2_PREFIX_8K:-/home/zyxciss/Docs/strata-inkling2-rtx3060/evidence/betelgeuse-4k-8k/ink2-prefix-7680.ids}
out_dir=${INK2_OUTPUT_DIR:-tools/inkling2/benchmarks/paired-4k-8k/raw-rerun}

mkdir -p "$out_dir"

run_case() {
    local label=$1
    local binary=$2
    local kv=$3
    local ctx=$4
    local prefix=$5
    local log="$out_dir/${label}-${kv}-${ctx}.log"

    printf 'Running %s %s %s\n' "$label" "$kv" "$ctx"
    env STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 INK2_MX_POOL_FRACTION=0.08 \
        "$binary" --model "$model" --max-context "$ctx" --chunk 4096 --kv "$kv" --tc 1 \
        --vram-reserve-mib 0 --cpu-threads 5 --kv-grow 1 --kv-grow-step 4096 \
        --tokens-file "$prefix" --n 512 --expert-frequencies "$routes" >"$log" 2>&1
    printf 'Saved %s\n' "$log"
}

for ctx in 4096 8192; do
    if [[ $ctx == 4096 ]]; then
        prefix=$prefix_4k
    else
        prefix=$prefix_8k
    fi
    for kv in k8v4 i8r; do
        run_case strata "$strata_bin" "$kv" "$ctx" "$prefix"
        run_case betelgeuse "$betelgeuse_bin" "$kv" "$ctx" "$prefix"
    done
done
