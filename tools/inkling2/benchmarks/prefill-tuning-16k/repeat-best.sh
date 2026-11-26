#!/usr/bin/env bash
set -euo pipefail

strata_bin=${STRATA_BIN:-<strata-bin>}
betelgeuse_bin=${BETELGEUSE_BIN:-build/bin/bel-inkling2}
model=${INK2_MODEL:-<checkpoint-dir>}
routes=${INK2_ROUTE_PROFILE:-<evidence-dir>/inkling2-routes-profiled.freq}
prefix=${INK2_PREFIX_SOURCE:-<evidence-dir>/prefill-prefix-16384.ids}
output_root=${INK2_OUTPUT_ROOT:-tools/inkling2/benchmarks/prefill-tuning-16k/raw/repeat-best}
cpu_threads=${INK2_CPU_THREADS:-5}
chunk=${INK2_CHUNK:-6144}
mx_fraction=${INK2_MX_POOL_FRACTION:-0.18}

mkdir -p "$output_root"

run_case() {
    local label=$1
    local binary=$2
    local kv=$3
    local log="$output_root/${label}-${kv}-chunk${chunk}-mx${mx_fraction}-cpu${cpu_threads}.log"

    printf 'Running %s kv=%s chunk=%s mx_fraction=%s cpu_threads=%s\n' "$label" "$kv" "$chunk" "$mx_fraction" "$cpu_threads"
    env STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 INK2_MX_POOL_FRACTION="$mx_fraction" \
        "$binary" --model "$model" --max-context 32768 --chunk "$chunk" --kv "$kv" --tc 1 \
        --vram-reserve-mib 0 --cpu-threads "$cpu_threads" --kv-grow 1 --kv-grow-step 4096 \
        --tokens-file "$prefix" --n 512 --expert-frequencies "$routes" >"$log" 2>&1
    printf 'Saved %s\n' "$log"
}

run_case betelgeuse "$betelgeuse_bin" k8v4
run_case strata "$strata_bin" k8v4
run_case strata "$strata_bin" i8r
run_case betelgeuse "$betelgeuse_bin" i8r
