#!/usr/bin/env bash
set -euo pipefail

data_root=${INK2_DATA_ROOT:-$HOME/fluxaeros-data/inkling2-dspark-2026-11-23}
model=${INK2_MODEL:-$HOME/Models/Inkling-2-Mini}
routes=${INK2_ROUTE_PROFILE:-$data_root/inkling2-routes-profiled.freq}
prompt=${INK2_PROMPT:-$data_root/prompts/strata-ink2-prompt-257.ids}
output_root=${INK2_OUTPUT_ROOT:-$data_root/raw}
baseline=${BASELINE_BIN:-$data_root/builds/cmake-dblk8/bin/bel-inkling2}
gen_tokens=${INK2_N:-512}
cpu_threads=${INK2_CPU_THREADS:-5}

mkdir -p "$output_root"

run_case() {
    local label=$1
    local binary=$2
    local kv=$3
    local spec=$4
    local log="$output_root/$label-$kv.log"
    local args=()
    if [[ "$spec" == 1 ]]; then
        # Speculative verification stages file-backed experts; graph capture is unsupported here.
        args=(--model "$model" --max-context 32768 --chunk 256 --kv "$kv" --tc 1 --graph 0
            --vram-reserve-mib 0 --cpu-threads "$cpu_threads" --kv-grow 1 --kv-grow-step 4096
            --tokens-file "$prompt" --n "$gen_tokens" --expert-frequencies "$routes" --spec 1)
    else
        args=(--model "$model" --max-context 32768 --chunk 4096 --kv "$kv" --tc 1 --graph 1
            --vram-reserve-mib 0 --cpu-threads "$cpu_threads" --kv-grow 1 --kv-grow-step 4096
            --tokens-file "$prompt" --n "$gen_tokens" --expert-frequencies "$routes")
    fi

    printf 'Running %s kv=%s spec=%s\n' "$label" "$kv" "$spec"
    if [[ "$spec" == 1 ]]; then
        env STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 \
            INK2_SPEC_FORCE=1 INK2_MX_POOL_FRACTION=0.08 "$binary" "${args[@]}" >"$log" 2>&1
    else
        env STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 \
            INK2_MX_POOL_FRACTION=0.08 "$binary" "${args[@]}" >"$log" 2>&1
    fi
    printf 'Saved %s\n' "$log"
}

run_case baseline "$baseline" k8v4 0
for block in 2 3 5 8; do
    if [[ "$block" == 8 ]]; then
        binary=$baseline
    else
        binary="$data_root/builds/cmake-dblk$block/bin/bel-inkling2"
    fi
    run_case "forced-spec-block$block" "$binary" k8v4 1
done
