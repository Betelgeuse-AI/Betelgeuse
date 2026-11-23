#!/usr/bin/env bash
set -euo pipefail

strata_bin=${STRATA_BIN:-/home/zyxciss/strata-inkling2-build/strata-ink2}
betelgeuse_bin=${BETELGEUSE_BIN:-build/bin/bel-inkling2}
model=${INK2_MODEL:-/home/zyxciss/Models/Inkling-2-Mini}
routes=${INK2_ROUTE_PROFILE:-/home/zyxciss/Docs/strata-inkling2-rtx3060/evidence/inkling2-routes-profiled.freq}
prefix_source=${INK2_PREFIX_SOURCE:-/home/zyxciss/Docs/strata-inkling2-rtx3060/evidence/prefill-prefix-16384.ids}
output_root=${INK2_OUTPUT_ROOT:-tools/inkling2/benchmarks/prefill-tuning-16k/raw}

mkdir -p "$output_root"
work_dir=$(mktemp -d /tmp/ink2-16k-tune.XXXXXX)
trap 'rm -rf "$work_dir"' EXIT

python3 - "$prefix_source" "$work_dir/prompt.ids" <<'PY'
from pathlib import Path
import sys

ids = [item for item in Path(sys.argv[1]).read_text().strip().replace("\n", ",").split(",") if item]
if len(ids) < 15872:
    raise SystemExit(f"source prefix has only {len(ids)} tokens; need 15872")
Path(sys.argv[2]).write_text(",".join(ids[:15872]))
PY

run_case() {
    local label=$1
    local binary=$2
    local kv=$3
    local chunk=$4
    local mx_fraction=$5
    local log="$output_root/${label}-${kv}-chunk${chunk}-mx${mx_fraction}.log"

    printf 'Running %s kv=%s chunk=%s mx_fraction=%s\n' "$label" "$kv" "$chunk" "$mx_fraction"
    env STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 INK2_MX_POOL_FRACTION="$mx_fraction" \
        "$binary" --model "$model" --max-context 16384 --chunk "$chunk" --kv "$kv" --tc 1 \
        --vram-reserve-mib 0 --cpu-threads 5 --kv-grow 1 --kv-grow-step 4096 \
        --tokens-file "$work_dir/prompt.ids" --n 512 --expert-frequencies "$routes" >"$log" 2>&1
    printf 'Saved %s\n' "$log"
}

for kv in k8v4 i8r; do
    run_case strata "$strata_bin" "$kv" 4096 0.08
    run_case betelgeuse "$betelgeuse_bin" "$kv" 4096 0.08
    run_case strata "$strata_bin" "$kv" 8192 0.30
    run_case betelgeuse "$betelgeuse_bin" "$kv" 8192 0.30
done
