#!/usr/bin/env bash
set -euo pipefail

strata_bin=${STRATA_BIN:-/home/zyxciss/strata-inkling2-build/strata-ink2}
betelgeuse_bin=${BETELGEUSE_BIN:-build/bin/bel-inkling2}
model=${INK2_MODEL:-/home/zyxciss/Models/Inkling-2-Mini}
routes=${INK2_ROUTE_PROFILE:-/home/zyxciss/Docs/strata-inkling2-rtx3060/evidence/inkling2-routes-profiled.freq}
prefix_source=${INK2_PREFIX_SOURCE:-/home/zyxciss/Docs/strata-inkling2-rtx3060/evidence/prefill-prefix-16384.ids}
output_root=${INK2_OUTPUT_ROOT:-tools/inkling2/benchmarks/scale-8k-16k-262k/raw}
run_strata_262k=${RUN_STRATA_262K:-1}

mkdir -p "$output_root"
work_dir=$(mktemp -d /tmp/ink2-scale.XXXXXX)
trap 'rm -rf "$work_dir"' EXIT

python3 - "$prefix_source" "$work_dir" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1]).read_text().strip()
ids = [item for item in src.replace("\n", ",").split(",") if item]
if not ids:
    raise SystemExit("empty source token prefix")

out = Path(sys.argv[2])
for context in (8192, 16384, 262144):
    prompt_tokens = context - 512
    repeated = (ids * ((prompt_tokens + len(ids) - 1) // len(ids)))[:prompt_tokens]
    (out / f"prompt-{context}.ids").write_text(",".join(repeated))
PY

run_case() {
    local label=$1
    local binary=$2
    local kv=$3
    local context=$4
    local prefix="$work_dir/prompt-$context.ids"
    local log="$output_root/${label}-${kv}-${context}.log"

    printf 'Running %s kv=%s context=%s\n' "$label" "$kv" "$context"
    env STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 INK2_MX_POOL_FRACTION=0.08 \
        "$binary" --model "$model" --max-context "$context" --chunk 4096 --kv "$kv" --tc 1 \
        --vram-reserve-mib 0 --cpu-threads 5 --kv-grow 1 --kv-grow-step 4096 \
        --tokens-file "$prefix" --n 512 --expert-frequencies "$routes" >"$log" 2>&1
    printf 'Saved %s\n' "$log"
}

for context in 8192 16384 262144; do
    for kv in k8v4 i8r; do
        if [[ "$context" != 262144 || "$run_strata_262k" == 1 ]]; then
            run_case strata "$strata_bin" "$kv" "$context"
        fi
        run_case betelgeuse "$betelgeuse_bin" "$kv" "$context"
    done
done
