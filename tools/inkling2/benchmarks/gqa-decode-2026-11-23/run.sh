#!/usr/bin/env bash
set -euo pipefail

binary=${BETELGEUSE_BIN:-build/bin/bel-inkling2}
model=${INK2_MODEL:-<checkpoint-dir>}
routes=${INK2_ROUTE_PROFILE:-<evidence-dir>/inkling2-routes-profiled.freq}
prefix_source=${INK2_PREFIX_SOURCE:-<evidence-dir>/prefill-prefix-16384.ids}
data_root=${INK2_DATA_ROOT:-$HOME/fluxaeros-data/inkling2-dspark-2026-11-23}
output_root=${INK2_OUTPUT_ROOT:-$data_root/gqa-decode-2026-11-23/raw}
work_dir=${INK2_PROMPT_ROOT:-$data_root/gqa-decode-2026-11-23/prompts}
run_256k=${RUN_256K:-1}

mkdir -p "$output_root" "$work_dir"

python3 - "$prefix_source" "$work_dir" <<'PY'
from pathlib import Path
import sys

ids = [item for item in Path(sys.argv[1]).read_text().strip().replace("\n", ",").split(",") if item]
out = Path(sys.argv[2])
for context in (8192, 16384, 65536, 262144):
    prompt_tokens = context - 512
    repeated = (ids * ((prompt_tokens + len(ids) - 1) // len(ids)))[:prompt_tokens]
    (out / f"prompt-{context}.ids").write_text(",".join(repeated))
PY

run_case() {
    local label=$1
    local kv=$2
    local context=$3
    local prompt="$work_dir/prompt-$context.ids"
    local log="$output_root/${label}-${kv}-${context}.log"

    printf 'Running %s kv=%s context=%s\n' "$label" "$kv" "$context"
    env STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 INK2_MX_POOL_FRACTION=0.08 \
        "$binary" --model "$model" --max-context "$context" --chunk 4096 --kv "$kv" --tc 1 \
        --vram-reserve-mib 0 --cpu-threads 5 --kv-grow 1 --kv-grow-step 4096 \
        --tokens-file "$prompt" --n 512 --expert-frequencies "$routes" >"$log" 2>&1
    printf 'Saved %s\n' "$log"
}

for context in 8192 16384; do
    for kv in k8v4 i8r; do
        run_case gqa "$kv" "$context"
    done
done
run_case gqa k8v4 65536
if [[ "$run_256k" == 1 ]]; then
    run_case gqa k8v4 262144
fi
