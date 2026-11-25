#!/usr/bin/env bash
set -euo pipefail

chunk=${1:?pass chunk size}
port=${2:-18090}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)
printf '%s\n' "$$" > "$here/raw/server-$chunk.pid"

exec env \
    STRATA_NUMA_POOL=0 \
    INK2_CPU_ACT=int8 \
    INK2_CPU_OPT=29 \
    INK2_OFF_MARGIN_MIB=64 \
    INK2_MX_POOL_FRACTION=0.08 \
    INK2_PIN_HOST_MIB=512 \
    INK2_ALLOW_TUNING_OVERRIDE=1 \
    INK2_CHUNK="$chunk" \
    INK2_RING_EXTRA=8192 \
    INK2_MISS_DIAG=1 \
    "$root/build-inkling-probe/betelgeuse-build/bin/bel-server" \
    -m /home/zyxciss/Models/Inkling-2-Mini \
    -c 32768 -t 5 --host 127.0.0.1 --port "$port" \
    --cache-type-k q8_0 --cache-type-v q4_0 -fa on
