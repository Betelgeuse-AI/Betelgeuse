#!/usr/bin/env bash
# Reproduce the restored direct-engine bel-server measurement.
# Example: ./run.sh 16896 16384 18080
set -euo pipefail

ctx=${1:-16896}
prompt=${2:-16384}
port=${3:-18080}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)
ids_file="$HOME/Docs/strata-inkling2-rtx3060/evidence/prefill-prefix-16384.ids"
prompt_ids="$here/raw/server-prompt-${prompt}.ids"

python3 - "$ids_file" "$prompt" > "$prompt_ids" <<'PY'
import pathlib, sys
ids = [int(x) for x in pathlib.Path(sys.argv[1]).read_text().replace('\n','').split(',') if x]
n = int(sys.argv[2])
print(','.join(map(str, (ids * 2)[:n])))
PY

STRATA_NUMA_POOL=0 INK2_CPU_ACT=int8 INK2_CPU_OPT=29 INK2_OFF_MARGIN_MIB=64 INK2_MX_POOL_FRACTION=0.08 \
    nohup "$root/build/bin/bel-server" -m "$HOME/Models/Inkling-2-Mini" -c "$ctx" -t 5 \
    --host 127.0.0.1 --port "$port" > "$here/raw/server-${ctx}.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

timeout 300 bash -c "until curl -sf http://127.0.0.1:${port}/health >/dev/null; do sleep 1; done"
grep -E "loaded direct Inkling|offload:" "$here/raw/server-${ctx}.log" | tail -3

python3 - "$port" "$prompt_ids" <<'PY'
import json, pathlib, sys, urllib.request
port = int(sys.argv[1]); ids = [int(x) for x in pathlib.Path(sys.argv[2]).read_text().split(',') if x]
base = f'http://127.0.0.1:{port}'
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
text = post('/detokenize', {'tokens': ids})['content']
print('prompt tokens', len(post('/tokenize', {'content': text})['tokens']))
for run in (1, 2):
    body = {'prompt': text, 'max_tokens': 512, 'temperature': 0.0,
            'top_k': 0, 'top_p': 1.0, 'min_p': 0.0, 'stream': False}
    t = post('/v1/completions', body)['timings']
    print('run', run, 'prompt', t['prompt_n'], 'decode', t['predicted_n'],
          'pp/s', round(t['prompt_per_second'], 1), 'tok/s', round(t['predicted_per_second'], 2),
          'total s', round((t['prompt_ms'] + t['predicted_ms']) / 1000.0, 2))
PY
