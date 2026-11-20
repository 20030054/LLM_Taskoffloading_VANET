#!/usr/bin/env bash
# One Slurm array task = one (or --count) manifest entries. Starts a private Ollama server for runs
# that need an LLM (unique port per job), works in node-local scratch, copies results back.
#   MANIFEST=... RESULTS=... sbatch --array=0-N%M experiments/slurm_array.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/experiments/env.sh"
IDX="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
COUNT="${RUNS_PER_TASK:-1}"
STRIDE="${ARRAY_STRIDE:-1}"
START=$((IDX * COUNT))
SCRATCH="${LITEOFF_SCRATCH:-${TMPDIR:-/tmp}}/liteoff_${SLURM_JOB_ID:-local}_${IDX}"
mkdir -p "$SCRATCH"
# does any of our entries need an LLM?
NEED_LLM=$(python3 - "$MANIFEST" "$START" "$COUNT" <<'PY'
import json,sys
m=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
s,c=int(sys.argv[2]),int(sys.argv[3])
print(int(any(e["spec"]["scheme"] in ("liteoff","b2","b3") for e in m[s:s+c])))
PY
)
OLLAMA_PID=""
if [ "$NEED_LLM" = "1" ]; then
  PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1])")
  export OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_HOST_URL="http://127.0.0.1:$PORT"
  export OLLAMA_LOG="$SCRATCH/ollama.log"
  "$ROOT/env/start_ollama.sh"
  OLLAMA_PID=$(pgrep -f "ollama serve" -n -u "$USER" || true)
fi
trap '[ -n "$OLLAMA_PID" ] && kill $OLLAMA_PID 2>/dev/null || true' EXIT
python3 "$ROOT/experiments/run_one.py" --manifest "$MANIFEST" --results "$SCRATCH/results" --index "$START" --count "$COUNT" --stride "$STRIDE" || STATUS=$?
mkdir -p "$RESULTS"
# copy only what the analysis needs
(cd "$SCRATCH/results" && find . \( -name summary.json -o -name results.sca -o -name spec.json -o -name sidecar_stats.json -o -name omnet.log \) -print0 | \
  tar --null -cf - -T - ) | (cd "$RESULTS" && tar -xf -)
rm -rf "$SCRATCH"
exit ${STATUS:-0}
