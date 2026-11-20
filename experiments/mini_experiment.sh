#!/usr/bin/env bash
# L3 mini experiment (plan 12.4): 2 sweep points x all schemes x 5 seeds on the small map,
# then the full analysis pipeline (tables, CIs, paired tests, figures) and a budget estimate.
#   experiments/mini_experiment.sh OUTDIR     (PARALLEL=4 by default; EXPS="E1 E6" to choose)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/experiments/env.sh"
OUT="${1:-$ROOT/results/mini}"; mkdir -p "$OUT"
MAN="$OUT/manifest.jsonl"
python3 "$ROOT/experiments/make_manifest.py" --profile mini --only ${EXPS:-E1} --out "$MAN.all" >/dev/null
# small map only for the local L3 run
python3 - "$MAN.all" "$MAN" "${MAPS:-grid}" <<'PY'
import json,sys
maps=set(sys.argv[3].split(","))
rows=[json.loads(l) for l in open(sys.argv[1])]
rows=[r for r in rows if r["map"] in maps]
with open(sys.argv[2],"w") as f:
    for i,r in enumerate(rows):
        r["idx"]=i; f.write(json.dumps(r)+"\n")
print(len(rows),"runs")
PY
export LITEOFF_SIDECAR_ARGS="${LITEOFF_SIDECAR_ARGS:---replay-db $OUT/replay.db --replay-mode rw}"
python3 "$ROOT/experiments/run_one.py" --manifest "$MAN" --results "$OUT" --all --parallel "${PARALLEL:-4}" || true
python3 "$ROOT/analysis/aggregate.py" "$OUT"
python3 "$ROOT/analysis/plots.py" "$OUT"
python3 "$ROOT/analysis/budget.py" "$OUT" --manifest "$ROOT/experiments/manifest_full.jsonl"
