#!/usr/bin/env bash
# L2 functional test (plan 12.4): every scheme at tiny scale, determinism, sanity of baselines,
# live vs replay LLM agreement. Usage: experiments/functional_test.sh OUTDIR
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/experiments/env.sh"
OUT="${1:-$ROOT/results/functional}"; mkdir -p "$OUT"
R="python3 $ROOT/experiments/runner.py"
COMMON="--map grid --vehicles 40 --duration ${DUR:-150} --warmup 30"
FAIL=0
check() { if python3 -c "$1"; then echo "PASS: $2"; else echo "FAIL: $2"; FAIL=1; fi; }
m() { echo "json.load(open('$OUT/$1/summary.json'))['metrics']['$2']"; }

echo "== 1. all non-learning schemes, 3 seeds"
for s in local rsu random rule basic-rule oracle; do for seed in 1 2 3; do
  $R $COMMON --scheme $s --seed $seed --out "$OUT/$s/s$seed" >/dev/null 2>&1 &
done; wait; done
for s in local rsu random rule basic-rule oracle; do for seed in 1 2 3; do
  check "import json;assert json.load(open('$OUT/$s/s$seed/summary.json'))['ok']" "$s seed $seed runs and balances"
done; done
check "import json,statistics as st
tcr=lambda s: st.mean(json.load(open(f'$OUT/{s}/s{i}/summary.json'))['metrics']['tcr'] for i in (1,2,3))
assert tcr('oracle')>=tcr('rule')>tcr('random') and tcr('rule')>tcr('local'), {s:tcr(s) for s in ('oracle','rule','local','random')}" \
  "ordering oracle >= rule > local, rule > random"
check "import json
g=lambda s,i:json.load(open(f'$OUT/{s}/s{i}/summary.json'))['metrics']['tasksGenerated']
assert len({g(s,1) for s in ('local','rule','oracle','random')})==1" "identical workload across schemes (paired seeds)"

echo "== 2. determinism (same seed twice)"
$R $COMMON --scheme rule --seed 1 --out "$OUT/rule_rerun/s1" >/dev/null 2>&1
check "import json
a=json.load(open('$OUT/rule/s1/summary.json'))['metrics'];b=json.load(open('$OUT/rule_rerun/s1/summary.json'))['metrics']
assert a==b, [k for k in a if a[k]!=b.get(k)]" "rule seed 1 bit-identical scalars"

echo "== 3. local-only degrades under load, RSU-only fails in coverage gaps"
$R $COMMON --scheme local --seed 1 --task-rate 3 --out "$OUT/local_load3/s1" >/dev/null 2>&1 &
$R --map highway --vehicles 80 --vmax 30 --duration ${DUR:-150} --warmup 30 --rsu-spacing 1500 --scheme rsu --seed 1 --out "$OUT/hw_rsu/s1" >/dev/null 2>&1 &
$R --map highway --vehicles 80 --vmax 30 --duration ${DUR:-150} --warmup 30 --rsu-spacing 1500 --scheme rule --seed 1 --out "$OUT/hw_rule/s1" >/dev/null 2>&1 &
wait
check "import json;assert $(m local_load3/s1 tcr) < $(m local/s1 tcr) - 0.05" "local-only TCR drops at 3 tasks/s"
check "import json;assert $(m hw_rsu/s1 tcr) < $(m hw_rule/s1 tcr)" "RSU-only below rule on sparse-RSU highway"

echo "== 4. learning baselines (1 short training episode + eval)"
RLD="$OUT/rl"; mkdir -p "$RLD"
for s in b1 b4 b5; do
  ( $R $COMMON --scheme $s --seed 901 --set "*.world.rlMode=train" --set "*.world.rlWeights=$RLD/$s.json" --out "$OUT/${s}_train/s901" >/dev/null 2>&1 && \
    $R $COMMON --scheme $s --seed 1 --set "*.world.rlWeights=$RLD/$s.json" --out "$OUT/$s/s1" >/dev/null 2>&1 ) &
done; wait
for s in b1 b4 b5; do check "import json;assert json.load(open('$OUT/$s/s1/summary.json'))['ok']" "$s trains and evaluates"; done

if [ "${SKIP_LLM:-0}" != "1" ] && curl -sf "$OLLAMA_HOST_URL/api/version" >/dev/null; then
  echo "== 5. LLM schemes (live Ollama), then replay of the same run"
  DB="$OUT/replay.db"; rm -f "$DB"
  SC="--replay-db $DB --replay-mode rw --log-decisions"
  LDUR="--map grid --vehicles 30 --duration ${LLM_DUR:-80} --warmup 20"
  $R $LDUR --scheme liteoff --seed 1 --sidecar-args "$SC" --out "$OUT/liteoff_live/s1" >/dev/null 2>&1
  check "import json;s=json.load(open('$OUT/liteoff_live/s1/summary.json'));assert s['ok'] and s['metrics']['llmCalls']>0, s.get('error_tail','')[-500:]" "liteoff (hybrid) live run, LLM called"
  $R $LDUR --scheme liteoff --seed 1 --sidecar-args "--replay-db $DB --backend replay --log-decisions" --out "$OUT/liteoff_replay/s1" >/dev/null 2>&1
  check "import json
a=[json.loads(l) for l in open('$OUT/liteoff_live/s1/decisions.jsonl')];b=[json.loads(l) for l in open('$OUT/liteoff_replay/s1/decisions.jsonl')]
n=min(len(a),len(b));agree=sum(x['slot']==y['slot'] for x,y in zip(a,b))/max(1,n)
print(f'   live/replay decisions: {len(a)}/{len(b)}, agreement {agree:.3f}')
assert n>0 and agree==1.0 and len(a)==len(b)" "replay reproduces live decisions exactly"
  check "import json
a=json.load(open('$OUT/liteoff_live/s1/summary.json'))['metrics'];b=json.load(open('$OUT/liteoff_replay/s1/summary.json'))['metrics']
d=[k for k in a if k not in ('meanLlmWallMs',) and a[k]!=b[k]];assert not d, d" "replay run has identical metrics"
  for s in b2 b3; do
    $R $LDUR --scheme $s --seed 1 --sidecar-args "$SC" --out "$OUT/$s/s1" >/dev/null 2>&1
    check "import json;s=json.load(open('$OUT/$s/s1/summary.json'));assert s['ok'], s.get('error_tail','')[-500:]" "$s (RSU-hosted LLM) runs"
  done
else
  echo "== 5. skipped (no Ollama at $OLLAMA_HOST_URL or SKIP_LLM=1)"
fi
echo; [ $FAIL = 0 ] && echo "L2 FUNCTIONAL: ALL PASS" || echo "L2 FUNCTIONAL: FAILURES"
exit $FAIL
