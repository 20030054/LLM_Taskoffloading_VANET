"""Offline evaluation of decision models on held-out contexts (E5 quality vs size; L4 check).

    python -m liteoff_llm.evaluate --data data/sft/test.jsonl --models liteoff-student-q4 qwen2.5:0.5b qwen2.5:3b

For each model: oracle agreement (exact target), agreement on target kind, parse rate, feasibility of
the chosen option (estimated completion <= budget using the logged context), mean/p95 wall latency,
prompt/output tokens, mean confidence. Also evaluates the rule policy as a reference.
Writes a JSON report (and prints a table).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .backends import OllamaBackend, ReplayStore
from .backends.ollama import LLMResult
from .encoder import SYSTEM_PROMPT, parse_answer
from .policy_ref import estimate, rule_policy


def evaluate_model(be, model, exs, store=None):
    rows = []
    for ex in exs:
        res = None
        if store:
            v = store.get(model, SYSTEM_PROMPT, ex["prompt"])
            res = LLMResult.from_dict(v) if v else None
        if res is None:
            res = be.chat(model, ex["system"], ex["prompt"])
            if store:
                store.put(model, SYSTEM_PROMPT, ex["prompt"], res.to_dict())
        ok, slot, split = parse_answer(res.raw, k=ex["meta"]["k"])
        ctx = ex["meta"]["ctx"]
        feas = ok and estimate(ctx, slot, split) <= float(ctx["task"]["budget_ms"]) / 1000.0
        kind = None
        if ok:
            kind = "local" if slot == 0 else ctx["cands"][slot - 1]["kind"]
        rows.append({"ok": ok, "hit": ok and slot == ex["meta"]["slot"], "kind_hit": kind == ex["meta"]["kind"], "feasible": feas,
                     "wall_ms": res.wall_ms, "n_in": res.n_in, "n_out": res.n_out, "conf": res.conf})
    return summarize(rows)


def summarize(rows):
    def mean(k):
        v = [float(r[k]) for r in rows if r.get(k) is not None]
        return float(np.mean(v)) if v else None
    w = [r["wall_ms"] for r in rows if r.get("wall_ms") is not None]
    return {"n": len(rows), "oracle_agreement": mean("hit"), "kind_agreement": mean("kind_hit"), "parse_rate": mean("ok"),
            "feasible_rate": mean("feasible"), "mean_ms": float(np.mean(w)) if w else None,
            "p95_ms": float(np.percentile(w, 95)) if w else None, "mean_n_in": mean("n_in"), "mean_n_out": mean("n_out"),
            "mean_conf": mean("conf")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="")
    ap.add_argument("--replay-db", default="")
    ap.add_argument("--num-thread", type=int, default=None)
    a = ap.parse_args(argv)
    exs = [json.loads(line) for line in open(a.data) if line.strip()][:a.limit]
    be = OllamaBackend(num_thread=a.num_thread)
    store = ReplayStore(a.replay_db) if a.replay_db else None
    rep = {"data": a.data, "n": len(exs), "models": {}}
    rule_rows = []
    for ex in exs:
        ctx = ex["meta"]["ctx"]
        s = rule_policy(ctx)
        kind = "local" if s == 0 else ctx["cands"][s - 1]["kind"]
        rule_rows.append({"ok": True, "hit": s == ex["meta"]["slot"], "kind_hit": kind == ex["meta"]["kind"],
                          "feasible": estimate(ctx, s, 0) <= float(ctx["task"]["budget_ms"]) / 1000.0, "wall_ms": None,
                          "n_in": None, "n_out": None, "conf": None})
    rep["models"]["rule"] = summarize(rule_rows)
    for m in a.models:
        t0 = time.time()
        rep["models"][m] = evaluate_model(be, m, exs, store)
        rep["models"][m]["eval_s"] = time.time() - t0
        be.unload(m)
    print(f"{'model':32s} {'agree':>6s} {'kind':>6s} {'parse':>6s} {'feas':>6s} {'ms':>7s} {'p95':>7s} {'n_in':>5s}")
    for m, r in rep["models"].items():
        f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
        print(f"{m:32s} {f(r['oracle_agreement']):>6s} {f(r['kind_agreement']):>6s} {f(r['parse_rate']):>6s} "
              f"{f(r['feasible_rate']):>6s} {f(r['mean_ms'], 0):>7s} {f(r['p95_ms'], 0):>7s} {f(r['mean_n_in'], 0):>5s}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
