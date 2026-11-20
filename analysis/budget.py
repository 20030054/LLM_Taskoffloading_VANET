#!/usr/bin/env python3
"""HPC budget estimate from measured local runs (plan 12.5: replace the 30-minute assumption).

Wall time per run is modelled as  wall = c_scheme * vehicles^p * duration  (fitted on the runs found
under RESULTS; p fitted globally, c per scheme family), then applied to every entry of the full
manifest. Reports CPU-hours, the largest single run, peak memory and the output size per run.

    python analysis/budget.py results/mini --manifest experiments/manifest_full.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np

FAMILY = {"liteoff": "llm", "b2": "llm", "b3": "llm", "b1": "rl", "b4": "rl", "b5": "rl"}


def fam(scheme):
    return FAMILY.get(scheme, "plain")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cpus-per-run", type=float, default=1.0)
    a = ap.parse_args()
    root = Path(a.results)
    runs = []
    for sp in root.rglob("summary.json"):
        s = json.loads(sp.read_text())
        if not s.get("ok"):
            continue
        sz = sum(f.stat().st_size for f in sp.parent.glob("*") if f.is_file() and f.name in ("summary.json", "results.sca", "spec.json", "sidecar_stats.json"))
        runs.append({"scheme": s["spec"]["scheme"], "veh": float(s["spec"]["vehicles"]), "dur": float(s["spec"]["duration"]),
                     "wall": float(s["wall_s"]), "rss": s.get("max_rss_mb") or 0, "bytes": sz})
    if len(runs) < 3:
        raise SystemExit("need >= 3 successful runs to fit the budget model")
    X = np.log([r["veh"] for r in runs])
    y = np.log([r["wall"] / r["dur"] for r in runs])
    fams = sorted({fam(r["scheme"]) for r in runs})
    # log(wall/dur) = log c_fam + p log veh
    A = np.column_stack([X] + [[1.0 if fam(r["scheme"]) == f else 0.0 for r in runs] for f in fams])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    p = float(np.clip(coef[0], 0.5, 2.5)) if len(set(X)) > 1 else 1.3
    c = {f: float(np.exp(np.mean([y[i] - p * X[i] for i, r in enumerate(runs) if fam(r["scheme"]) == f]))) for f in fams}
    c_default = max(c.values())
    man = [json.loads(line) for line in open(a.manifest) if line.strip()]
    tot, worst, by_exp = 0.0, 0.0, {}
    for e in man:
        sp = e["spec"]
        w = c.get(fam(sp["scheme"]), c_default) * float(sp["vehicles"]) ** p * float(sp["duration"])
        tot += w
        worst = max(worst, w)
        by_exp[e["exp"]] = by_exp.get(e["exp"], 0) + w
    rep = {"fitted_runs": len(runs), "exponent_vehicles": p, "c_per_family": c,
           "manifest_runs": len(man), "cpu_hours": tot * a.cpus_per_run / 3600, "by_experiment_cpu_hours": {k: v / 3600 for k, v in by_exp.items()},
           "max_single_run_hours": worst / 3600, "suggested_walltime_h": round(max(1.0, 2.0 * worst / 3600), 1),
           "peak_rss_mb_local": max(r["rss"] for r in runs), "mean_output_kb_per_run": float(np.mean([r["bytes"] for r in runs]) / 1024),
           "total_output_gb": float(np.mean([r["bytes"] for r in runs]) * len(man) / 1e9),
           "note": "LLM runs: wall time includes live Ollama inference on the local CPU; with replay hits or faster nodes it drops."}
    out = root / "analysis" / "budget_estimate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
