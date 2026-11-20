#!/usr/bin/env python3
"""Aggregate run results: per-run table, mean +- 95% CI per (experiment, point, map, scheme),
paired tests of LiteOff vs every other scheme (identical seeds, Holm-corrected), run budget.

    python analysis/aggregate.py results/mini            -> results/mini/analysis/{runs,agg,tests}.csv, budget.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import holm, mean_ci, paired_test  # noqa: E402

METRICS = ["tcr", "tcrSafety", "tcrPerception", "tcrInfotainment", "meanDelay", "meanDecisionDelay", "meanTxDelay",
           "meanQueueDelay", "meanExecDelay", "meanResultDelay", "packetDropRatio", "overheadRatio", "offloadSuccessRate",
           "meanDecisionLatency", "energyPerTaskJ", "reoffloadRate", "oracleAgreement", "jainServerLoad", "llmCalls",
           "fallbackTotal", "malformedRate", "pathShare_local", "pathShare_rule", "pathShare_cache", "pathShare_llm",
           "pathShare_rsu-llm", "pathShare_fallback", "bytes_model", "modelUpdatesCompleted", "lateRatio", "dropRatio"]
TESTED = ["tcr", "meanDelay", "packetDropRatio", "overheadRatio", "energyPerTaskJ"]


def collect(root: Path):
    rows = []
    for sp in root.rglob("summary.json"):
        rel = sp.parent.relative_to(root).parts
        try:
            s = json.loads(sp.read_text())
        except json.JSONDecodeError:
            continue
        if len(rel) >= 5:
            exp, point, mp, scheme, seed = rel[-5:]
        else:
            exp, point, mp, scheme, seed = "adhoc", "default", s["spec"]["map"], s["spec"]["scheme"], f"s{s['spec']['seed']}"
        r = {"exp": exp, "point": point, "map": mp, "scheme": scheme, "seed": int(str(seed).lstrip("s")), "ok": s.get("ok", False),
             "wall_s": s.get("wall_s"), "max_rss_mb": s.get("max_rss_mb"), "vehicles": s["spec"].get("vehicles")}
        for m in METRICS:
            r[m] = s.get("metrics", {}).get(m, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--reference", default="liteoff", help="scheme compared against all others")
    a = ap.parse_args()
    root = Path(a.results)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = collect(root)
    if df.empty:
        sys.exit(f"no summary.json under {root}")
    df.to_csv(out / "runs.csv", index=False)
    good = df[df.ok]
    agg = []
    for key, g in good.groupby(["exp", "point", "map", "scheme"]):
        r = dict(zip(["exp", "point", "map", "scheme"], key), n=len(g))
        for m in METRICS:
            mu, lo, hi = mean_ci(g[m].tolist())
            r[m], r[m + "_lo"], r[m + "_hi"] = mu, lo, hi
        agg.append(r)
    pd.DataFrame(agg).to_csv(out / "agg.csv", index=False)
    tests = []
    for key, g in good.groupby(["exp", "point", "map"]):
        ref = g[g.scheme == a.reference].set_index("seed")
        if ref.empty:
            continue
        for other, go in g[g.scheme != a.reference].groupby("scheme"):
            go = go.set_index("seed")
            seeds = ref.index.intersection(go.index)
            for m in TESTED:
                t = paired_test(ref.loc[seeds, m].values, go.loc[seeds, m].values)
                tests.append(dict(zip(["exp", "point", "map"], key), reference=a.reference, other=other, metric=m, **t))
    td = pd.DataFrame(tests)
    if not td.empty:
        for m, idx in td.groupby("metric").groups.items():
            td.loc[idx, "p_holm"] = holm(td.loc[idx, "p_t"].fillna(1.0).tolist())
    td.to_csv(out / "tests.csv", index=False)
    budget = {"runs": int(len(df)), "ok": int(df.ok.sum()), "failed": df[~df.ok][["exp", "point", "map", "scheme", "seed"]].to_dict("records"),
              "wall_s_mean": float(df.wall_s.mean()), "wall_s_p95": float(df.wall_s.quantile(0.95)),
              "max_rss_mb": float(df.max_rss_mb.max()) if df.max_rss_mb.notna().any() else None,
              "wall_s_by_scheme": df.groupby("scheme").wall_s.mean().round(1).to_dict(),
              "wall_s_per_vehicle_simsec": None}
    (out / "budget.json").write_text(json.dumps(budget, indent=1, default=str))
    print(f"{len(df)} runs ({budget['ok']} ok) -> {out}")
    show = pd.DataFrame(agg)
    if not show.empty:
        cols = ["exp", "point", "map", "scheme", "n", "tcr", "meanDelay", "packetDropRatio", "overheadRatio", "oracleAgreement"]
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(show[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
