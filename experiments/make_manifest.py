#!/usr/bin/env python3
"""Expand experiments.yaml into a run manifest (one JSON line per independent run).

    python experiments/make_manifest.py --profile full --out experiments/manifest_full.jsonl [--only E1 E2]

Multi-key sweeps vary one factor at a time around the defaults. The array index of a Slurm job
array maps 1:1 to the manifest line number (0-based).
"""
import argparse
import collections
import itertools
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def expand(cfg, profile, only=None):
    prof = cfg["profiles"][profile]
    seeds = prof["seeds"]
    limit = prof.get("sweep_limit")
    rows = []
    for exp, e in cfg["experiments"].items():
        if only and exp not in only:
            continue
        schemes = cfg["groups"][e["schemes"]] if isinstance(e["schemes"], str) else e["schemes"]
        points = [("default", {})]
        if e.get("sweep"):
            points = []
            for key, vals in e["sweep"].items():
                for v in (vals[:limit] if limit else vals):
                    points.append((f"{key}={v}", {key: v}))
        for (pname, pt), mp, sname, seed in itertools.product(points, e["maps"], schemes, seeds):
            spec = dict(cfg["defaults"])
            spec.update(cfg.get("maps", {}).get(mp, {}))
            spec["map"] = mp
            sch = cfg["schemes"][sname]
            spec["scheme"] = sch["scheme"]
            params = dict(sch.get("params", {}))
            for k, v in pt.items():
                if k == "vehicles_per_unit":
                    spec["vehicles"] = round(v * cfg["density_to_vehicles"][mp], 1)
                elif k in ("vehicles", "vmax", "task_rate", "rsu_spacing"):
                    spec[k] = v
                else:
                    params[k] = v
            spec.update(duration=prof["duration"], warmup=prof["warmup"], seed=seed, params=params,
                        tag=f"{mp}-{exp}-{sname}-{pname}-s{seed}")
            rows.append({"exp": exp, "point": pname, "map": mp, "scheme_name": sname, "seed": seed, "spec": spec,
                         "outdir": f"{exp}/{pname.replace('*.', '').replace('**.', '').replace('/', '_')}/{mp}/{sname}/s{seed}"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "experiments.yaml"))
    ap.add_argument("--profile", default="mini")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--schemes", nargs="*", help="restrict to these scheme names")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    rows = expand(cfg, a.profile, a.only)
    if a.schemes:
        rows = [r for r in rows if r["scheme_name"] in a.schemes]
    # deduplicate identical specs across experiments (e.g. default points)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            r["idx"] = i
            f.write(json.dumps(r) + "\n")
    c = collections.Counter(r["exp"] for r in rows)
    print(f"{len(rows)} runs -> {a.out}")
    for k, v in c.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
