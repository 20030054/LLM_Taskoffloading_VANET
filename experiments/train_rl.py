#!/usr/bin/env python3
"""Train the learning-based baselines (B1 DQN, B4 federated meta-DQN, B5 multi-agent AC).

Episodes run sequentially (weights persist between episodes in <rl_dir>/<scheme>.json) on training
seeds that are disjoint from the evaluation seeds (seeds_from in experiments.yaml).
    python experiments/train_rl.py --scheme b1 --episodes 30 --rl-dir models/rl --results results/rl_train
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheme", required=True, choices=["b1", "b4", "b5"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--warmup", type=float, default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--vehicles", type=float, default=None)
    ap.add_argument("--rl-dir", default=str(HERE.parent / "models" / "rl"))
    ap.add_argument("--results", default=str(HERE.parent / "results" / "rl_train"))
    ap.add_argument("--fresh", action="store_true", help="delete existing weights first")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(HERE / "experiments.yaml"))
    rc = cfg["rl_training"]
    mp = a.map or rc["map"]
    spec0 = dict(cfg["defaults"])
    spec0.update(cfg["maps"].get(mp, {}))
    w = Path(a.rl_dir) / f"{a.scheme}.json"
    if a.fresh and w.exists():
        w.unlink()
    log = []
    episodes = a.episodes or rc["episodes"]
    for ep in range(episodes):
        seed = rc["seeds_from"] + ep
        spec = dict(spec0, map=mp, scheme=a.scheme, seed=seed, duration=a.duration or rc["duration"], warmup=a.warmup or rc["warmup"],
                    tag=f"{mp}-rltrain-{a.scheme}-s{seed}",
                    params={"*.world.rlMode": "train", "*.world.rlWeights": str(w.resolve())})
        if a.vehicles:
            spec["vehicles"] = a.vehicles
        t0 = time.time()
        s = runner.run(spec, Path(a.results) / a.scheme / f"ep{ep:03d}", quiet=True)
        m = s.get("metrics", {})
        log.append({"episode": ep, "seed": seed, "ok": s["ok"], "tcr": m.get("tcr"), "meanDelay": m.get("meanDelay"), "wall_s": time.time() - t0})
        print(f"[{a.scheme}] episode {ep + 1}/{episodes} seed={seed} ok={s['ok']} TCR={m.get('tcr', float('nan')):.3f}", flush=True)
        if not s["ok"]:
            print(s.get("error_tail", "")[-1500:])
            sys.exit(1)
    (Path(a.rl_dir) / f"{a.scheme}_train_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
