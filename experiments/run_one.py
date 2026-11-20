#!/usr/bin/env python3
"""Run manifest entries by index (Slurm array task, or a local loop).

    python experiments/run_one.py --manifest M.jsonl --index 17 --results results/full
    python experiments/run_one.py --manifest M.jsonl --all --parallel 4 --results results/mini

Each run writes <results>/<outdir>/summary.json; existing successful runs are skipped (resume).
Environment variables in params (e.g. ${STUDENT_Q4}, ${RL_DIR}) are expanded from experiments/env.sh
defaults plus the job environment. Runs needing an LLM expect Ollama at $OLLAMA_HOST_URL (run_one.sh
starts a private server per job on HPC).
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner  # noqa: E402

ENV_DEFAULTS = {
    "STUDENT_Q4": "liteoff-student-q4", "STUDENT_F16": "liteoff-student-f16", "STUDENT_NODISTILL": "liteoff-student-nodistill-q4",
    "BASE_Q4": "qwen2.5:0.5b", "TEACHER": "qwen2.5:3b", "RL_DIR": str(HERE.parent / "models" / "rl"),
}


def expand(v):
    if isinstance(v, str):
        env = {**ENV_DEFAULTS, **os.environ}
        for k, val in env.items():
            v = v.replace("${" + k + "}", val)
    return v


def load(manifest):
    return [json.loads(line) for line in open(manifest) if line.strip()]


def run_entry(e, results, sidecar_args, keep=False):
    out = Path(results) / e["outdir"]
    sp = out / "summary.json"
    if sp.exists():
        try:
            if json.loads(sp.read_text()).get("ok"):
                return {"idx": e["idx"], "skipped": True, "ok": True}
        except json.JSONDecodeError:
            pass
    spec = dict(e["spec"])
    spec["params"] = {k: expand(v) for k, v in spec.get("params", {}).items()}
    sa = spec.get("sidecar_args", "") + " " + sidecar_args
    spec["sidecar_args"] = expand(sa).strip()
    if os.environ.get("LITEOFF_LATENCY_FILE"):
        spec["latency_file"] = os.environ["LITEOFF_LATENCY_FILE"]
    s = runner.run(spec, out, keep_scenario=keep)
    return {"idx": e["idx"], "ok": s["ok"], "wall_s": s["wall_s"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--index", type=int, nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1, help="with --index i: run i, i+stride, ... (pack several runs per job)")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--sidecar-args", default=os.environ.get("LITEOFF_SIDECAR_ARGS", ""))
    a = ap.parse_args()
    entries = load(a.manifest)
    if a.all:
        todo = entries
    else:
        idx = []
        for i in a.index or [int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))]:
            idx += [i + k * a.stride for k in range(a.count)]
        todo = [entries[i] for i in idx if i < len(entries)]
    t0 = time.time()
    res = []
    if a.parallel > 1:
        with cf.ThreadPoolExecutor(a.parallel) as ex:
            for r in ex.map(lambda e: run_entry(e, a.results, a.sidecar_args), todo):
                res.append(r)
    else:
        for e in todo:
            res.append(run_entry(e, a.results, a.sidecar_args))
    bad = [r["idx"] for r in res if not r["ok"]]
    print(f"done {len(res)} runs in {time.time() - t0:.0f}s, failed: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
