#!/usr/bin/env python3
"""Measure LLM inference latency per model and fit the simulator's latency model.

For every model, prompts of increasing length are built with the real context encoder (K = 0..5
candidates, feature families from basic to full, plus padded long prompts). Each prompt is sent
`--reps` times (first call per model is a warm-up and is discarded). The inference time is
`total_duration - load_duration` reported by Ollama. A linear model is fitted:

        t_ms = a + b * (n_in - prefix_tok) + c * n_out

where n_in - prefix_tok are the prompt tokens actually evaluated (the system prompt / chat template
prefix stays in the KV cache between decisions, as it would on the vehicle). Written to
latency_model.csv (columns: model,a_ms,b_ms_per_in,c_ms_per_out,power_w,mem_mb,prefix_tok), which
the OMNeT++ LatencyModel reads. A JSON report with raw samples, R^2 and hardware info is also saved.

Power: RAPL usually needs root, so `power_w` defaults to an edge-board figure (--power-w, 15 W) and
is only used to convert modelled inference time into energy. Re-run this script on the target
hardware (HPC node, Jetson-class board) to obtain hardware-specific latency models (experiment E11).
"""
import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "llm"))
from liteoff_llm.backends import OllamaBackend  # noqa: E402
from liteoff_llm.encoder import SYSTEM_PROMPT, encode  # noqa: E402
from liteoff_llm.synthetic import random_context  # noqa: E402

FAMILY_SETS = ["basic", "mobility,link", "mobility,link,resources,task", "mobility,link,resources,task,trust,road"]


def cpu_info():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def profile_model(be: OllamaBackend, model: str, reps: int, rng: random.Random, n_prompts: int):
    samples = []
    be.chat(model, SYSTEM_PROMPT, "warm up", num_predict=4)  # load + warm-up
    prompts = []
    for i in range(n_prompts):
        k = i % 6
        fam = FAMILY_SETS[(i // 6) % len(FAMILY_SETS)]
        p = encode(random_context(rng, k), fam)
        if i % 7 == 6:  # a few long prompts to pin the slope
            p = p + "\nNOTE: " + " ".join(f"obs{j}=ok" for j in range(rng.randint(40, 160)))
        prompts.append(p)
    for p in prompts:
        for _ in range(reps):
            t0 = time.perf_counter()
            r = be.chat(model, SYSTEM_PROMPT, p)
            wall = (time.perf_counter() - t0) * 1000
            samples.append({"n_in": r.n_in, "n_eval": r.extra["n_eval"], "n_cached": r.extra["n_cached"], "rep": _,
                            "n_out": r.n_out, "infer_ms": r.prompt_eval_ms + r.eval_ms, "wall_ms": wall,
                            "prompt_eval_ms": r.prompt_eval_ms, "eval_ms": r.eval_ms})
    # fit on the tokens that were actually evaluated (Ollama re-uses the KV cache of a shared prefix)
    X = np.array([[1.0, s["n_eval"], s["n_out"]] for s in samples])
    y = np.array([s["wall_ms"] for s in samples])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum())
    a, b, c = (max(0.0, float(v)) for v in coef)
    # tokens that stay cached between *different* decisions (system prompt + template + common header)
    first = [s["n_cached"] for s in samples if s["rep"] == 0]
    prefix = int(np.median(first)) if first else 0
    return {"prefix_tok": prefix, "a_ms": a, "b_ms_per_in": b, "c_ms_per_out": c, "r2": float(r2), "n": len(samples),
            "mean_n_in": float(X[:, 1].mean()), "mean_ms": float(y.mean()), "p95_ms": float(np.percentile(y, 95)), "samples": samples}


def model_mem_mb(be, model):
    try:
        import urllib.request
        d = json.loads(urllib.request.urlopen(be.url + "/api/ps", timeout=5).read())
        for m in d.get("models", []):
            if m["name"] == model or m["model"] == model:
                return m.get("size", 0) / 1e6
    except Exception:
        pass
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="ollama tags; use name=tag to store under another name")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--num-thread", type=int, default=None)
    ap.add_argument("--power-w", type=float, default=15.0)
    ap.add_argument("--out", default=str(REPO / "profiling" / "latency_model.csv"))
    ap.add_argument("--report", default=None)
    ap.add_argument("--merge", action="store_true", help="keep rows of models not profiled now")
    a = ap.parse_args()
    be = OllamaBackend(num_thread=a.num_thread, logprobs=False)
    rng = random.Random(7)
    out = Path(a.out)
    rows = {}
    if a.merge and out.exists():
        for line in out.read_text().splitlines():
            if line and not line.startswith(("#", "model")):
                rows[line.split(",")[0]] = line
    report = {"host": platform.node(), "cpu": cpu_info(), "threads": a.num_thread, "time": time.strftime("%F %T"), "models": {}}
    for spec in a.models:
        name, tag = spec.split("=", 1) if "=" in spec else (spec, spec)
        print(f"profiling {tag} ...", flush=True)
        r = profile_model(be, tag, a.reps, rng, a.prompts)
        r["mem_mb"] = model_mem_mb(be, tag)
        report["models"][name] = r
        rows[name] = f"{name},{r['a_ms']:.3f},{r['b_ms_per_in']:.5f},{r['c_ms_per_out']:.3f},{a.power_w},{r['mem_mb']:.0f},{r['prefix_tok']}"
        print(f"  {name}: t = {r['a_ms']:.1f} + {r['b_ms_per_in']:.3f}*(n_in-{r['prefix_tok']}) + {r['c_ms_per_out']:.2f}*n_out ms "
              f"(R2={r['r2']:.3f}, mean {r['mean_ms']:.0f} ms @ {r['mean_n_in']:.0f} tok, p95 {r['p95_ms']:.0f} ms, mem {r['mem_mb']:.0f} MB)")
        be.unload(tag)
    # tiny MLP policy used by the RL baselines (measured in numpy)
    if "rl-mlp" not in rows:
        sys.path.insert(0, str(REPO / "llm"))
        from liteoff_llm.rl import MLP
        mlp = MLP([69, 64, 64, 6], np.random.default_rng(0))
        x = np.zeros((1, 69), np.float32)
        t0 = time.perf_counter()
        for _ in range(2000):
            mlp.forward(x)
        ms = (time.perf_counter() - t0) / 2000 * 1000
        rows["rl-mlp"] = f"rl-mlp,{ms:.4f},0,0,{a.power_w},1,0"
    hdr = f"# LiteOff latency model, host={report['host']} cpu={report['cpu']} threads={a.num_thread} {report['time']}\n"
    out.write_text(hdr + "model,a_ms,b_ms_per_in,c_ms_per_out,power_w,mem_mb,prefix_tok\n" + "\n".join(rows[k] for k in sorted(rows)) + "\n")
    rp = Path(a.report) if a.report else out.with_suffix(".json")
    old = json.loads(rp.read_text()) if (a.merge and rp.exists()) else {"models": {}}
    old.update({k: v for k, v in report.items() if k != "models"})
    old.setdefault("models", {}).update(report["models"])
    rp.write_text(json.dumps(old, indent=1))
    print(f"wrote {out} and {rp}")


if __name__ == "__main__":
    main()
