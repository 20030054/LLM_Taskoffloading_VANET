"""Teacher labelling for distillation (plan 6.2, step 2).

The large teacher model (default qwen2.5:3b locally; a 7B-70B model on the HPC GPU nodes) reads the
same compact prompt plus privileged per-option completion estimates (offline only, --no-hints to
disable) and answers with grammar-constrained JSON {reason, target, split}. Decisions that agree
with the full-information oracle are kept, which gives the distilled training set (answer = teacher
decision, rationale stored in meta). Disagreeing samples are logged for analysis.

Writes:  <out>/distilled.jsonl  (agreeing samples, SFT format)
         <out>/teacher_all.jsonl (every sample with teacher output)
         <out>/teacher_stats.json (agreement rate, parse rate, latency)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .backends import OllamaBackend, ReplayStore
from .encoder import format_answer
from .policy_ref import estimate

TEACHER_SYSTEM = (
    "You are an expert controller for vehicular task offloading. Choose where the task runs so that it finishes "
    "before its deadline ('left'). Target 0 = this vehicle, 1..K = the listed candidates. A remote option only works "
    "if upload + queue + execution + result return fit in 'left' and the link lasts long enough (link_left). Prefer "
    "local when it meets the deadline comfortably. For splittable tasks, split = percent kept local (0 = send all). "
    "Answer in JSON: first a one-sentence reason, then target and split.")


def hints(ctx):
    """Privileged hints for the offline teacher only: estimated completion of every option (policy_ref)."""
    budget = float(ctx["task"]["budget_ms"])
    parts = []
    for slot in range(len(ctx.get("cands", [])) + 1):
        e = estimate(ctx, slot) * 1000
        name = "local" if slot == 0 else f"C{slot}"
        parts.append(f"{name}={'infeasible' if e == float('inf') else f'{e:.0f}ms'}")
    return f"ESTIMATES (budget {budget:.0f}ms): " + " ".join(parts)


def schema(k):
    return {"type": "object", "properties": {"reason": {"type": "string", "maxLength": 100},
                                             "target": {"type": "integer", "enum": list(range(k + 1))},
                                             "split": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["reason", "target", "split"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="SFT jsonl (from dataset.py), e.g. data/sft/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="qwen2.5:3b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-predict", type=int, default=72)
    ap.add_argument("--no-hints", action="store_true", help="teacher sees only the student prompt")
    ap.add_argument("--replay-db", default="", help="cache teacher answers (resume after interruption)")
    a = ap.parse_args(argv)
    be = OllamaBackend(num_predict=a.num_predict, logprobs=False)
    store = ReplayStore(a.replay_db) if a.replay_db else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exs = [json.loads(line) for line in open(a.input) if line.strip()]
    if a.limit:
        exs = exs[:a.limit]
    n = agree = parsed = 0
    lat = 0.0
    t0 = time.time()
    with open(out / "distilled.jsonl", "w") as fd, open(out / "teacher_all.jsonl", "w") as fa:
        for i, e in enumerate(exs):
            k = e["meta"]["k"]
            prompt = e["prompt"] if a.no_hints else e["prompt"] + "\n" + hints(e["meta"]["ctx"])
            v = store.get(a.teacher, TEACHER_SYSTEM, prompt) if store else None
            if v is None:
                r = be.chat(a.teacher, TEACHER_SYSTEM, prompt, num_predict=a.num_predict, fmt=schema(k))
                v = {"raw": r.raw, "wall_ms": r.wall_ms}
                if store:
                    store.put(a.teacher, TEACHER_SYSTEM, prompt, v)
            raw = v["raw"]
            lat += v["wall_ms"]
            try:
                j = json.loads(raw)
                slot, split, reason = int(j["target"]), int(j.get("split", 0)), str(j.get("reason", ""))
                ok = 0 <= slot <= k and 0 <= split <= 100
            except (ValueError, KeyError, TypeError):
                ok, slot, split, reason = False, -1, 0, ""
            n += 1
            parsed += ok
            is_agree = ok and slot == e["meta"]["slot"]
            agree += is_agree
            rationale = reason[:400]
            rec = dict(e)
            rec["meta"] = dict(e["meta"], teacher_raw=raw, teacher_slot=slot if ok else None, teacher_agree=bool(is_agree),
                               rationale=rationale)
            fa.write(json.dumps(rec) + "\n")
            if is_agree:
                rec2 = dict(rec)
                rec2["answer"] = format_answer(slot, split if "split=" in e["answer"] else 0, "split=" in e["answer"])
                fd.write(json.dumps(rec2) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(exs)} agree={agree / n:.2f} parse={parsed / n:.2f} {lat / n:.0f} ms/sample", flush=True)
    stats = {"teacher": a.teacher, "n": n, "parse_rate": parsed / max(1, n), "oracle_agreement": agree / max(1, n),
             "kept": agree, "mean_ms": lat / max(1, n), "elapsed_s": time.time() - t0}
    (out / "teacher_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
