"""Build supervised (SFT) datasets from simulator decision logs.

Input: dataset.jsonl files written by runs with `--dataset` (any scheme; `explore` gives the most
diverse states). Each record holds the context JSON, the full-information oracle label, the action
actually taken and its outcome.

Output: JSONL with {"system","prompt","answer","meta"} where answer = oracle label, split into
train / val / test **by scenario (map) and seed** so no run contributes to two splits:
    --train-maps grid,highway --test-maps spider  --val-frac 0.15 (of train seeds)

Gate filter (default on): only keep contexts that would reach the LLM in the simulator (local not
feasible with margin, >=2 candidates, task not tiny), so the student trains on the cases it serves.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
from pathlib import Path

from .encoder import SYSTEM_PROMPT, encode, format_answer


def gate_passes(ctx, local_margin=0.7, tiny_cycles=5e7):
    t, s = ctx["task"], ctx["self"]
    if float(s.get("local_est_ms", 0)) <= local_margin * float(t["budget_ms"]):
        return False
    if len(ctx.get("cands", [])) <= 1:
        return False
    if float(t["cycles"]) < tiny_cycles:
        return False
    return True


def load_records(patterns):
    files = []
    for p in patterns:
        files += glob.glob(p, recursive=True)
    for f in sorted(set(files)):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def scenario_of(rec):
    tag = rec.get("scenario", "")
    m = re.match(r"(?P<map>[a-z]+)-.*?s(?P<seed>\d+)$", tag)
    if m:
        return m.group("map"), int(m.group("seed"))
    return tag.split("-")[0] or "unknown", 0


def to_example(rec, families=None, label="oracle"):
    ctx = rec["ctx"]
    if label == "oracle":
        slot, split = int(rec["oracle"]["slot"]), int(rec["oracle"]["split"])
    else:  # outcome: keep the taken action only if it finished in time
        if rec["outcome"]["status"] != "in_time":
            return None
        slot, split = int(rec["chosen"]["slot"]), int(rec["chosen"]["split"])
    if not ctx["task"].get("split"):
        split = 0
    mp, seed = scenario_of(rec)
    return {"system": SYSTEM_PROMPT, "prompt": encode(ctx, families), "answer": format_answer(slot, split, bool(ctx["task"].get("split"))),
            "meta": {"map": mp, "seed": seed, "slot": slot, "split": split, "k": len(ctx.get("cands", [])),
                     "kind": "local" if slot == 0 else ctx["cands"][slot - 1]["kind"], "cls": ctx["task"]["cls"],
                     "outcome": rec["outcome"]["status"], "ctx": ctx}}


def balance(examples, rng, max_ratio=2.0):
    """Down-sample over-represented target kinds to at most max_ratio x the median kind count."""
    by = collections.defaultdict(list)
    for e in examples:
        by[e["meta"]["kind"]].append(e)
    if not by:
        return examples
    counts = sorted(len(v) for v in by.values())
    cap = int(max_ratio * counts[len(counts) // 2]) or 1
    out = []
    for v in by.values():
        rng.shuffle(v)
        out += v[:cap]
    rng.shuffle(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="dataset.jsonl files / globs")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--train-maps", default="grid,highway")
    ap.add_argument("--test-maps", default="spider")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--families", default=None, help="override context families (default: as logged)")
    ap.add_argument("--label", choices=["oracle", "outcome"], default="oracle")
    ap.add_argument("--no-gate-filter", action="store_true")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    train_maps = set(a.train_maps.split(","))
    test_maps = set(a.test_maps.split(","))
    by_split = {"train": [], "val": [], "test": [], "test_seen_map": []}
    recs = [r for r in load_records(a.inputs) if a.no_gate_filter or gate_passes(r["ctx"])]
    seeds = sorted({scenario_of(r) for r in recs if scenario_of(r)[0] in train_maps})
    rng2 = random.Random(a.seed + 1)
    rng2.shuffle(seeds)
    n_val = max(1, int(round(a.val_frac * len(seeds)))) if len(seeds) > 1 else 0
    val_seeds = set(seeds[:n_val])
    # hold out one more train-map seed as a "seen map, unseen seed" test set
    seen_test = set(seeds[n_val:n_val + 1]) if len(seeds) > 2 else set()
    for r in recs:
        ex = to_example(r, a.families, a.label)
        if ex is None:
            continue
        sc = scenario_of(r)
        if sc[0] in test_maps:
            by_split["test"].append(ex)
        elif sc in val_seeds:
            by_split["val"].append(ex)
        elif sc in seen_test:
            by_split["test_seen_map"].append(ex)
        elif sc[0] in train_maps:
            by_split["train"].append(ex)
    if not a.no_balance:
        by_split["train"] = balance(by_split["train"], rng)
    for k in by_split:
        rng.shuffle(by_split[k])
        lim = a.max_train if k == "train" else a.max_eval
        if lim:
            by_split[k] = by_split[k][:lim]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"records_used": len(recs), "val_seeds": sorted(map(list, val_seeds)), "seen_map_test_seeds": sorted(map(list, seen_test))}
    for k, v in by_split.items():
        with open(out / f"{k}.jsonl", "w") as f:
            for e in v:
                f.write(json.dumps(e) + "\n")
        stats[k] = {"n": len(v), "kinds": dict(collections.Counter(e["meta"]["kind"] for e in v)),
                    "maps": dict(collections.Counter(e["meta"]["map"] for e in v))}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
