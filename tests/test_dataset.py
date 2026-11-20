import json
import random

from liteoff_llm import dataset
from liteoff_llm.synthetic import random_context


def rec(mp, seed, rng):
    ctx = random_context(rng, 4)
    ctx["self"]["local_est_ms"] = ctx["task"]["budget_ms"] * 2  # passes the gate
    ctx["task"]["cycles"] = 1e9
    return {"scenario": f"{mp}-v40-s{seed}", "ctx": ctx, "oracle": {"slot": rng.randint(0, 4), "split": 0},
            "chosen": {"slot": 0, "split": 0, "path": "explore"}, "outcome": {"status": "in_time", "delay_ms": 10}}


def test_split_by_map_and_seed(tmp_path):
    rng = random.Random(0)
    f = tmp_path / "d.jsonl"
    with open(f, "w") as fh:
        for mp, seeds in (("grid", range(1, 7)), ("highway", range(1, 4)), ("spider", range(1, 3))):
            for s in seeds:
                for _ in range(30):
                    fh.write(json.dumps(rec(mp, s, rng)) + "\n")
    dataset.main([str(f), "--out", str(tmp_path / "out"), "--no-balance"])
    sp = {k: [json.loads(l) for l in open(tmp_path / "out" / f"{k}.jsonl")] for k in ("train", "val", "test", "test_seen_map")}
    key = lambda e: (e["meta"]["map"], e["meta"]["seed"])  # noqa: E731
    tr, va, te, ts = ({key(e) for e in sp[k]} for k in ("train", "val", "test", "test_seen_map"))
    assert tr and va and te and ts
    assert not (tr & va) and not (tr & te) and not (tr & ts) and not (va & ts)
    assert {m for m, _ in te} == {"spider"}
    e = sp["train"][0]
    assert e["answer"].startswith("target=") and e["prompt"].startswith("TASK:")


def test_gate_filter():
    rng = random.Random(1)
    ctx = random_context(rng, 4)
    ctx["self"]["local_est_ms"] = 1
    assert not dataset.gate_passes(ctx)
    ctx["self"]["local_est_ms"] = 1e6
    ctx["task"]["cycles"] = 1e9
    assert dataset.gate_passes(ctx)
    ctx["cands"] = ctx["cands"][:1]
    assert not dataset.gate_passes(ctx)
