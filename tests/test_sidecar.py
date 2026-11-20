import argparse
import random

from liteoff_llm.backends import ReplayStore
from liteoff_llm.encoder import SYSTEM_PROMPT, encode
from liteoff_llm.sidecar import Service
from liteoff_llm.synthetic import random_context


def make(tmp_path, **kw):
    a = argparse.Namespace(port=0, host="127.0.0.1", workdir=str(tmp_path), ollama_url="http://127.0.0.1:1", num_predict=12,
                           num_thread=None, no_logprobs=False, replay_db=str(tmp_path / "r.db"), replay_mode="read",
                           backend="replay", model_map="sim-student=tag-student", rl_weights="", log_decisions=True)
    for k, v in kw.items():
        setattr(a, k, v)
    return Service(a)


def test_replay_decision_and_miss(tmp_path):
    ctx = random_context(random.Random(5), 3)
    st = ReplayStore(tmp_path / "r.db")
    st.put("tag-student", SYSTEM_PROMPT, encode(ctx), {"raw": "target=2", "n_in": 180, "n_out": 4, "conf": 0.9, "wall_ms": 5})
    st.close()
    svc = make(tmp_path)
    assert svc.handle({"op": "hello", "scheme": "liteoff"})["ok"]
    r = svc.handle({"op": "decide", "scheme": "liteoff", "model": "sim-student", "vid": 1, "tid": 1, "ctx": ctx})
    assert r["ok"] and r["slot"] == 2 and r["n_in"] == 180 and r["source"] == "replay"
    ctx2 = random_context(random.Random(6), 3)
    assert svc.handle({"op": "decide", "scheme": "liteoff", "model": "sim-student", "vid": 1, "tid": 2, "ctx": ctx2}) == {"error": "replay_miss"}
    assert svc.handle({"op": "bye"})["ok"]
    assert (tmp_path / "sidecar_stats.json").exists() and (tmp_path / "decisions.jsonl").exists()


def test_rl_protocol(tmp_path):
    svc = make(tmp_path, replay_db="")
    svc.handle({"op": "hello", "scheme": "b1", "rlMode": "train", "rlWeights": str(tmp_path / "w.json"), "seed": 1})
    ctx = random_context(random.Random(7), 4)
    r = svc.handle({"op": "decide", "scheme": "b1", "vid": 3, "tid": 9, "region": 1, "ctx": ctx})
    assert r["ok"] and 0 <= r["slot"] <= 4
    assert svc.handle({"op": "feedback", "scheme": "b1", "tid": 9, "reward": 1.0})["ok"]
    assert svc.handle({"op": "fedavg", "region": 1})["ok"]
    svc.handle({"op": "bye"})
    assert (tmp_path / "w.json").exists()
    assert "error" in svc.handle({"op": "nope"})
