import random

import numpy as np

from liteoff_llm.backends import ReplayStore
from liteoff_llm.rl import SPLITS, RLManager, featurize
from liteoff_llm.synthetic import random_context


def test_replay_store(tmp_path):
    s = ReplayStore(tmp_path / "a.db")
    assert s.get("m", "sys", "p") is None
    s.put("m", "sys", "p", {"raw": "target=1", "n_in": 3, "n_out": 2})
    assert s.get("m", "sys", "p")["raw"] == "target=1"
    assert s.get("m2", "sys", "p") is None
    t = ReplayStore(tmp_path / "b.db")
    t.put("x", "s", "q", {"raw": "target=0", "n_in": 1, "n_out": 1})
    s.merge_from(tmp_path / "b.db")
    assert len(s) == 2
    ro = ReplayStore(tmp_path / "a.db", readonly=True)
    assert ro.get("x", "s", "q")["raw"] == "target=0"


def test_featurize_shapes():
    ctx = random_context(random.Random(0), 3)
    x, m = featurize(ctx, 5)
    assert x.shape == (10 + 5 * 12,) and m.tolist() == [True, True, True, True, False, False]
    x2, _ = featurize(ctx, 5, multi_agent=True)
    assert x2.shape[0] == x.shape[0] + 3


def _bandit(manager, n=1500, seed=1):
    """Reward 1 for the slot whose candidate is an RSU (0 if none), -1 otherwise."""
    rng = random.Random(seed)
    hits = []
    for tid in range(n):
        ctx = random_context(rng, 4)
        best = next((c["slot"] for c in ctx["cands"] if c["kind"] == "rsu"), 0)
        rep = manager.decide({"ctx": ctx, "vid": tid % 7, "tid": tid, "region": tid % 2})
        r = 1.0 if rep["slot"] == best else -1.0
        hits.append(rep["slot"] == best)
        manager.feedback({"tid": tid, "reward": r})
    return np.mean(hits[:200]), np.mean(hits[-300:])


def test_b1_action_space_is_local_or_rsu_with_split():
    m = RLManager("b1", "train", None, seed=0)
    rng = random.Random(3)
    for tid in range(300):
        ctx = random_context(rng, 5)
        rep = m.decide({"ctx": ctx, "vid": 1, "tid": tid})
        assert rep["slot"] == 0 or ctx["cands"][rep["slot"] - 1]["kind"] == "rsu"
        assert rep["split"] in SPLITS
        if not ctx["task"]["split"] or rep["slot"] == 0:
            assert rep["split"] == 0
        m.feedback({"tid": tid, "reward": 0.0})


def test_b1_handover_penalty():
    m = RLManager("b1", "train", None, seed=0)
    ctx = random_context(random.Random(4), 3)
    m.decide({"ctx": ctx, "vid": 1, "tid": 1})
    m.feedback({"tid": 1, "reward": 1.0, "reoffloaded": 1})
    assert abs(m.last[1]["r"] - (1.0 - m.handover_penalty)) < 1e-9


def test_dqn_learns(tmp_path):
    m = RLManager("b1", "train", str(tmp_path / "b1.json"), seed=0)
    m.agent.eps_decay = 300
    early, late = _bandit(m)
    assert late > early + 0.2, (early, late)
    m.close()
    assert (tmp_path / "b1.json").exists()
    ev = RLManager("b1", "eval", str(tmp_path / "b1.json"))
    assert ev.agent.steps > 0


def test_ppo_learns_b4(tmp_path):
    m = RLManager("b4", "train", str(tmp_path / "b4.json"), seed=0)
    early, late = _bandit(m, 6000)
    assert late > early + 0.2, (early, late)


def test_madsaspg_learns_b5(tmp_path):
    m = RLManager("b5", "train", str(tmp_path / "b5.json"), seed=0)
    early, late = _bandit(m, 8000)
    assert late > early + 0.2, (early, late)
    m.close()
    ev = RLManager("b5", "eval", str(tmp_path / "b5.json"))
    assert ev.agent.steps > 0
    import pytest
    with pytest.raises(ValueError):
        RLManager("b1", "eval", str(tmp_path / "b5.json"))


def test_fedavg_meta_step(tmp_path):
    m = RLManager("b4", "train", str(tmp_path / "b4.json"), seed=0)
    _bandit(m, 400)
    assert len(m.regions) == 2
    g0 = [p.copy() for p in m.agent.params()]
    assert m.fedavg() == 2
    p0 = [a.params()[0] for a in m.regions.values()]
    assert np.allclose(p0[0], p0[1]) and np.allclose(p0[0], m.agent.params()[0])
    assert not np.allclose(g0[0], m.agent.params()[0])
