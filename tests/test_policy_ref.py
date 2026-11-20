import math

from liteoff_llm.policy_ref import estimate, rule_policy


def ctx_like_cpp():
    """Same numbers as makeCtx() in sim/tests/test_core.cc (JSON units)."""
    return {"task": {"bytes": 100e3, "cycles": 1e9, "result_bytes": 1e3, "deadline_ms": 300, "budget_ms": 300, "split": False},
            "self": {"cpu_ghz": 2.0, "backlog_ms": 0, "cbr": 0.0},
            "cands": [{"slot": 1, "kind": "rsu", "dist": 120, "link_left": 15, "rate_mbps": 12, "cpu_ghz": 20, "backlog_ms": 0,
                       "load": 0, "trust": 0.95},
                      {"slot": 2, "kind": "veh", "dist": 60, "link_left": 0.2, "rate_mbps": 18, "cpu_ghz": 3, "backlog_ms": 0,
                       "load": 0, "trust": 0.8}]}


def test_estimates_match_cpp():
    c = ctx_like_cpp()
    assert math.isclose(estimate(c, 0), 0.5)
    assert math.isclose(estimate(c, 1), 100e3 * 8 / 7.2e6 + 0.05 + 1e3 * 8 / 7.2e6)
    assert math.isinf(estimate(c, 2)) and math.isinf(estimate(c, 3))
    c["task"]["split"] = True
    assert math.isclose(estimate(c, 1, 50), max(0.25, 50e3 * 8 / 7.2e6 + 0.025 + 500 * 8 / 7.2e6))


def test_rule_matches_cpp():
    c = ctx_like_cpp()
    assert rule_policy(c) == 1
    c["self"]["cpu_ghz"] = 10
    assert rule_policy(c) == 0
