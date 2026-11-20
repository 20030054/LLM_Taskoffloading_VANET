import random

from liteoff_llm.encoder import ALL_FAMILIES, SYSTEM_PROMPT, encode, format_answer, parse_answer
from liteoff_llm.synthetic import random_context


def test_full_prompt_has_all_families():
    ctx = random_context(random.Random(0), 5)
    ctx["task"]["split"] = True
    p = encode(ctx)
    assert p.startswith("TASK:") and "ROAD:" in p and "link_left=" in p and "trust=" in p and "rate=" in p
    assert p.count("\nC") == 5
    assert p.splitlines()[-1] == "ANSWER: target=<0-5>; split=<0-100>"


def test_prompt_length_budget():
    # plan: 150-300 tokens; a token is ~3-4 characters for this numeric text
    for s in range(20):
        p = encode(random_context(random.Random(s), 5))
        assert len(p) < 1400, len(p)


def test_basic_context_only_speed_distance():
    p = encode(random_context(random.Random(1), 4), "basic")
    assert "link_left" not in p and "trust" not in p and "rate=" not in p and "ROAD" not in p
    assert "dist=" in p or "via_rsu" in p


def test_family_increments_are_monotone():
    ctx = random_context(random.Random(2), 5)
    prev = 0
    for i in range(1, len(ALL_FAMILIES) + 1):
        n = len(encode(ctx, ",".join(ALL_FAMILIES[:i])))
        assert n >= prev
        prev = n


def test_pred_family():
    p = encode(random_context(random.Random(3), 3), "mobility,pred")
    assert "dist_in_2s=" in p


def test_answer_roundtrip_and_parser_parity_with_cpp():
    assert parse_answer(format_answer(3, 40)) == (True, 3, 40)
    assert parse_answer(format_answer(2, 0, splittable=False)) == (True, 2, 0)
    # same cases as sim/tests/test_core.cc
    assert parse_answer("target=2; split=30; conf=0.8") == (True, 2, 30)
    assert parse_answer("Target: c1, split: 0")[:2] == (True, 1)
    assert parse_answer("target=local")[1] == 0
    assert parse_answer("TARGET = 0 ; split = no")[0]
    assert not parse_answer("I think candidate 2 is best")[0]
    assert not parse_answer("target=two")[0]
    assert not parse_answer("target=1; split=150")[0]
    assert not parse_answer("")[0]
    assert not parse_answer("target=7", k=5)[0]


def test_system_prompt_mentions_format():
    assert "target=<id>" in SYSTEM_PROMPT
