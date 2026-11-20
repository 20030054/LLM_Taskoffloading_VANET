"""Python reference of the C++ completion estimator and greedy rule (sim/src/core/Core.cc).

Operates on the logged context JSON (values in the units contextToJson writes). Used for offline
evaluation (feasibility of a model's choice) and cross-checked against the C++ core in the tests.
"""
from __future__ import annotations

MAC_EFF, LINK_MARGIN, RANGE = 0.6, 1.2, 300.0


def _local(ctx, f=1.0):
    s, t = ctx["self"], ctx["task"]
    return float(s["backlog_ms"]) / 1000.0 + float(t["cycles"]) * f / (float(s["cpu_ghz"]) * 1e9)


def _remote(ctx, c, f):
    t, s = ctx["task"], ctx["self"]
    eff = float(c["rate_mbps"]) * 1e6 * MAC_EFF * max(0.1, 1.0 - float(s.get("cbr", 0)))
    up = float(t["bytes"]) * f * 8 / eff
    down = float(t["result_bytes"]) * f * 8 / eff
    bh = 2 * float(c.get("backhaul_ms", 0)) / 1000.0 if c["kind"] == "cloud" else 0.0
    ex = float(t["cycles"]) * f / (float(c["cpu_ghz"]) * 1e9)
    total = up + bh + float(c["backlog_ms"]) / 1000.0 + ex + down
    ll = float(c["link_left"])
    feasible = ll >= total * LINK_MARGIN if c["kind"] == "veh" else ll >= up * LINK_MARGIN
    return total if feasible else float("inf")


def estimate(ctx, slot, split=0):
    """Expected completion time [s] of a decision (inf if the link cannot carry it)."""
    cands = ctx.get("cands", [])
    if slot <= 0:
        return _local(ctx)
    if slot > len(cands):
        return float("inf")
    keep = (min(max(split, 0), 100) if ctx["task"].get("split") else 0) / 100.0
    f = 1.0 - keep
    if f <= 0:
        return _local(ctx)
    r = _remote(ctx, cands[slot - 1], f)
    return max(r, _local(ctx, keep) if keep > 0 else 0.0)


def rule_score(c, w_link=1.0, w_load=1.0, w_dist=0.5, w_trust=0.5):
    load = max(float(c.get("load", 0)), min(1.0, float(c["backlog_ms"]) / 1000.0))
    link = min(float(c["link_left"]), 20.0) / 20.0
    dist = 0.5 if c["kind"] == "cloud" else min(1.0, float(c["dist"]) / RANGE)
    return w_link * link - w_load * load - w_dist * dist + w_trust * float(c.get("trust", 0.8))


def rule_policy(ctx, local_margin=0.7):
    budget = float(ctx["task"]["budget_ms"]) / 1000.0
    loc = _local(ctx)
    if loc <= local_margin * budget:
        return 0
    best, best_s = -1, -float("inf")
    for i, c in enumerate(ctx.get("cands", [])):
        if estimate(ctx, i + 1) > budget:
            continue
        s = rule_score(c)
        if s > best_s:
            best, best_s = i + 1, s
    if best > 0:
        return best
    if loc <= budget:
        return 0
    slot, be = 0, loc
    for i in range(len(ctx.get("cands", []))):
        e = estimate(ctx, i + 1)
        if e < be:
            slot, be = i + 1, e
    return slot
