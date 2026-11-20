"""Context encoder: turns the simulator's structured context (JSON) into a compact prompt.

This module is the single source of truth for prompts: the side-car, the dataset builder, teacher
labelling, training and evaluation all call `encode()`.

Feature families (plan section 5):
    mobility   relative speed, closing speed, predicted link lifetime
    link       expected PHY rate, beacon loss, channel busy ratio
    resources  CPU, backlog, load, queue, energy
    task       priority, splittable, result size (size/cycles/deadline are always present)
    trust      past success rate, recent timeouts
    road       road type, density, time to junction, RSU distance, coverage left, route overlap
Special:
    basic      only speed and distance (E7 lower bound) - used when no other family is listed
    pred       B3 digital-twin style predicted distance in 2 s
"""
from __future__ import annotations

import json
import re

ALL_FAMILIES = ("mobility", "link", "resources", "task", "trust", "road")

SYSTEM_PROMPT = (
    "You are the task-offloading controller of a vehicle. Pick where the task runs so that it "
    "finishes before its deadline. Target 0 = run on this vehicle; 1..K = the listed candidates. "
    "For splittable tasks also give split = percent of the task kept local (0 = send the whole task). "
    "Reply with exactly: target=<id>  (splittable: target=<id>; split=<0-100>)"
)

ANSWER_RE = re.compile(r"target\s*[=:]\s*\"?<?\s*(local|self|c(?:and)?\d+|\d+)", re.I)
SPLIT_RE = re.compile(r"split\s*[=:]\s*\"?<?\s*(no|none|\d+(?:\.\d+)?)", re.I)


def parse_families(spec: str | None):
    if not spec or spec.strip() in ("all", "full", "*"):
        return set(ALL_FAMILIES)
    fam = {f.strip() for f in spec.split(",") if f.strip()}
    return fam


def _f(x, nd=0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "?"
    if nd == 0:
        return str(int(round(x)))
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def _size(b):
    b = float(b)
    return f"{b / 1e6:.2f}MB" if b >= 1e6 else f"{b / 1e3:.0f}KB"


def _lvl(x, lo, hi, names=("low", "med", "high")):
    x = float(x)
    return names[0] if x < lo else (names[2] if x >= hi else names[1])


def encode(ctx: dict | str, families: str | None = None) -> str:
    """Render the context as a short prompt (target 150-300 tokens for K=5)."""
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
    fam = parse_families(families if families is not None else ctx.get("families"))
    basic_only = not (fam & set(ALL_FAMILIES))
    t, s, r = ctx["task"], ctx["self"], ctx.get("road", {})
    cands = ctx.get("cands", [])
    L = []
    task = (f"TASK: size={_size(t['bytes'])} cycles={_f(float(t['cycles']) / 1e9, 2)}G "
            f"deadline={_f(t['deadline_ms'])}ms left={_f(t['budget_ms'])}ms")
    if "task" in fam:
        prio = {2: "high", 1: "med", 0: "low"}.get(int(round(float(t.get("prio", 0)))), "low")
        task += f" prio={prio} split={'yes' if t.get('split') else 'no'} result={_size(t.get('result_bytes', 0))}"
    L.append(task)
    me = f"SELF: speed={_f(s.get('speed', 0))}m/s"
    if "resources" in fam or basic_only:
        me += f" cpu={_f(s['cpu_ghz'], 1)}GHz"
    if "resources" in fam:
        me += (f" backlog={_f(s['backlog_ms'])}ms queue={_f(s.get('queue', 0))} load={_f(100 * float(s.get('load', 0)))}% "
               f"energy={'ok' if float(s.get('energy', 1)) > 0.2 else 'low'}")
    if "link" in fam:
        me += f" cbr={_f(s.get('cbr', 0), 2)}"
    L.append(me)
    for c in cands:
        kind = c["kind"]
        parts = [f"C{c['slot']} {kind}:"]
        if kind == "cloud":
            parts.append("via_rsu")
        else:
            parts.append(f"dist={_f(c['dist'])}m")
            if kind == "veh":
                parts.append(f"rel_speed={_f(c['rel_speed'])}m/s")
        if "mobility" in fam:
            parts.append(f"link_left={_f(min(float(c['link_left']), 60))}s")
            if kind == "veh":
                parts.append("closing" if float(c.get("closing", 0)) < -0.5 else ("leaving" if float(c.get("closing", 0)) > 0.5 else "steady"))
        if "pred" in fam and kind != "cloud":
            parts.append(f"dist_in_2s={_f(c.get('pred_dist_2s', c['dist']))}m")
        if "link" in fam:
            parts.append(f"rate={_f(c['rate_mbps'])}Mbps")
            if kind != "cloud":
                parts.append(f"loss={_f(100 * float(c.get('loss', 0)))}%")
        if "resources" in fam:
            parts.append(f"cpu={_f(c['cpu_ghz'], 1)}GHz")
            if kind == "cloud":
                parts.append(f"backhaul={_f(2 * float(c.get('backhaul_ms', 0)))}ms")
            else:
                parts.append(f"backlog={_f(c['backlog_ms'])}ms load={_f(100 * float(c.get('load', 0)))}%")
        if "trust" in fam:
            parts.append(f"trust={_f(c.get('trust', 0.8), 2)}")
            if int(float(c.get("timeouts", 0))) > 0:
                parts.append(f"timeouts={_f(c['timeouts'])}")
        if "road" in fam and kind == "veh":
            parts.append(f"route_overlap={'yes' if c.get('route_overlap') else 'no'}")
        L.append(" ".join(parts))
    if not cands:
        L.append("NO CANDIDATES IN RANGE")
    if "road" in fam:
        ji = float(r.get("junction_in", -1))
        rd = float(r.get("rsu_dist", -1))
        L.append(f"ROAD: type={r.get('type', '?')} density={_lvl(r.get('density', 0), 60, 150)} "
                 f"junction_in={'?' if ji < 0 else _f(ji) + 's'} rsu_dist={'none' if rd < 0 else _f(rd) + 'm'} "
                 f"coverage_left={_f(min(float(r.get('coverage_left', 0)), 60))}s")
    L.append(f"ANSWER: target=<0-{len(cands)}>" + ("; split=<0-100>" if t.get("split") else ""))
    return "\n".join(L)


def format_answer(slot: int, split: int = 0, splittable: bool = True) -> str:
    return f"target={int(slot)}; split={int(split)}" if splittable else f"target={int(slot)}"


def parse_answer(raw: str, k: int | None = None):
    """Python mirror of the C++ parser. Returns (ok, slot, split)."""
    if not raw:
        return False, -1, 0
    m = ANSWER_RE.search(raw)
    if not m:
        return False, -1, 0
    tv = m.group(1).lower()
    if tv in ("local", "self"):
        slot = 0
    else:
        slot = int(re.sub(r"\D", "", tv))
    split = 0
    ms = SPLIT_RE.search(raw)
    if ms and ms.group(1).lower() not in ("no", "none"):
        split = int(round(float(ms.group(1))))
        if not 0 <= split <= 100:
            return False, slot, 0
    if k is not None and not 0 <= slot <= k:
        return False, slot, split
    return True, slot, split
