"""Minimal parser for OMNeT++ .sca scalar files (no opp_scavetool dependency)."""
import math
import shlex
from pathlib import Path


def parse_sca(path):
    """Return {'world': {name: value}, 'modules': {module: {name: value}}, 'attrs': {...}, 'params': {...}}."""
    world, modules, attrs, params = {}, {}, {}, {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("scalar "):
            try:
                _, mod, name, val = shlex.split(line)[:4]
            except ValueError:
                continue
            v = float(val) if val not in ("nan", "-nan") else math.nan
            if mod.endswith(".world"):
                world[name] = v
            modules.setdefault(mod, {})[name] = v
        elif line.startswith("attr "):
            parts = shlex.split(line)
            if len(parts) >= 3:
                attrs[parts[1]] = parts[2]
        elif line.startswith("config ") or line.startswith("param "):
            parts = shlex.split(line)
            if len(parts) >= 3:
                params[parts[1]] = parts[2]
    return {"world": world, "modules": modules, "attrs": attrs, "params": params}


def check_accounting(m):
    """Task accounting: generated = in-time + late + dropped, metrics finite and in range."""
    need = ["tasksGenerated", "tasksInTime", "tasksLate", "tasksDropped", "tcr", "packetDropRatio", "overheadRatio"]
    miss = [k for k in need if k not in m]
    if miss:
        return False, f"missing scalars {miss}"
    g, a, l, d = (int(m[k]) for k in need[:4])
    if g <= 0:
        return False, "no tasks generated"
    if a + l + d != g:
        return False, f"accounting mismatch {a}+{l}+{d}!={g}"
    for k in ("tcr", "packetDropRatio", "overheadRatio"):
        if not (0.0 <= m[k] <= 1.0):
            return False, f"{k}={m[k]} out of range"
    return True, f"accounting ok ({g}={a}+{l}+{d})"
