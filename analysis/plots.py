#!/usr/bin/env python3
"""Figures from analysis/agg.csv: metric vs swept variable, one line per scheme, 95% CI bands
(numeric sweeps) or grouped bars (categorical sweeps), plus LiteOff decision-path shares.

    python analysis/plots.py results/mini   -> results/mini/figures/*.png

Colour follows the scheme (fixed mapping, never by rank); beyond 8 schemes the colours repeat with
a different line style/marker so identity never relies on colour alone.
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDER = ["liteoff", "rule", "oracle", "b1", "b2", "b3", "b4", "b5", "local", "rsu", "random",
         "abl_basic_context", "abl_no_distill", "abl_no_quant", "abl_no_gate", "abl_no_validator", "abl_large_on_node", "abl_rsu_only_llm"]
STYLES = ["-", "--", ":"]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = {"tcr": "Task completion ratio", "meanDelay": "Mean end-to-end delay (s)", "packetDropRatio": "Packet drop ratio",
           "overheadRatio": "Overhead ratio (control/total bytes)", "energyPerTaskJ": "Energy per task (J)",
           "meanDecisionLatency": "Decision latency (s)", "oracleAgreement": "Agreement with oracle"}


def style(scheme):
    i = ORDER.index(scheme) if scheme in ORDER else len(ORDER) + abs(hash(scheme)) % 8
    return dict(color=PALETTE[i % 8], linestyle=STYLES[(i // 8) % 3], marker=MARKERS[(i // 8) % 3])


def setup(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def split_point(p):
    if "=" not in p:
        return None, p
    k, v = p.split("=", 1)
    return k, v


def num(v):
    try:
        return float(re.sub(r"[a-zA-Z]+$", "", v))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    a = ap.parse_args()
    root = Path(a.results)
    agg = pd.read_csv(root / "analysis" / "agg.csv")
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    n = 0
    for (exp, mp), g in agg.groupby(["exp", "map"]):
        g = g.copy()
        g[["key", "val"]] = g.point.apply(lambda p: pd.Series(split_point(p)))
        for key, gk in g.groupby("key", dropna=False):
            numeric = gk.val.map(num).notna().all() and gk.val.nunique() > 1
            for m, label in METRICS.items():
                if m not in gk or gk[m].isna().all():
                    continue
                fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
                schemes = [s for s in ORDER if s in set(gk.scheme)] + sorted(set(gk.scheme) - set(ORDER))
                if numeric:
                    for s in schemes:
                        d = gk[gk.scheme == s].assign(x=lambda t: t.val.map(num)).sort_values("x")
                        st = style(s)
                        ax.plot(d.x, d[m], linewidth=2, markersize=6, label=s, **st)
                        ax.fill_between(d.x, d[m + "_lo"], d[m + "_hi"], color=st["color"], alpha=0.12, linewidth=0)
                    ax.set_xlabel(str(key).replace("*.world.", "").replace("**.appl.", ""), color=INK)
                else:
                    vals = list(dict.fromkeys(gk.val))
                    w = 0.8 / max(1, len(schemes))
                    for j, s in enumerate(schemes):
                        d = gk[gk.scheme == s].set_index("val").reindex(vals)
                        st = style(s)
                        x = np.arange(len(vals)) + (j - (len(schemes) - 1) / 2) * w
                        ax.bar(x, d[m], width=w * 0.92, color=st["color"], label=s, edgecolor="white", linewidth=1)
                        ax.errorbar(x, d[m], yerr=[d[m] - d[m + "_lo"], d[m + "_hi"] - d[m]], fmt="none", ecolor=MUTED, elinewidth=1)
                    ax.set_xticks(np.arange(len(vals)))
                    ax.set_xticklabels([v[:22] for v in vals], rotation=20, ha="right")
                setup(ax, label)
                ax.set_title(f"{exp} ({mp})", color=INK, loc="left", fontsize=11)
                if len(schemes) > 1:
                    ax.legend(frameon=False, fontsize=7, ncol=2 if len(schemes) > 6 else 1, loc="best")
                fig.tight_layout()
                kk = re.sub(r"[^A-Za-z0-9]+", "_", str(key))
                fig.savefig(figs / f"{exp}_{mp}_{kk}_{m}.png")
                plt.close(fig)
                n += 1
    # decision path shares of LiteOff
    lo = agg[agg.scheme == "liteoff"]
    cols = [c for c in agg.columns if c.startswith("pathShare_") and not c.endswith(("_lo", "_hi"))]
    if not lo.empty and cols:
        d = lo.groupby("exp")[cols].mean()
        d = d.loc[:, d.sum() > 0]
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
        bottom = np.zeros(len(d))
        for i, c in enumerate(d.columns):
            ax.bar(d.index, d[c], bottom=bottom, color=PALETTE[i % 8], label=c.replace("pathShare_", ""), edgecolor="white", linewidth=1.5)
            bottom += d[c].values
        setup(ax, "Share of decisions")
        ax.set_title("LiteOff decision paths (gate / cache / LLM / fallback)", color=INK, loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(figs / "liteoff_paths.png")
        plt.close(fig)
        n += 1
    print(f"{n} figures -> {figs}")


if __name__ == "__main__":
    main()
