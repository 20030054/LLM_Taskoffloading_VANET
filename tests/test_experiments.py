import json
from pathlib import Path

import yaml

import make_manifest
from omnet_results import check_accounting, parse_sca
import stats

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_mini_and_full():
    cfg = yaml.safe_load(open(ROOT / "experiments" / "experiments.yaml"))
    mini = make_manifest.expand(cfg, "mini")
    full = make_manifest.expand(cfg, "full")
    assert len(full) > len(mini) > 0
    outdirs = [r["outdir"] for r in full]
    assert len(outdirs) == len(set(outdirs)), "every run needs its own output directory"
    e1 = [r for r in full if r["exp"] == "E1H"]
    assert {r["spec"]["vehicles"] for r in e1} == {60.0, 120.0, 240.0, 360.0}
    for r in full:
        s = r["spec"]
        assert s["scheme"] in {"local", "rsu", "random", "rule", "oracle", "liteoff", "b1", "b2", "b3", "b4", "b5"}
        assert s["duration"] > s["warmup"]


def test_sca_parse_and_accounting(tmp_path):
    f = tmp_path / "r.sca"
    f.write_text("version 3\nrun General-0\nattr configname General\n"
                 "scalar LiteOffScenario.world tasksGenerated 10\nscalar LiteOffScenario.world tasksInTime 6\n"
                 "scalar LiteOffScenario.world tasksLate 3\nscalar LiteOffScenario.world tasksDropped 1\n"
                 "scalar LiteOffScenario.world tcr 0.6\nscalar LiteOffScenario.world packetDropRatio 0.1\n"
                 "scalar LiteOffScenario.world overheadRatio 0.02\nscalar LiteOffScenario.rsu[0].appl servedCycles 1e9\n")
    sc = parse_sca(f)
    assert sc["world"]["tcr"] == 0.6 and sc["modules"]["LiteOffScenario.rsu[0].appl"]["servedCycles"] == 1e9
    assert check_accounting(sc["world"])[0]
    sc["world"]["tasksLate"] = 4
    assert not check_accounting(sc["world"])[0]


def test_stats_ci_and_paired():
    m, lo, hi = stats.mean_ci([1, 2, 3, 4, 5])
    assert m == 3 and lo < 3 < hi
    r = stats.paired_test([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7.5])
    assert r["p_t"] < 0.05 and r["n"] == 6 and r["effect_dz"] < 0
    adj = stats.holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and all(0 <= p <= 1 for p in adj)
