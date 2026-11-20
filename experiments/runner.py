#!/usr/bin/env python3
"""Run one LiteOff simulation end to end.

A run spec (JSON/YAML dict or CLI flags) names the scenario, the scheme and any ini overrides:
    map, vehicles, vmax, seed, duration, warmup, rsu_spacing, scheme, params{ini-key: value}
The runner
  1. generates the SUMO scenario for this run (routes calibrated to the target density),
  2. writes run.ini (overrides first, then rsus.ini, then sim/simulations/base.ini),
  3. starts the Python side-car (and checks Ollama) when the scheme needs it,
  4. runs opp_run (Cmdenv), parses the scalar file and checks task accounting,
  5. writes summary.json into the run directory.

Usage:
  python experiments/runner.py --map grid --vehicles 40 --scheme rule --seed 1 --duration 120 --out results/smoke
  python experiments/runner.py --spec spec.json --out DIR
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "llm"))
from omnet_results import parse_sca, check_accounting  # noqa: E402

VEINS = Path(os.environ.get("VEINS_HOME", Path.home() / "sim" / "veins-veins-5.3.1"))
SIDECAR_SCHEMES = {"liteoff", "b1", "b2", "b3", "b4", "b5"}
LLM_SCHEMES = {"liteoff", "b2", "b3"}
ROAD_TYPE = {"grid": "urban", "highway": "highway", "spider": "suburban"}

DEFAULTS = dict(map="grid", vehicles=40, vmax=13.9, seed=1, duration=300.0, warmup=60.0, rsu_spacing=500.0,
                scheme="rule", params={}, task_rate=1.0, trace=False, dataset=False, tag="",
                sidecar_args="", latency_file=str(REPO / "profiling" / "latency_model.csv"))


def ini_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v).strip()
    if s.lower() in ("true", "false"):
        return s.lower()
    # already an ini expression (quoted string, quantity with unit, function call)?
    if s.startswith('"') or (s[:1].isdigit() and s[-1:].isalpha()) or ("(" in s and s.endswith(")")):
        return s
    try:
        float(s)
        return s
    except ValueError:
        return json.dumps(s)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ollama_up(url):
    import urllib.request
    try:
        urllib.request.urlopen(url.rstrip("/") + "/api/version", timeout=2).read()
        return True
    except Exception:
        return False


def start_sidecar(rundir: Path, spec, log):
    port = free_port()
    cmd = [sys.executable, "-m", "liteoff_llm.sidecar", "--port", str(port), "--workdir", str(rundir)]
    cmd += shlex.split(spec.get("sidecar_args", ""))
    env = dict(os.environ, PYTHONPATH=str(REPO / "llm") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=rundir)
    t0 = time.time()
    while time.time() - t0 < 60:
        if proc.poll() is not None:
            raise RuntimeError(f"side-car exited early, see {log.name}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc, port
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("side-car did not start")


def write_ini(rundir: Path, spec, scen: Path, port: int):
    p = dict(spec["params"])
    w = {
        "*.world.scheme": spec["scheme"],
        "*.world.warmup": f"{spec['warmup']}s",
        "*.world.roadType": ROAD_TYPE.get(spec["map"], "urban"),
        "*.world.scenarioTag": spec.get("tag") or f"{spec['map']}-v{spec['vehicles']}-s{spec['seed']}",
        "*.world.sidecarPort": port,
        "*.node[*].appl.taskRate": f"{spec['task_rate']}Hz",
        "*.node[*].appl.taskStopTime": f"{spec['duration'] - 3}s",
    }
    meta = json.loads((scen / "meta.json").read_text())
    if not meta.get("obstacles"):
        cfg = REPO / "sim" / "simulations" / "config_noobs.xml"
        w["*.**.nic.phy80211p.analogueModels"] = f'xmldoc("{cfg}")'
    if spec.get("latency_file") and Path(spec["latency_file"]).exists():
        w["*.world.latencyFile"] = str(Path(spec["latency_file"]).resolve())
    if spec.get("trace"):
        w["*.world.traceFile"] = str(rundir / "trace.csv")
    if spec.get("dataset"):
        w["*.world.datasetFile"] = str(rundir / "dataset.jsonl")
    lines = ["[General]", f"sim-time-limit = {spec['duration']}s", f"seed-set = {spec['seed']}",
             f"*.manager.seed = {spec['seed']}", f'*.manager.configFile = "{scen / "run.sumo.cfg"}"',
             f'output-scalar-file = "{rundir / "results.sca"}"', f'output-vector-file = "{rundir / "results.vec"}"']
    for k, v in {**w, **p}.items():   # explicit params override defaults above
        lines.append(f"{k} = {ini_value(v)}")
    lines.append(f"include {scen / 'rsus.ini'}")
    lines.append(f"include {REPO / 'sim' / 'simulations' / 'base.ini'}")
    (rundir / "run.ini").write_text("\n".join(lines) + "\n")


def run(spec, out: Path, keep_scenario=False, quiet=False):
    spec = {**DEFAULTS, **spec}
    spec["params"] = dict(spec.get("params") or {})
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    t_start = time.time()
    scen = out / "scenario"
    gen = [sys.executable, str(REPO / "scenarios" / "gen_scenarios.py"), "routes", "--map", spec["map"],
           "--vehicles", str(spec["vehicles"]), "--vmax", str(spec["vmax"]), "--seed", str(spec["seed"]),
           "--duration", str(spec["duration"]), "--warmup", str(min(spec["warmup"], spec["duration"] / 2)),
           "--rsu-spacing", str(spec["rsu_spacing"]), "--out", str(scen)]
    subprocess.run(gen, check=True, capture_output=True, text=True)
    scen_meta = json.loads((scen / "meta.json").read_text())

    side = None
    port = 0
    log = open(out / "sidecar.log", "w")
    try:
        if spec["scheme"] in SIDECAR_SCHEMES:
            if spec["scheme"] in LLM_SCHEMES:
                url = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
                if "--backend replay" not in spec.get("sidecar_args", "") and not ollama_up(url):
                    raise RuntimeError(f"Ollama not reachable at {url} (start it with env/start_ollama.sh)")
            side, port = start_sidecar(out, spec, log)
        write_ini(out, spec, scen, port)
        oppenv = dict(os.environ)
        cmd = ["opp_run", "-u", "Cmdenv", "-l", str(VEINS / "src" / "veins"), "-l", str(REPO / "sim" / "src" / "liteoff"),
               "-n", f"{VEINS / 'src' / 'veins'}:{REPO / 'sim' / 'src'}", "-f", str(out / "run.ini"), "-c", "General"]
        with open(out / "omnet.log", "w") as olog:
            r = subprocess.run(cmd, cwd=REPO / "sim" / "simulations", stdout=olog, stderr=subprocess.STDOUT, env=oppenv)
        ok = r.returncode == 0
    finally:
        if side is not None:
            side.send_signal(signal.SIGTERM)
            try:
                side.wait(10)
            except subprocess.TimeoutExpired:
                side.kill()
        log.close()
    wall = time.time() - t_start
    summary = {"ok": ok, "wall_s": round(wall, 2), "spec": spec, "scenario": scen_meta}
    if ok and (out / "results.sca").exists():
        sc = parse_sca(out / "results.sca")
        summary["metrics"] = sc["world"]
        acc_ok, msg = check_accounting(sc["world"])
        summary["accounting_ok"] = acc_ok
        summary["accounting_msg"] = msg
        summary["ok"] = acc_ok
    else:
        tail = (out / "omnet.log").read_text(errors="replace")[-3000:] if (out / "omnet.log").exists() else ""
        summary["error_tail"] = tail
    try:
        import resource
        summary["max_rss_mb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    except Exception:
        pass
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if not keep_scenario:
        for f in ("routes.rou.xml",):
            (scen / f).unlink(missing_ok=True)
    if not quiet:
        m = summary.get("metrics", {})
        print(f"[{'OK' if summary['ok'] else 'FAIL'}] {out.name}: scheme={spec['scheme']} wall={wall:.1f}s "
              f"tasks={m.get('tasksGenerated')} TCR={m.get('tcr', float('nan')):.3f} "
              f"delay={m.get('meanDelay', float('nan'))} PDR={m.get('packetDropRatio')} {summary.get('accounting_msg', '')}")
        if not summary["ok"] and "error_tail" in summary:
            print(summary["error_tail"][-1500:])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", help="JSON file with the run spec")
    ap.add_argument("--out", required=True)
    for k, v in DEFAULTS.items():
        if k in ("params",):
            continue
        t = type(v)
        if t is bool:
            ap.add_argument(f"--{k.replace('_', '-')}", action="store_true", default=None)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=t, default=None)
    ap.add_argument("--set", action="append", default=[], help="ini override KEY=VALUE (repeatable)")
    ap.add_argument("--keep-scenario", action="store_true")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text()) if a.spec else {}
    for k in DEFAULTS:
        v = getattr(a, k, None)
        if v is not None and k != "params":
            spec[k] = v
    params = dict(spec.get("params") or {})
    for kv in a.set:
        k, v = kv.split("=", 1)
        params[k.strip()] = v.strip()
    spec["params"] = params
    s = run(spec, Path(a.out), keep_scenario=a.keep_scenario)
    sys.exit(0 if s["ok"] else 1)


if __name__ == "__main__":
    main()
