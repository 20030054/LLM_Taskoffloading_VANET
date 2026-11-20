#!/usr/bin/env python3
"""SUMO scenario generation for LiteOff-LLM.

Sub-commands
  nets                      build the three base networks into scenarios/nets/<map>/
  routes  --map M ...       build a per-run scenario directory (routes calibrated to a
                            target mean vehicle count, sumo.cfg, rsus.ini, meta.json)

Maps
  grid     urban Manhattan grid, 1 km x 1 km, 5x5 junctions, traffic lights, buildings
  highway  3 km, 3 lanes per direction, one on/off ramp pair, 130 km/h limit
  spider   held-out suburban "ring road + radial streets" network (generalisation test)

Coordinates: Veins maps SUMO (x, y) to OMNeT (x - minX + margin, (maxY - minY) - (y - minY) + margin)
with margin = 25 m (TraCIScenarioManager default). RSU positions are written in OMNeT coordinates.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
NETS = HERE / "nets"
SUMO_HOME = Path(os.environ.get("SUMO_HOME", "/usr/share/sumo"))
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"
MARGIN = 25.0
MAPS = ("grid", "highway", "spider")


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"command failed: {' '.join(map(str, cmd))}")
    return r


# --------------------------------------------------------------------------- nets
def build_grid(out: Path):
    run(["netgenerate", "--grid", "--grid.number", "5", "--grid.length", "250",
         "--default.lanenumber", "2", "--default.speed", "16.67",
         "--default-junction-type", "traffic_light", "--no-turnarounds", "true",
         "--output-file", str(out / "net.net.xml")])
    # buildings in every block (obstacle shadowing); road corridor of 30 m kept free
    polys = ['<additional>']
    k = 0
    for i in range(4):
        for j in range(4):
            x0, y0 = i * 250 + 18, j * 250 + 18
            x1, y1 = (i + 1) * 250 - 18, (j + 1) * 250 - 18
            shape = f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"  # open ring: Veins closes it (a repeated vertex breaks its intersection parity)
            polys.append(f'  <poly id="b{k}" type="building" color="1,0,0" fill="1" layer="-1" shape="{shape}"/>')
            k += 1
    polys.append('</additional>')
    (out / "poly.xml").write_text("\n".join(polys) + "\n")


def build_highway(out: Path):
    nodes = """<nodes>
  <node id="W" x="0" y="0"/> <node id="A" x="1000" y="0"/> <node id="B" x="2000" y="0"/>
  <node id="E" x="3000" y="0"/> <node id="RA" x="1000" y="-250"/> <node id="RB" x="2000" y="250"/>
</nodes>"""
    edges = """<edges>
  <edge id="WA" from="W" to="A" numLanes="3" speed="36.1"/> <edge id="AW" from="A" to="W" numLanes="3" speed="36.1"/>
  <edge id="AB" from="A" to="B" numLanes="3" speed="36.1"/> <edge id="BA" from="B" to="A" numLanes="3" speed="36.1"/>
  <edge id="BE" from="B" to="E" numLanes="3" speed="36.1"/> <edge id="EB" from="E" to="B" numLanes="3" speed="36.1"/>
  <edge id="RAin" from="RA" to="A" numLanes="1" speed="22.2"/> <edge id="RAout" from="A" to="RA" numLanes="1" speed="22.2"/>
  <edge id="RBin" from="RB" to="B" numLanes="1" speed="22.2"/> <edge id="RBout" from="B" to="RB" numLanes="1" speed="22.2"/>
</edges>"""
    with tempfile.TemporaryDirectory() as td:
        Path(td, "n.nod.xml").write_text(nodes)
        Path(td, "e.edg.xml").write_text(edges)
        run(["netconvert", "-n", f"{td}/n.nod.xml", "-e", f"{td}/e.edg.xml", "--no-turnarounds", "true",
             "-o", str(out / "net.net.xml")])
    (out / "poly.xml").write_text("<additional/>\n")


def build_spider(out: Path):
    run(["netgenerate", "--spider", "--spider.arm-number", "6", "--spider.circle-number", "4",
         "--spider.space-radius", "150", "--default.lanenumber", "1", "--default.speed", "16.67",
         "--default-junction-type", "priority", "--no-turnarounds", "true",
         "--output-file", str(out / "net.net.xml")])
    (out / "poly.xml").write_text("<additional/>\n")


def net_info(net: Path):
    root = ET.parse(net).getroot()
    loc = root.find("location")
    x0, y0, x1, y1 = map(float, loc.get("convBoundary").split(","))
    junctions = [(float(j.get("x")), float(j.get("y"))) for j in root.findall("junction")
                 if j.get("type") not in ("internal", "dead_end") or True]
    junctions = [(x, y) for (x, y), j in zip(junctions, root.findall("junction")) if j.get("type") != "internal"]
    length = 0.0
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        lanes = e.findall("lane")
        if lanes:
            length += float(lanes[0].get("length"))
    return (x0, y0, x1, y1), junctions, length


def cmd_nets(_args):
    builders = {"grid": build_grid, "highway": build_highway, "spider": build_spider}
    for m, fn in builders.items():
        d = NETS / m
        d.mkdir(parents=True, exist_ok=True)
        fn(d)
        (x0, y0, x1, y1), _, length = net_info(d / "net.net.xml")
        meta = {"map": m, "bounds": [x0, y0, x1, y1], "area_km2": (x1 - x0) * (y1 - y0) / 1e6,
                "road_km": length / 1000.0}
        if m == "highway":
            meta["area_km2"] = 3.0 * 0.05  # corridor, density is per km per direction
        (d / "net.json").write_text(json.dumps(meta, indent=1))
        print(f"built {m}: bounds={meta['bounds']} road_km={meta['road_km']:.2f}")


# ------------------------------------------------------------------------- routes
def to_omnet(x, y, b):
    x0, y0, x1, y1 = b
    return x - x0 + MARGIN, (y1 - y0) - (y - y0) + MARGIN


def place_rsus(m, bounds, junctions, spacing):
    x0, y0, x1, y1 = bounds
    pts = []
    if m == "highway":
        n = max(1, int(round((x1 - x0) / spacing)))
        step = (x1 - x0) / n
        pts = [(x0 + step * (i + 0.5), 12.0) for i in range(n)]
    else:
        nx = max(1, int(round((x1 - x0) / spacing)))
        ny = max(1, int(round((y1 - y0) / spacing)))
        sx, sy = (x1 - x0) / nx, (y1 - y0) / ny
        for i in range(nx):
            for j in range(ny):
                tx, ty = x0 + sx * (i + 0.5), y0 + sy * (j + 0.5)
                jx, jy = min(junctions, key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2)
                if (jx, jy) not in pts:
                    pts.append((jx, jy))
    return pts


def write_sumocfg(path: Path, net, rou, poly, end):
    path.write_text(f"""<configuration>
  <input>
    <net-file value="{net}"/>
    <route-files value="{rou}"/>
    <additional-files value="{poly}"/>
  </input>
  <time><begin value="0"/><end value="{end}"/><step-length value="0.1"/></time>
  <processing><time-to-teleport value="60"/></processing>
  <report><no-step-log value="true"/><no-warnings value="true"/><xml-validation value="never"/></report>
</configuration>
""")


def mean_running(cfg: Path, warmup: float, end: float, tmp: Path):
    summ = tmp / "summary.xml"
    run(["sumo", "-c", str(cfg), "--summary-output", str(summ), "--no-step-log", "true",
         "--no-warnings", "true", "--end", str(end)])
    vals = [float(s.get("running")) for s in ET.parse(summ).getroot().findall("step")
            if warmup <= float(s.get("time")) <= end]
    return sum(vals) / max(1, len(vals))


def make_trips(net, out_rou, period, end, seed, vmax, maptype, tmp: Path):
    vtype = tmp / "vtype.add.xml"
    sigma = "0.5"
    vtype.write_text(f'<additional><vType id="car" maxSpeed="{vmax:.2f}" speedFactor="1.0" '
                     f'speedDev="0.1" sigma="{sigma}" length="4.5" minGap="2.5"/></additional>\n')
    trips = tmp / "trips.xml"
    fringe = "50" if maptype == "highway" else "5"
    min_dist = "1500" if maptype == "highway" else "300"
    run([sys.executable, str(RANDOM_TRIPS), "-n", str(net), "-o", str(trips), "-r", str(out_rou),
         "-b", "0", "-e", str(end), "-p", f"{period:.4f}", "--seed", str(seed), "--fringe-factor", fringe,
         "--min-distance", min_dist, "--validate", "--trip-attributes", 'type="car" departLane="best" departSpeed="max"',
         "--additional-file", str(vtype), "--remove-loops"])
    # embed vType in route file so the run directory is self contained
    rroot = ET.parse(out_rou).getroot()
    if rroot.find("vType") is None:
        rroot.insert(0, ET.fromstring(vtype.read_text()).find("vType"))
        ET.ElementTree(rroot).write(out_rou)


def cmd_routes(a):
    netdir = NETS / a.map
    net = netdir / "net.net.xml"
    if not net.exists():
        raise SystemExit(f"{net} missing: run `gen_scenarios.py nets` first")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bounds, junctions, road_len = net_info(net)
    end = a.duration + 5
    rou = out / "routes.rou.xml"
    cfg = out / "run.sumo.cfg"
    write_sumocfg(cfg, net.resolve(), rou, (netdir / "poly.xml").resolve(), end)
    # initial guess: mean trip time ~ (typical route length / speed)
    typical = 2500.0 if a.map == "highway" else 900.0
    period = (typical / max(5.0, 0.8 * a.vmax)) / a.vehicles
    rng = random.Random(a.seed)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for it in range(a.calib_iters + 1):
            make_trips(net, rou, period, end, a.seed, a.vmax, a.map, tmp)
            if it == a.calib_iters:
                break
            got = mean_running(cfg, min(a.warmup, end / 2), end, tmp)
            if got <= 0:
                period *= 0.5
                continue
            ratio = got / a.vehicles
            if abs(ratio - 1) < 0.08:
                break
            period *= max(0.3, min(3.0, ratio))
        measured = mean_running(cfg, min(a.warmup, end / 2), end, tmp)
    rsus = place_rsus(a.map, bounds, junctions, a.rsu_spacing)
    x0, y0, x1, y1 = bounds
    play_x, play_y = (x1 - x0) + 2 * MARGIN, (y1 - y0) + 2 * MARGIN
    lines = [f"*.playgroundSizeX = {play_x + 100:.0f}m", f"*.playgroundSizeY = {play_y + 100:.0f}m",
             f"*.numRsu = {len(rsus)}"]
    om = []
    for i, (x, y) in enumerate(rsus):
        ox, oy = to_omnet(x, y, bounds)
        om.append([round(ox, 1), round(oy, 1)])
        lines += [f"*.rsu[{i}].mobility.x = {ox:.1f}", f"*.rsu[{i}].mobility.y = {oy:.1f}", f"*.rsu[{i}].mobility.z = 3"]
    (out / "rsus.ini").write_text("\n".join(lines) + "\n")
    meta = {"map": a.map, "vehicles_target": a.vehicles, "vehicles_measured": round(measured, 2),
            "vmax": a.vmax, "seed": a.seed, "duration": a.duration, "period": period,
            "rsu_spacing": a.rsu_spacing, "rsus_omnet": om, "bounds": bounds,
            "obstacles": a.map == "grid", "road_km": road_len / 1000}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    _ = rng


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("nets")
    r = sp.add_parser("routes")
    r.add_argument("--map", choices=MAPS, required=True)
    r.add_argument("--vehicles", type=float, required=True, help="target mean number of running vehicles")
    r.add_argument("--vmax", type=float, default=13.9, help="vehicle max speed m/s")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--duration", type=float, default=300)
    r.add_argument("--warmup", type=float, default=60)
    r.add_argument("--rsu-spacing", type=float, default=500)
    r.add_argument("--calib-iters", type=int, default=2)
    r.add_argument("--out", required=True)
    a = p.parse_args()
    {"nets": cmd_nets, "routes": cmd_routes}[a.cmd](a)


if __name__ == "__main__":
    main()
