"""Synthetic contexts with the same schema as the simulator's contextToJson() (profiling and tests)."""
import random


def random_context(rng: random.Random, k: int = 5, families="mobility,link,resources,task,trust,road"):
    cls = rng.choice(["safety", "perception", "infotainment"])
    deadline = {"safety": rng.uniform(50, 150), "perception": rng.uniform(300, 500), "infotainment": rng.uniform(1000, 2000)}[cls]
    task = {"id": rng.randint(1, 10**9), "bytes": rng.uniform(2e4, 1e6), "cycles": rng.uniform(5e7, 2e9),
            "result_bytes": rng.choice([200, 2000, 10000]), "deadline_ms": deadline, "budget_ms": deadline * rng.uniform(0.7, 1.0),
            "prio": {"safety": 2, "perception": 1, "infotainment": 0}[cls], "cls": cls, "split": cls != "safety"}
    me = {"cpu_ghz": rng.uniform(1, 3), "backlog_ms": rng.uniform(0, 400), "queue": rng.randint(0, 5), "load": rng.random(),
          "energy": rng.random(), "speed": rng.uniform(0, 33), "cbr": rng.uniform(0, 0.6)}
    me["local_est_ms"] = me["backlog_ms"] + task["cycles"] / (me["cpu_ghz"] * 1e9) * 1000
    cands = []
    for i in range(k):
        kind = rng.choice(["veh", "veh", "rsu", "cloud"]) if i else rng.choice(["veh", "rsu"])
        c = {"slot": i + 1, "kind": kind, "node": rng.randint(10, 500), "dist": rng.uniform(10, 300), "rel_speed": rng.uniform(0, 40),
             "closing": rng.uniform(-20, 20), "link_left": rng.uniform(0.5, 60), "rate_mbps": rng.choice([6, 9, 12, 18]),
             "cpu_ghz": {"veh": rng.uniform(1, 3), "rsu": rng.uniform(10, 30), "cloud": 100}[kind],
             "load": rng.random(), "backlog_ms": rng.uniform(0, 300) if kind != "cloud" else 0, "queue": rng.randint(0, 6),
             "loss": rng.uniform(0, 0.3), "rssi": rng.uniform(-92, -60), "trust": rng.uniform(0.5, 1), "timeouts": rng.randint(0, 2),
             "energy": rng.random(), "backhaul_ms": 50 if kind == "cloud" else 0, "pred_dist_2s": rng.uniform(10, 350),
             "beacon_age": rng.uniform(0, 1.5), "route_overlap": rng.random() < 0.4}
        cands.append(c)
    road = {"density": rng.uniform(10, 250), "neighbors": rng.randint(0, 30), "junction_in": rng.uniform(0, 30),
            "rsu_dist": rng.uniform(20, 600), "coverage_left": rng.uniform(0, 60), "type": rng.choice(["urban", "highway", "suburban"])}
    return {"t": rng.uniform(0, 600), "vid": rng.randint(10, 500), "families": families, "task": task, "self": me, "road": road, "cands": cands}
