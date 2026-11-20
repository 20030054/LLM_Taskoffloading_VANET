"""Side-car decision service for the OMNeT++ simulation (JSON lines over TCP).

Protocol (one JSON object per line, one reply line per request):
  {"op":"hello", "scheme":..., "rlMode":..., "rlWeights":..., "seed":...}      -> {"ok":true}
  {"op":"decide","scheme":"liteoff|b2|b3|b1|b4|b5","model":...,"vid":..,"tid":..,"region":..,"ctx":{...}}
        -> {"ok":bool,"raw":"target=2; split=0","slot":2,"split":0,"conf":0.93|null,"n_in":..,"n_out":..,
            "wall_ms":..,"source":"live|replay"}            (or {"error":"..."})
  {"op":"feedback","scheme":..,"tid":..,"reward":..}  -> RL learning signal
  {"op":"fedavg","region":..}                          -> B4 federated averaging round
  {"op":"bye"}                                         -> flush (saves RL weights in train mode)

LLM backends: live Ollama, replay store (SQLite), or both (replay first, live on miss, store result).
The simulator charges the *modelled* latency (profiling/latency_model.csv) from n_in/n_out, so the
results do not depend on the wall-clock speed of the machine running the side-car.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import sys
import time
from pathlib import Path

from .backends import OllamaBackend, ReplayStore
from .backends.ollama import DEFAULT_URL, LLMResult
from .encoder import SYSTEM_PROMPT, encode, parse_answer
from .rl import RLManager

LLM_SCHEMES = {"liteoff", "b2", "b3"}
RL_SCHEMES = {"b1", "b4", "b5"}


class Service:
    def __init__(self, a):
        self.a = a
        self.backend = OllamaBackend(a.ollama_url, num_predict=a.num_predict, num_thread=a.num_thread, logprobs=not a.no_logprobs)
        self.replay = None
        if a.replay_db:
            self.replay = ReplayStore(a.replay_db, readonly=(a.replay_mode == "read"))
        self.model_map = dict(kv.split("=", 1) for kv in a.model_map.split(",") if "=" in kv) if a.model_map else {}
        self.rl: RLManager | None = None
        self.hello = {}
        self.stats = {"decide": 0, "live": 0, "replay": 0, "errors": 0, "malformed": 0, "wall_ms": 0.0}
        self.log = open(Path(a.workdir) / "decisions.jsonl", "w") if a.log_decisions else None

    # --------------------------------------------------------------------------------------------
    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "hello":
            self.hello = req
            scheme = req.get("scheme", "")
            if scheme in RL_SCHEMES:
                weights = req.get("rlWeights") or self.a.rl_weights
                self.rl = RLManager(scheme, req.get("rlMode", "eval"), weights or None, seed=int(req.get("seed", 0) or 0))
            return {"ok": True}
        if op == "decide":
            self.stats["decide"] += 1
            scheme = req.get("scheme", "")
            if scheme in RL_SCHEMES:
                if self.rl is None:
                    self.rl = RLManager(scheme, req.get("mode", "eval"), self.a.rl_weights or None)
                return self.rl.decide(req)
            return self.decide_llm(req)
        if op == "feedback":
            return self.rl.feedback(req) if self.rl else {"ok": False}
        if op == "fedavg":
            n = self.rl.fedavg(req.get("region")) if self.rl else 0
            return {"ok": True, "regions": n}
        if op == "bye":
            self.close()
            return {"ok": True, "stats": self.stats}
        return {"error": f"unknown op {op}"}

    def decide_llm(self, req):
        ctx = req["ctx"]
        model = self.model_map.get(req.get("model", ""), req.get("model", ""))
        prompt = encode(ctx)
        res = None
        if self.replay and self.a.replay_mode in ("read", "rw"):
            v = self.replay.get(model, SYSTEM_PROMPT, prompt)
            if v is not None:
                res = LLMResult.from_dict(v)
                self.stats["replay"] += 1
        if res is None:
            if self.a.replay_mode == "read":
                self.stats["errors"] += 1
                return {"error": "replay_miss"}
            try:
                res = self.backend.chat(model, SYSTEM_PROMPT, prompt)
            except Exception as e:  # noqa: BLE001
                self.stats["errors"] += 1
                return {"error": f"backend: {e}"}
            self.stats["live"] += 1
            self.stats["wall_ms"] += res.wall_ms
            if self.replay and self.a.replay_mode == "rw":
                self.replay.put(model, SYSTEM_PROMPT, prompt, res.to_dict())
        ok, slot, split = parse_answer(res.raw, k=len(ctx.get("cands", [])))
        if not ok:
            self.stats["malformed"] += 1
        out = {"ok": ok, "raw": res.raw, "slot": slot, "split": split, "conf": res.conf, "n_in": res.n_in, "n_out": res.n_out,
               "wall_ms": round(res.wall_ms, 2), "source": res.source}
        if self.log:
            self.log.write(json.dumps({"tid": req.get("tid"), "model": model, "scheme": req.get("scheme"), "prompt": prompt, **out}) + "\n")
        return out

    def close(self):
        if self.rl:
            self.rl.close()
        if self.log:
            self.log.flush()
        stats_path = Path(self.a.workdir) / "sidecar_stats.json"
        s = dict(self.stats)
        if self.replay:
            s["replay_hits"], s["replay_misses"] = self.replay.hits, self.replay.misses
        stats_path.write_text(json.dumps(s, indent=1))


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        svc: Service = self.server.svc
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                rep = svc.handle(json.loads(line))
            except Exception as e:  # noqa: BLE001
                rep = {"error": f"{type(e).__name__}: {e}"}
            self.wfile.write((json.dumps(rep, separators=(",", ":")) + "\n").encode())
            self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--ollama-url", default=DEFAULT_URL)
    ap.add_argument("--num-predict", type=int, default=12)
    ap.add_argument("--num-thread", type=int, default=int(os.environ.get("LITEOFF_LLM_THREADS", "0")) or None)
    ap.add_argument("--no-logprobs", action="store_true")
    ap.add_argument("--replay-db", default=os.environ.get("LITEOFF_REPLAY_DB", ""))
    ap.add_argument("--replay-mode", choices=["off", "read", "rw"], default="rw")
    ap.add_argument("--backend", choices=["ollama", "replay"], default="ollama", help="replay == --replay-mode read")
    ap.add_argument("--model-map", default=os.environ.get("LITEOFF_MODEL_MAP", ""), help="simname=ollamatag,...")
    ap.add_argument("--rl-weights", default="")
    ap.add_argument("--log-decisions", action="store_true")
    a = ap.parse_args(argv)
    if a.backend == "replay":
        a.replay_mode = "read"
    svc = Service(a)
    srv = Server((a.host, a.port), Handler)
    srv.svc = svc

    def stop(*_):
        svc.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"sidecar listening on {a.host}:{a.port} ({time.strftime('%H:%M:%S')})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
