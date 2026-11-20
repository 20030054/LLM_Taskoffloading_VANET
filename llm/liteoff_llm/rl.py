"""Learning-based baselines implemented in numpy (no torch needed at simulation time).

Re-implementations are *simplified* (the original papers' code is not available); see docs/BASELINES.md.
  B1  DRL offloading (DRLO-VANET style): Double DQN over [task, self, K candidate slots] -> target slot.
  B4  FedMAGS style federated meta-DRL: one DQN learner per RSU region, FedAvg every fedPeriod
      (triggered by the RSUs in the simulator), meta-initialisation from the global model and fast
      online adaptation at evaluation time.
  B5  Multi-agent DRL (MADSASPG style): shared-parameter actor-critic, every vehicle is an agent,
      state augmented with neighbourhood features (candidate count, mean candidate load).
Reward: in time -> 1 - 0.5 * delay/deadline, late -> -0.5, dropped -> -1 (sent by the simulator).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

K_MAX = 5
SLOT_FEATS = 12


def featurize(ctx: dict, k: int = K_MAX, multi_agent=False) -> tuple[np.ndarray, np.ndarray]:
    t, s = ctx["task"], ctx["self"]
    f = [float(t["bytes"]) / 1e6, float(t["cycles"]) / 2e9, float(t["budget_ms"]) / 2000.0, float(t.get("prio", 0)) / 2.0,
         1.0 if t.get("split") else 0.0, float(s["cpu_ghz"]) / 3.0, min(float(s["backlog_ms"]), 3000) / 1000.0,
         float(s.get("load", 0)), float(s.get("cbr", 0)), min(float(s.get("local_est_ms", 0)), 5000) / float(max(1.0, t["budget_ms"]))]
    cands = ctx.get("cands", [])[:k]
    mask = np.zeros(k + 1, dtype=bool)
    mask[0] = True
    for i in range(k):
        if i < len(cands):
            c = cands[i]
            kind = c["kind"]
            mask[i + 1] = True
            f += [1.0, kind == "veh", kind == "rsu", kind == "cloud", min(float(c["link_left"]), 20) / 20.0,
                  float(c["rate_mbps"]) / 27.0, np.log10(max(1.0, float(c["cpu_ghz"]) * 1e9)) / 11.0,
                  min(float(c["backlog_ms"]), 3000) / 1000.0, float(c.get("load", 0)), float(c.get("trust", 0.8)),
                  min(float(c["dist"]), 600) / 300.0, float(c.get("loss", 0))]
        else:
            f += [0.0] * SLOT_FEATS
    if multi_agent:
        loads = [float(c.get("load", 0)) for c in cands] or [0.0]
        f += [len(cands) / k, float(np.mean(loads)), float(ctx.get("road", {}).get("density", 0)) / 200.0]
    return np.asarray(f, dtype=np.float32), mask


class MLP:
    """Tiny MLP with Adam, enough for DQN / actor-critic on ~70 inputs."""

    def __init__(self, sizes, rng, lr=1e-3):
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return self.W + self.b

    def set_params(self, ps):
        n = len(self.W)
        self.W = [p.copy() for p in ps[:n]]
        self.b = [p.copy() for p in ps[n:]]

    def forward(self, x):
        hs = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
            hs.append(h)
        return h, hs

    def backward(self, hs, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (hs[i] > 0)
        self.step(gW + gb)

    def step(self, grads, clip=5.0):
        self.t += 1
        ps = self.params()
        for i, (p, g) in enumerate(zip(ps, grads)):
            g = np.clip(g, -clip, clip)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class DQNAgent:
    def __init__(self, dim, n_actions, seed=0, lr=1e-3, gamma=0.5, buffer=20000, batch=64, target_sync=200,
                 eps_start=1.0, eps_end=0.05, eps_decay=4000):
        self.rng = np.random.default_rng(seed)
        self.q = MLP([dim, 64, 64, n_actions], self.rng, lr)
        self.tq = MLP([dim, 64, 64, n_actions], self.rng, lr)
        self.tq.set_params(self.q.params())
        self.n = n_actions
        self.gamma, self.batch, self.target_sync = gamma, batch, target_sync
        self.buf = []
        self.cap = buffer
        self.steps = 0
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay

    def eps(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps / self.eps_decay)

    def act(self, x, mask, explore):
        if explore and self.rng.random() < self.eps():
            return int(self.rng.choice(np.flatnonzero(mask)))
        q, _ = self.q.forward(x[None])
        q = np.where(mask, q[0], -1e9)
        return int(np.argmax(q))

    def store(self, s, m, a, r, s2, m2, done):
        self.buf.append((s, m, a, r, s2, m2, done))
        if len(self.buf) > self.cap:
            self.buf.pop(0)

    def learn(self):
        self.steps += 1
        if len(self.buf) < self.batch:
            return
        idx = self.rng.integers(0, len(self.buf), self.batch)
        S = np.stack([self.buf[i][0] for i in idx])
        A = np.array([self.buf[i][2] for i in idx])
        R = np.array([self.buf[i][3] for i in idx], np.float32)
        S2 = np.stack([self.buf[i][4] for i in idx])
        M2 = np.stack([self.buf[i][5] for i in idx])
        D = np.array([self.buf[i][6] for i in idx], np.float32)
        q2_online, _ = self.q.forward(S2)
        a2 = np.argmax(np.where(M2, q2_online, -1e9), 1)  # double DQN
        q2_t, _ = self.tq.forward(S2)
        y = R + self.gamma * (1 - D) * q2_t[np.arange(self.batch), a2]
        q, hs = self.q.forward(S)
        d = np.zeros_like(q)
        err = q[np.arange(self.batch), A] - y
        d[np.arange(self.batch), A] = np.clip(err, -1, 1) / self.batch  # Huber gradient
        self.q.backward(hs, d)
        if self.steps % self.target_sync == 0:
            self.tq.set_params(self.q.params())

    def state(self):
        return {"params": [p.tolist() for p in self.q.params()], "steps": self.steps}

    def load(self, st):
        ps = [np.asarray(p, np.float32) for p in st["params"]]
        self.q.set_params(ps)
        self.tq.set_params(ps)
        self.steps = int(st.get("steps", 0))


class ActorCritic:
    """Shared-parameter one-step actor-critic (B5)."""

    def __init__(self, dim, n_actions, seed=0, lr=3e-3):
        self.rng = np.random.default_rng(seed)
        self.pi = MLP([dim, 64, n_actions], self.rng, lr)
        self.v = MLP([dim, 64, 1], self.rng, lr * 2)
        self.n = n_actions
        self.steps = 0

    def probs(self, x, mask):
        z, hs = self.pi.forward(x[None])
        z = np.where(mask, z[0], -1e9)
        z = z - z.max()
        p = np.exp(z) * mask
        return p / p.sum(), hs

    def act(self, x, mask, explore):
        p, _ = self.probs(x, mask)
        if explore:
            return int(self.rng.choice(self.n, p=p))
        return int(np.argmax(p))

    def update(self, x, mask, a, r, batch=32):
        """Collect one-step transitions; one Adam step per mini-batch (single-sample PG is too noisy)."""
        self.buf = getattr(self, "buf", [])
        self.buf.append((x, mask, a, r))
        if len(self.buf) < batch:
            return
        X = np.stack([t[0] for t in self.buf])
        M = np.stack([t[1] for t in self.buf])
        A = np.array([t[2] for t in self.buf])
        R = np.array([t[3] for t in self.buf], np.float32)
        self.buf = []
        self.steps += 1
        n = len(R)
        v, hv = self.v.forward(X)
        adv = R - v[:, 0]
        self.v.backward(hv, ((v[:, 0] - R) / n)[:, None].astype(np.float32))
        z, hs = self.pi.forward(X)
        z = np.where(M, z, -1e9)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) * M
        p /= p.sum(1, keepdims=True)
        g = p.copy()
        g[np.arange(n), A] -= 1.0
        g *= adv[:, None]
        # entropy bonus (maximise H): dH/dz = -p (log p + H)
        logp = np.log(p + 1e-12) * M
        H = -(p * logp).sum(1, keepdims=True)
        g += 0.01 * p * (logp + H) * M
        self.pi.backward(hs, (g / n).astype(np.float32))

    def state(self):
        return {"pi": [p.tolist() for p in self.pi.params()], "v": [p.tolist() for p in self.v.params()], "steps": self.steps}

    def load(self, st):
        self.pi.set_params([np.asarray(p, np.float32) for p in st["pi"]])
        self.v.set_params([np.asarray(p, np.float32) for p in st["v"]])
        self.steps = int(st.get("steps", 0))


class RLManager:
    """Holds the agents of one simulation run and maps side-car requests onto them."""

    def __init__(self, scheme: str, mode: str = "eval", weights: str | None = None, seed: int = 0, k: int = K_MAX):
        self.scheme, self.mode, self.k = scheme, mode, k
        self.train = mode == "train"
        self.weights = Path(weights) if weights else None
        self.multi = scheme == "b5"
        dim = featurize({"task": {"bytes": 0, "cycles": 0, "budget_ms": 1, "split": False},
                         "self": {"cpu_ghz": 1, "backlog_ms": 0}, "cands": []}, k, self.multi)[0].shape[0]
        self.dim = dim
        if scheme == "b5":
            self.agent = ActorCritic(dim, k + 1, seed)
        else:
            self.agent = DQNAgent(dim, k + 1, seed, eps_start=1.0 if self.train else 0.0)
        self.regions: dict[int, DQNAgent] = {}  # B4 local learners
        self.pending: dict[int, dict] = {}  # tid -> decision info
        self.last: dict[int, dict] = {}  # vid -> last completed transition waiting for next state (DQN)
        self.decisions = 0
        if self.weights and self.weights.exists():
            self.load(self.weights)

    # ------------------------------------------------------------------ B4 helpers
    def _region_agent(self, region):
        if self.scheme != "b4":
            return self.agent
        if region not in self.regions:
            a = DQNAgent(self.dim, self.k + 1, seed=int(region) % 1000, lr=1e-3 if self.train else 3e-4,
                         eps_start=0.3 if self.train else 0.0)
            a.q.set_params(self.agent.q.params())  # meta-initialisation from the global model
            a.tq.set_params(self.agent.q.params())
            a.steps = self.agent.steps
            self.regions[region] = a
        return self.regions[region]

    def fedavg(self, region=None):
        if self.scheme != "b4" or not self.regions:
            return 0
        ps = [a.q.params() for a in self.regions.values()]
        avg = [np.mean([p[i] for p in ps], axis=0) for i in range(len(ps[0]))]
        # Reptile-style meta step for the global model, then broadcast
        g = self.agent.q.params()
        beta = 0.5
        new = [gi + beta * (ai - gi) for gi, ai in zip(g, avg)]
        self.agent.q.set_params(new)
        self.agent.tq.set_params(new)
        for a in self.regions.values():
            a.q.set_params(new)
            a.tq.set_params(new)
        return len(ps)

    # ------------------------------------------------------------------ protocol
    def decide(self, req):
        ctx = req["ctx"]
        x, mask = featurize(ctx, self.k, self.multi)
        agent = self._region_agent(int(req.get("region", -1)))
        explore = self.train or (self.scheme == "b4")  # B4 keeps adapting online (meta-learning)
        a = agent.act(x, mask, explore and self.train)
        vid, tid = int(req["vid"]), int(req["tid"])
        # complete the previous transition of this vehicle with the new state
        prev = self.last.pop(vid, None)
        if prev is not None and isinstance(agent, DQNAgent):
            agent.store(prev["x"], prev["mask"], prev["a"], prev["r"], x, mask, False)
            if self.train or self.scheme == "b4":
                agent.learn()
        self.pending[tid] = {"x": x, "mask": mask, "a": a, "vid": vid, "region": int(req.get("region", -1))}
        self.decisions += 1
        return {"ok": True, "slot": a, "split": 0, "raw": f"target={a}; split=0", "n_in": 0, "n_out": 0, "conf": None}

    def feedback(self, req):
        d = self.pending.pop(int(req["tid"]), None)
        if d is None:
            return {"ok": False}
        r = float(req["reward"])
        if isinstance(self.agent, ActorCritic):
            if self.train:
                self.agent.update(d["x"], d["mask"], d["a"], r)
        else:
            d["r"] = r
            self.last[d["vid"]] = d
        return {"ok": True}

    def close(self):
        # flush terminal transitions
        for vid, d in list(self.last.items()):
            agent = self._region_agent(d["region"]) if self.scheme == "b4" else self.agent
            if isinstance(agent, DQNAgent):
                agent.store(d["x"], d["mask"], d["a"], d["r"], d["x"], d["mask"], True)
        if self.train and self.weights:
            if self.scheme == "b4":
                self.fedavg()
            self.save(self.weights)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"scheme": self.scheme, "state": self.agent.state()}))
        tmp.replace(path)

    def load(self, path):
        st = json.loads(Path(path).read_text())
        self.agent.load(st["state"])
