"""Learning-based baselines implemented in numpy (no torch needed at simulation time).

Aligned with the original papers as far as their descriptions allow (see docs/BASELINES.md):
  B1  DRLO-VANET (Sci. Rep. 16:10582, 2026): DQN chooses local execution or one of the RSUs; the
      paper's SAC split-ratio head is approximated by discrete split levels {0,30,50,70}% (joint
      action). Reward = deadline-aware latency term - handover (re-offload) penalty.
  B4  FedMAGS (arXiv:2605.18437): PPO learner per MEC server (RSU region), federated first-order
      meta-learning (FedAvg of parameter deltas + meta step) and fast online adaptation (few local
      PPO updates) at evaluation. The DAG/GAT/Seq2Seq part does not apply to single/two-part tasks.
  B5  MADSASPG (J. Grid Computing 23:13, 2025): multi-agent distributed deterministic policy gradient
      with a centralised, action-semantic critic: every vehicle is an agent with a shared actor; the
      critic sees the agent's state, its action vector, the *semantics* of the action (features of the
      chosen target) and the recent joint action of the other agents in the same RSU region.
Reward from the simulator: in time -> 1 - 0.5*delay/deadline, late -> -0.5, dropped -> -1.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

K_MAX = 5
SLOT_FEATS = 12
SPLITS = (0, 30, 50, 70)


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


def slot_semantics(x: np.ndarray, k: int = K_MAX) -> np.ndarray:
    """(k+1) x SLOT_FEATS matrix: features of each action's target (row 0 = local: zeros + flag)."""
    base = 10
    m = np.zeros((k + 1, SLOT_FEATS), np.float32)
    m[0, 0] = -1.0  # local marker
    m[1:] = x[base:base + k * SLOT_FEATS].reshape(k, SLOT_FEATS)
    return m


# ---------------------------------------------------------------------------------------------- MLP
class MLP:
    """Tiny MLP with Adam; backprop returns parameter grads and the input gradient."""

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
        self.W = [np.asarray(p, np.float32).copy() for p in ps[:n]]
        self.b = [np.asarray(p, np.float32).copy() for p in ps[n:]]

    def forward(self, x):
        hs = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
            hs.append(h)
        return h, hs

    def backprop(self, hs, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ d
            gb[i] = d.sum(0)
            d = d @ self.W[i].T
            if i > 0:
                d = d * (hs[i] > 0)
        return gW + gb, d

    def backward(self, hs, dout):
        g, dx = self.backprop(hs, dout)
        self.step(g)
        return dx

    def step(self, grads, clip=5.0):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params(), grads)):
            g = np.clip(g, -clip, clip)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def masked_softmax(z, mask):
    z = np.where(mask, z, -1e9)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z) * mask
    return p / p.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------------------------- B1
class DQNAgent:
    """Double DQN with replay and target network (B1 uses a joint (slot, split) action space)."""

    def __init__(self, dim, n_actions, seed=0, lr=1e-3, gamma=0.5, buffer=20000, batch=64, target_sync=200,
                 eps_start=1.0, eps_end=0.05, eps_decay=4000):
        self.rng = np.random.default_rng(seed)
        self.q = MLP([dim, 64, 64, n_actions], self.rng, lr)
        self.tq = MLP([dim, 64, 64, n_actions], self.rng, lr)
        self.tq.set_params(self.q.params())
        self.n = n_actions
        self.gamma, self.batch, self.target_sync = gamma, batch, target_sync
        self.buf = deque(maxlen=buffer)
        self.steps = 0
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay

    def eps(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps / self.eps_decay)

    def act(self, x, mask, explore):
        if explore and self.rng.random() < self.eps():
            return int(self.rng.choice(np.flatnonzero(mask)))
        q, _ = self.q.forward(x[None])
        return int(np.argmax(np.where(mask, q[0], -1e9)))

    def store(self, s, m, a, r, s2, m2, done):
        self.buf.append((s, m, a, r, s2, m2, done))

    def learn(self):
        self.steps += 1
        if len(self.buf) < self.batch:
            return
        idx = self.rng.integers(0, len(self.buf), self.batch)
        b = [self.buf[i] for i in idx]
        S = np.stack([t[0] for t in b])
        A = np.array([t[2] for t in b])
        R = np.array([t[3] for t in b], np.float32)
        S2 = np.stack([t[4] for t in b])
        M2 = np.stack([t[5] for t in b])
        D = np.array([t[6] for t in b], np.float32)
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


def b1_mask(ctx, slot_mask, k):
    """DRLO-VANET action space: local or an RSU, each with a split level (split only if splittable)."""
    cands = ctx.get("cands", [])[:k]
    split_ok = bool(ctx["task"].get("split"))
    m = np.zeros((k + 1) * len(SPLITS), bool)
    for slot in range(k + 1):
        if not slot_mask[slot]:
            continue
        if slot > 0 and cands[slot - 1]["kind"] != "rsu":
            continue
        for j in range(len(SPLITS)):
            if j == 0 or (slot > 0 and split_ok):
                m[slot * len(SPLITS) + j] = True
    return m


# ---------------------------------------------------------------------------------------------- B4
class PPOAgent:
    """PPO with clipped surrogate (one-step decisions: advantage = r - V(s)), mini-batch epochs."""

    def __init__(self, dim, n_actions, seed=0, lr=3e-3, clip=0.2, epochs=4, batch=64, minibatch=32, ent=0.01):
        self.rng = np.random.default_rng(seed)
        self.pi = MLP([dim, 64, 64, n_actions], self.rng, lr)
        self.v = MLP([dim, 64, 1], self.rng, lr)
        self.n, self.clip, self.epochs, self.batch, self.minibatch, self.ent = n_actions, clip, epochs, batch, minibatch, ent
        self.buf = []
        self.steps = 0

    def act(self, x, mask, explore):
        z, _ = self.pi.forward(x[None])
        p = masked_softmax(z, mask[None])[0]
        a = int(self.rng.choice(self.n, p=p)) if explore else int(np.argmax(p))
        return a, float(np.log(p[a] + 1e-12))

    def store(self, x, mask, a, logp, r):
        self.buf.append((x, mask, a, logp, r))
        if len(self.buf) >= self.batch:
            self.update()

    def update(self):
        X = np.stack([t[0] for t in self.buf])
        M = np.stack([t[1] for t in self.buf])
        A = np.array([t[2] for t in self.buf])
        LP = np.array([t[3] for t in self.buf], np.float32)
        R = np.array([t[4] for t in self.buf], np.float32)
        self.buf = []
        v, _ = self.v.forward(X)
        adv = R - v[:, 0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        n = len(R)
        for _ in range(self.epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, self.minibatch):
                i = perm[s:s + self.minibatch]
                m = len(i)
                z, hs = self.pi.forward(X[i])
                p = masked_softmax(z, M[i])
                ratio = np.exp(np.log(p[np.arange(m), A[i]] + 1e-12) - LP[i])
                active = ((adv[i] >= 0) & (ratio < 1 + self.clip)) | ((adv[i] < 0) & (ratio > 1 - self.clip))
                onehot = np.zeros_like(p)
                onehot[np.arange(m), A[i]] = 1.0
                # d(-ratio*A)/dz = -A*ratio*(onehot - p) where the clip is inactive
                g = -(adv[i] * ratio * active)[:, None] * (onehot - p)
                logp = np.log(p + 1e-12) * M[i]
                H = -(p * logp).sum(1, keepdims=True)
                g += self.ent * p * (logp + H) * M[i]  # entropy bonus
                self.pi.backward(hs, (g / m).astype(np.float32))
                vv, hv = self.v.forward(X[i])
                self.v.backward(hv, ((vv[:, 0] - R[i]) / m)[:, None].astype(np.float32))
        self.steps += 1

    def params(self):
        return self.pi.params() + self.v.params()

    def set_params(self, ps):
        n = len(self.pi.params())
        self.pi.set_params(ps[:n])
        self.v.set_params(ps[n:])

    def state(self):
        return {"params": [p.tolist() for p in self.params()], "steps": self.steps}

    def load(self, st):
        self.set_params([np.asarray(p, np.float32) for p in st["params"]])
        self.steps = int(st.get("steps", 0))


# ---------------------------------------------------------------------------------------------- B5
class MADSASPGAgent:
    """Shared deterministic actor + centralised action-semantic critic (MADDPG-style, one-step)."""

    def __init__(self, dim, n_actions, seed=0, lr_actor=1e-3, lr_critic=2e-3, batch=64, buffer=20000, tau=0.5,
                 eps_start=0.5, eps_end=0.02, eps_decay=4000):
        self.rng = np.random.default_rng(seed)
        self.n = n_actions
        self.cin = dim + n_actions + SLOT_FEATS + n_actions  # state, action, action semantics, others' joint action
        self.actor = MLP([dim, 64, n_actions], self.rng, lr_actor)
        self.critic = MLP([self.cin, 64, 64, 1], self.rng, lr_critic)
        self.batch, self.tau = batch, tau
        self.buf = deque(maxlen=buffer)
        self.steps = 0
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay

    def eps(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps / self.eps_decay)

    def policy(self, X, M):
        z, hs = self.actor.forward(X)
        return masked_softmax(z / self.tau, M), hs

    def act(self, x, mask, explore):
        a_vec, _ = self.policy(x[None], mask[None])
        if explore and self.rng.random() < self.eps():
            return int(self.rng.choice(np.flatnonzero(mask))), a_vec[0]
        return int(np.argmax(a_vec[0])), a_vec[0]

    def critic_in(self, X, A_vec, others):
        sem = np.einsum("bk,bkf->bf", A_vec, np.stack([slot_semantics(x, self.n - 1) for x in X]))
        return np.concatenate([X, A_vec, sem, others], 1).astype(np.float32)

    def store(self, x, mask, a, r, others):
        self.buf.append((x, mask, a, r, others))
        self.steps += 1
        if len(self.buf) >= self.batch and self.steps % 4 == 0:
            self.learn()

    def learn(self):
        idx = self.rng.integers(0, len(self.buf), self.batch)
        b = [self.buf[i] for i in idx]
        X = np.stack([t[0] for t in b])
        M = np.stack([t[1] for t in b])
        A = np.array([t[2] for t in b])
        R = np.array([t[3] for t in b], np.float32)
        O = np.stack([t[4] for t in b])
        n = len(R)
        onehot = np.zeros((n, self.n), np.float32)
        onehot[np.arange(n), A] = 1.0
        # critic: Q(s, a, sem(a), others) -> r (one-step return)
        q, hq = self.critic.forward(self.critic_in(X, onehot, O))
        self.critic.backward(hq, ((q[:, 0] - R) / n)[:, None].astype(np.float32))
        # actor: deterministic policy gradient  dQ/da * da/dz
        a_vec, ha = self.policy(X, M)
        q, hq = self.critic.forward(self.critic_in(X, a_vec, O))
        _, dcin = self.critic.backprop(hq, -np.ones((n, 1), np.float32) / n)  # maximise Q
        d = self.dim_slices(dcin, X)
        # softmax Jacobian: dz = (p * (d - sum(p*d))) / tau
        dz = a_vec * (d - (a_vec * d).sum(1, keepdims=True)) / self.tau
        self.actor.backward(ha, (dz * M).astype(np.float32))

    def dim_slices(self, dcin, X):
        dim = X.shape[1]
        d_a = dcin[:, dim:dim + self.n]
        d_sem = dcin[:, dim + self.n:dim + self.n + SLOT_FEATS]
        sem_mats = np.stack([slot_semantics(x, self.n - 1) for x in X])  # b x n x F
        return d_a + np.einsum("bf,bkf->bk", d_sem, sem_mats)

    def state(self):
        return {"actor": [p.tolist() for p in self.actor.params()], "critic": [p.tolist() for p in self.critic.params()], "steps": self.steps}

    def load(self, st):
        self.actor.set_params([np.asarray(p, np.float32) for p in st["actor"]])
        self.critic.set_params([np.asarray(p, np.float32) for p in st["critic"]])
        self.steps = int(st.get("steps", 0))


# ---------------------------------------------------------------------------------------------- manager
class RLManager:
    """Holds the agents of one simulation run and maps side-car requests onto them."""

    def __init__(self, scheme: str, mode: str = "eval", weights: str | None = None, seed: int = 0, k: int = K_MAX,
                 handover_penalty: float = 0.3):
        self.scheme, self.mode, self.k = scheme, mode, k
        self.train = mode == "train"
        self.weights = Path(weights) if weights else None
        self.multi = scheme == "b5"
        self.handover_penalty = handover_penalty
        x0, _ = featurize({"task": {"bytes": 0, "cycles": 0, "budget_ms": 1, "split": False}, "self": {"cpu_ghz": 1, "backlog_ms": 0}, "cands": []},
                          k, self.multi)
        self.dim = x0.shape[0]
        if scheme == "b1":
            self.agent = DQNAgent(self.dim, (k + 1) * len(SPLITS), seed, eps_start=1.0 if self.train else 0.0)
        elif scheme == "b4":
            self.agent = PPOAgent(self.dim, k + 1, seed)
        elif scheme == "b5":
            self.agent = MADSASPGAgent(self.dim, k + 1, seed)
        else:
            raise ValueError(scheme)
        self.regions: dict[int, PPOAgent] = {}
        self.region_actions: dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        self.pending: dict[int, dict] = {}
        self.last: dict[int, dict] = {}
        self.decisions = 0
        self.fed_rounds = 0
        if self.weights and self.weights.exists():
            self.load(self.weights)

    # ------------------------------------------------------------------ B4 federated meta-learning
    def _region_agent(self, region):
        if region not in self.regions:
            a = PPOAgent(self.dim, self.k + 1, seed=int(region) % 1000, lr=3e-3 if self.train else 1e-3)
            a.set_params(self.agent.params())  # meta-initialisation from the global model
            self.regions[region] = a
        return self.regions[region]

    def fedavg(self, region=None, beta=0.5):
        """First-order federated meta step: theta <- theta + beta * mean_k(theta_k - theta)."""
        if self.scheme != "b4" or not self.regions:
            return 0
        g = self.agent.params()
        deltas = [[pk - gk for pk, gk in zip(a.params(), g)] for a in self.regions.values()]
        new = [gk + beta * np.mean([d[i] for d in deltas], axis=0) for i, gk in enumerate(g)]
        self.agent.set_params(new)
        for a in self.regions.values():
            a.set_params(new)
        self.fed_rounds += 1
        return len(deltas)

    def _others(self, region, vid):
        hist = [a for v, a in self.region_actions[region] if v != vid]
        o = np.zeros(self.k + 1, np.float32)
        for a in hist:
            o[a] += 1
        return o / max(1, len(hist))

    # ------------------------------------------------------------------ protocol
    def decide(self, req):
        ctx = req["ctx"]
        x, mask = featurize(ctx, self.k, self.multi)
        vid, tid, region = int(req["vid"]), int(req["tid"]), int(req.get("region", -1))
        info = {"x": x, "vid": vid, "region": region}
        split = 0
        if self.scheme == "b1":
            jm = b1_mask(ctx, mask, self.k)
            a = self.agent.act(x, jm, self.train)
            slot, split = divmod(a, len(SPLITS))
            split = SPLITS[split]
            prev = self.last.pop(vid, None)  # chain transitions per vehicle (s, a, r, s')
            if prev is not None:
                self.agent.store(prev["x"], prev["mask"], prev["a"], prev["r"], x, jm, False)
                if self.train:
                    self.agent.learn()
            info.update(mask=jm, a=a)
        elif self.scheme == "b4":
            agent = self._region_agent(region)
            a, logp = agent.act(x, mask, True)  # PPO stays stochastic; adapts online (fast adaptation)
            slot = a
            info.update(mask=mask, a=a, logp=logp)
        else:  # b5
            a, _ = self.agent.act(x, mask, self.train)
            slot = a
            info.update(mask=mask, a=a, others=self._others(region, vid))
            self.region_actions[region].append((vid, a))
        self.pending[tid] = info
        self.decisions += 1
        return {"ok": True, "slot": int(slot), "split": int(split), "raw": f"target={slot}; split={split}", "n_in": 0, "n_out": 0, "conf": None}

    def feedback(self, req):
        d = self.pending.pop(int(req["tid"]), None)
        if d is None:
            return {"ok": False}
        r = float(req["reward"])
        if self.scheme == "b1":
            r -= self.handover_penalty * float(req.get("reoffloaded", 0))  # DRLO-VANET handover cost
            d["r"] = r
            self.last[d["vid"]] = d
        elif self.scheme == "b4":
            self._region_agent(d["region"]).store(d["x"], d["mask"], d["a"], d["logp"], r)
        else:
            if self.train:
                self.agent.store(d["x"], d["mask"], d["a"], r, d["others"])
        return {"ok": True}

    def close(self):
        if self.scheme == "b1":
            for d in self.last.values():
                self.agent.store(d["x"], d["mask"], d["a"], d["r"], d["x"], d["mask"], True)
        if self.train and self.weights:
            if self.scheme == "b4":
                self.fedavg()
            self.save(self.weights)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"scheme": self.scheme, "state": self.agent.state(), "fed_rounds": self.fed_rounds}))
        tmp.replace(path)

    def load(self, path):
        st = json.loads(Path(path).read_text())
        if st.get("scheme") != self.scheme:
            raise ValueError(f"weights {path} are for {st.get('scheme')}, not {self.scheme}")
        self.agent.load(st["state"])
