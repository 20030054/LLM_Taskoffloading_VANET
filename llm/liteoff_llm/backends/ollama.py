"""Ollama HTTP backend with constrained, deterministic decoding and logprob-based confidence."""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field

DEFAULT_URL = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")


@dataclass
class LLMResult:
    raw: str
    n_in: int
    n_out: int
    conf: float | None  # probability of the chosen target token (None if unavailable)
    margin: float | None  # p(top1) - p(top2) at the target token
    wall_ms: float
    prompt_eval_ms: float = 0.0
    eval_ms: float = 0.0
    source: str = "live"
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {"raw": self.raw, "n_in": self.n_in, "n_out": self.n_out, "conf": self.conf, "margin": self.margin,
                "wall_ms": self.wall_ms, "prompt_eval_ms": self.prompt_eval_ms, "eval_ms": self.eval_ms}

    @classmethod
    def from_dict(cls, d, source="replay"):
        return cls(raw=d["raw"], n_in=int(d["n_in"]), n_out=int(d["n_out"]), conf=d.get("conf"), margin=d.get("margin"),
                   wall_ms=float(d.get("wall_ms", 0)), prompt_eval_ms=float(d.get("prompt_eval_ms", 0)),
                   eval_ms=float(d.get("eval_ms", 0)), source=source)


def _post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def target_confidence(logprobs):
    """Probability (and top1-top2 margin) of the first digit token after 'target='."""
    if not logprobs:
        return None, None
    text = ""
    for i, tok in enumerate(logprobs):
        before = text
        text += tok.get("token", "")
        # first token that carries a digit after "target" has been seen
        if re.search(r"target\s*[=:]\s*$", before, re.I) or (re.search(r"target\s*[=:]", before, re.I) and re.search(r"\d", tok.get("token", ""))):
            if not re.search(r"\d", tok.get("token", "")):
                continue
            p1 = math.exp(tok["logprob"])
            tops = sorted((math.exp(t["logprob"]) for t in tok.get("top_logprobs", [])), reverse=True)
            p2 = tops[1] if len(tops) > 1 else 0.0
            return p1, p1 - p2
    return None, None


class OllamaBackend:
    def __init__(self, url=DEFAULT_URL, num_predict=12, temperature=0.0, seed=0, num_ctx=2048, num_thread=None,
                 timeout=120, logprobs=True, keep_alive="24h"):
        self.url = url.rstrip("/")
        self.opts = {"temperature": temperature, "num_predict": num_predict, "seed": seed, "num_ctx": num_ctx, "top_k": 1}
        if num_thread:
            self.opts["num_thread"] = int(num_thread)
        self.timeout = timeout
        self.logprobs = logprobs
        self.keep_alive = keep_alive

    def available(self):
        try:
            urllib.request.urlopen(self.url + "/api/version", timeout=2).read()
            return True
        except Exception:
            return False

    def models(self):
        d = json.loads(urllib.request.urlopen(self.url + "/api/tags", timeout=5).read())
        return [m["name"] for m in d.get("models", [])]

    def chat(self, model: str, system: str, prompt: str, num_predict=None, fmt=None) -> LLMResult:
        """fmt: optional JSON schema -> grammar-constrained decoding (always parseable output)."""
        opts = dict(self.opts)
        if num_predict:
            opts["num_predict"] = num_predict
        payload = {"model": model, "stream": False, "keep_alive": self.keep_alive, "options": opts,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if fmt is not None:
            payload["format"] = fmt
        if self.logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        t0 = time.perf_counter()
        d = _post(self.url + "/api/chat", payload, self.timeout)
        wall = (time.perf_counter() - t0) * 1000
        raw = d.get("message", {}).get("content", "")
        conf, margin = target_confidence(d.get("logprobs"))
        n_eval = int(d.get("prompt_eval_count", 0))
        n_cached = int(d.get("prompt_eval_cached_count", 0) or 0)
        return LLMResult(raw=raw, n_in=n_eval + n_cached, n_out=int(d.get("eval_count", 0)), conf=conf, margin=margin, wall_ms=wall,
                         prompt_eval_ms=d.get("prompt_eval_duration", 0) / 1e6, eval_ms=d.get("eval_duration", 0) / 1e6,
                         extra={"n_eval": n_eval, "n_cached": n_cached})

    def unload(self, model):
        try:
            _post(self.url + "/api/generate", {"model": model, "keep_alive": 0}, 30)
        except Exception:
            pass
