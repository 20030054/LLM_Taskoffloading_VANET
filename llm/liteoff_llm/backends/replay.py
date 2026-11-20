"""Replay store: SQLite cache of LLM answers keyed by (model, system, prompt).

Used to (a) make reruns deterministic and fast, (b) share answers across HPC jobs (read-only copy
per job + merge afterwards), (c) run bulk experiments without a live LLM ("replay" mode).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def key(model: str, system: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{system}\x00{prompt}".encode()).hexdigest()


class ReplayStore:
    def __init__(self, path: str | Path, readonly=False):
        self.path = str(path)
        self.readonly = readonly
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        uri = f"file:{self.path}?mode=ro" if readonly else f"file:{self.path}"
        self.db = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=60)
        if not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS answers (k TEXT PRIMARY KEY, model TEXT, prompt TEXT, value TEXT)")
            self.db.commit()
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, model, system, prompt):
        with self.lock:
            row = self.db.execute("SELECT value FROM answers WHERE k=?", (key(model, system, prompt),)).fetchone()
        if row:
            self.hits += 1
            return json.loads(row[0])
        self.misses += 1
        return None

    def put(self, model, system, prompt, value: dict):
        if self.readonly:
            return
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO answers VALUES (?,?,?,?)",
                            (key(model, system, prompt), model, prompt, json.dumps(value)))
            self.db.commit()

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]

    def merge_from(self, other_path):
        """Merge another store into this one (HPC post-processing)."""
        with self.lock:
            self.db.execute("ATTACH DATABASE ? AS o", (str(other_path),))
            self.db.execute("INSERT OR IGNORE INTO answers SELECT * FROM o.answers")
            self.db.commit()
            self.db.execute("DETACH DATABASE o")

    def close(self):
        self.db.close()
