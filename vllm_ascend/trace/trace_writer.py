from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceFile:
    path: Path
    lock: threading.Lock


class JSONLTraceWriter:
    """Thread-safe JSONL writer with low overhead."""

    def __init__(self, trace_dir: str, run_id: str) -> None:
        self._trace_dir = Path(trace_dir).expanduser().resolve()
        self._run_id = run_id
        self._files: dict[str, TraceFile] = {}
        self._init_lock = threading.Lock()

    @property
    def run_dir(self) -> Path:
        return self._trace_dir / self._run_id

    def _ensure_file(self, name: str) -> TraceFile:
        with self._init_lock:
            existing = self._files.get(name)
            if existing is not None:
                return existing
            self.run_dir.mkdir(parents=True, exist_ok=True)
            path = self.run_dir / f"{name}.jsonl"
            tf = TraceFile(path=path, lock=threading.Lock())
            self._files[name] = tf
            return tf

    def write(self, name: str, record: dict[str, Any]) -> None:
        tf = self._ensure_file(name)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with tf.lock:
            with open(tf.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")


def make_run_id(prefix: str = "run") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    pid = os.getpid()
    return f"{prefix}_{ts}_{pid}"

