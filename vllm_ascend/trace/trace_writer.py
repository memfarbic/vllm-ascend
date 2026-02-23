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


def _sanitize_path_component(s: str) -> str:
    out: list[str] = []
    for ch in (s or "").strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out)
    return cleaned or "unknown"


def make_file_suffix() -> str:
    """Build a stable suffix to avoid cross-process file collisions.

    Prefer rank-based suffix (stable across runs) over pid-based suffix.
    Include node identity if available.

    Node id sources (best-effort):
    - VLLM_ASCEND_NODE_ID (user-specified)
    - SLURM_NODEID (Slurm)
    - OMPI_COMM_WORLD_NODE_RANK (OpenMPI)
    """

    hostname = _sanitize_path_component(os.getenv("HOSTNAME", "") or os.uname().nodename)
    node_id = (
        os.getenv("VLLM_ASCEND_NODE_ID")
        or os.getenv("SLURM_NODEID")
        or os.getenv("OMPI_COMM_WORLD_NODE_RANK")
        or ""
    ).strip()
    node_part = f"n{_sanitize_path_component(node_id)}_" if node_id else ""

    # Best-effort rank detection (global rank preferred).
    global_rank = (
        os.getenv("RANK")
        or os.getenv("HCCL_RANK_ID")
        or os.getenv("VLLM_DP_RANK")
        or ""
    ).strip()

    local_rank = (
        os.getenv("LOCAL_RANK")
        or os.getenv("TP_RANK")
        or ""
    ).strip()

    if global_rank:
        gr = _sanitize_path_component(global_rank)
        lr = _sanitize_path_component(local_rank) if local_rank else ""
        extra = f"_lr{lr}" if lr and lr != gr else ""
        return f"{node_part}h{hostname}_r{gr}{extra}"

    if local_rank:
        lr = _sanitize_path_component(local_rank)
        pid = os.getpid()
        return f"{node_part}h{hostname}_lr{lr}_pid{pid}"

    pid = os.getpid()
    return f"{node_part}h{hostname}_pid{pid}"


class JSONLTraceWriter:
    """Thread-safe JSONL writer with low overhead."""

    def __init__(self, trace_dir: str, run_id: str, file_suffix: str | None = None) -> None:
        self._trace_dir = Path(trace_dir).expanduser().resolve()
        self._run_id = run_id
        self._file_suffix = _sanitize_path_component(file_suffix) if file_suffix else make_file_suffix()
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
            safe_name = _sanitize_path_component(name)
            path = self.run_dir / f"{safe_name}.{self._file_suffix}.jsonl"
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
