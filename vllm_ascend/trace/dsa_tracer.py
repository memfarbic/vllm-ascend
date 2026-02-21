from __future__ import annotations

import atexit
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .trace_writer import JSONLTraceWriter, make_run_id


@dataclass
class StepContext:
    step_idx: int
    req_ids: list[str]


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return default if val is None or val == "" else val


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


class DSATracer:
    def __init__(self) -> None:
        self._enabled = bool(_env_int("VLLM_ASCEND_DSA_TRACE", 0))
        self._decode_only = bool(_env_int("VLLM_ASCEND_DSA_TRACE_DECODE_ONLY", 1))
        self._sample_rate = float(_env_float("VLLM_ASCEND_DSA_TRACE_SAMPLE_RATE", 1.0))
        self._max_steps = int(_env_int("VLLM_ASCEND_DSA_TRACE_MAX_STEPS", -1))
        self._layer_filter = _env_str("VLLM_ASCEND_DSA_TRACE_LAYER_FILTER", "").strip()
        self._trace_dir = _env_str("VLLM_ASCEND_TRACE_DIR", "./trace_out")

        self._run_id = make_run_id(prefix="dsa")
        self._writer = JSONLTraceWriter(self._trace_dir, self._run_id) if self._enabled else None

        self._step_ctx: StepContext | None = None
        self._rng = random.Random(_env_int("VLLM_ASCEND_DSA_TRACE_SEED", 0))
        self._prefix_hit_tokens: dict[str, int] = {}

        if self._enabled:
            atexit.register(self._finalize)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def decode_only(self) -> bool:
        return self._decode_only

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> str:
        assert self._writer is not None
        return str(self._writer.run_dir)

    def _finalize(self) -> None:
        # Placeholder for future run-level summary.
        return

    def _should_sample(self) -> bool:
        if not self._enabled:
            return False
        if self._sample_rate >= 1.0:
            return True
        return self._rng.random() < self._sample_rate

    def _layer_allowed(self, layer_name: str) -> bool:
        if not self._layer_filter:
            return True
        # Comma-separated allowlist
        allowed = {x.strip() for x in self._layer_filter.split(",") if x.strip()}
        return layer_name in allowed

    def set_step_context(self, step_idx: int, req_ids: list[str]) -> None:
        if not self._enabled:
            return
        if self._max_steps >= 0 and step_idx >= self._max_steps:
            return
        self._step_ctx = StepContext(step_idx=step_idx, req_ids=req_ids)

    def get_batch_req_ids(self) -> list[str] | None:
        if self._step_ctx is None:
            return None
        return self._step_ctx.req_ids

    def record_topk_and_blocks(
        self,
        *,
        layer_name: str,
        attn_state: str,
        block_size: int,
        request_id: str,
        req_idx: int,
        seq_len_current: int,
        query_token_pos: int | None,
        selected_token_pos_by_head: np.ndarray,
        selected_block_ids: list[int] | None,
        unique_token_pos_count: int,
        offset_min: int | None,
        offset_median: int | None,
        offset_max: int | None,
        unique_blocks: int | None,
        tokens_per_touched_block_stats: dict[str, float] | None,
        prefix_cached_blocks: int | None = None,
        dsa_prefix_intersection_ratio: float | None = None,
        dsa_prefix_hot_blocks: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._should_sample():
            return
        if not self._layer_allowed(layer_name):
            return
        if self._step_ctx is None:
            return

        record: dict[str, Any] = {
            "type": "dsa_topk",
            "run_id": self._run_id,
            "ts_us": int(time.time() * 1e6),
            "layer_name": layer_name,
            "attn_state": attn_state,
            "step_idx": self._step_ctx.step_idx,
            "request_id": request_id,
            "req_idx": int(req_idx),
            "seq_len_current": int(seq_len_current),
            "batch_req_ids": self._step_ctx.req_ids,
            "query_token_pos": None if query_token_pos is None else int(query_token_pos),
            "block_size": block_size,
            "selected_token_pos_by_head": selected_token_pos_by_head.tolist(),
            "selected_block_ids": selected_block_ids,
            "unique_token_pos_count": int(unique_token_pos_count),
            "offset_min": None if offset_min is None else int(offset_min),
            "offset_median": None if offset_median is None else int(offset_median),
            "offset_max": None if offset_max is None else int(offset_max),
        }
        if prefix_cached_blocks is not None:
            record["prefix_cached_blocks"] = int(prefix_cached_blocks)
            record["prefix_cache_hit"] = bool(prefix_cached_blocks > 0)
        if dsa_prefix_intersection_ratio is not None:
            record["dsa_prefix_intersection_ratio"] = float(dsa_prefix_intersection_ratio)
        if dsa_prefix_hot_blocks is not None:
            record["dsa_prefix_hot_blocks"] = dsa_prefix_hot_blocks
        if unique_blocks is not None:
            record["unique_blocks"] = int(unique_blocks)
        if tokens_per_touched_block_stats is not None:
            record["tokens_per_touched_block"] = tokens_per_touched_block_stats

        assert self._writer is not None
        self._writer.write("dsa_access", record)

    def record_kv_io(
        self,
        *,
        tier: str,
        op: str,
        request_id: str,
        bytes_read: int,
        read_ops: int,
        batch_size: int | None,
        latency_us: int | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._should_sample():
            return
        if self._step_ctx is None:
            # KV IO can happen outside attention execution, but still within a model step.
            step_idx = -1
        else:
            step_idx = self._step_ctx.step_idx

        record: dict[str, Any] = {
            "type": "kv_io",
            "run_id": self._run_id,
            "ts_us": int(time.time() * 1e6),
            "step_idx": step_idx,
            "tier": tier,
            "op": op,
            "request_id": request_id,
            "bytes_read": int(bytes_read),
            "read_ops": int(read_ops),
            "batch_size": None if batch_size is None else int(batch_size),
            "latency_us": None if latency_us is None else int(latency_us),
        }
        if extra:
            record["extra"] = extra
        assert self._writer is not None
        self._writer.write("kv_io", record)

    def record_prefix_cache(
        self,
        *,
        request_id: str,
        prefix_hit_tokens: int,
        block_size: int,
        source: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._should_sample():
            return
        try:
            self._prefix_hit_tokens[str(request_id)] = int(prefix_hit_tokens)
        except Exception:
            pass
        prefix_blocks = int(prefix_hit_tokens // block_size) if block_size > 0 else 0
        record: dict[str, Any] = {
            "type": "prefix_cache",
            "run_id": self._run_id,
            "ts_us": int(time.time() * 1e6),
            "step_idx": -1 if self._step_ctx is None else self._step_ctx.step_idx,
            "request_id": request_id,
            "prefix_cache_hit": bool(prefix_hit_tokens > 0),
            "prefix_hit_tokens": int(prefix_hit_tokens),
            "block_size": int(block_size),
            "prefix_cached_blocks": int(prefix_blocks),
            "source": source,
        }
        if extra:
            record["extra"] = extra
        assert self._writer is not None
        self._writer.write("prefix_cache", record)

    def get_prefix_hit_tokens(self, request_id: str) -> int:
        return int(self._prefix_hit_tokens.get(str(request_id), 0))


_GLOBAL_TRACER: DSATracer | None = None


def get_dsa_tracer() -> DSATracer:
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = DSATracer()
    return _GLOBAL_TRACER

