from __future__ import annotations

import argparse
import csv
import json
import random
import uuid
from pathlib import Path
from typing import Any


def _get_first(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to BurstGPT CSV file")
    ap.add_argument("--output", required=True, help="Output workload JSONL path")
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--arrival-time-col",
        default="timestamp",
        help="Arrival time column name (best effort; default: timestamp)",
    )
    ap.add_argument(
        "--input-len-col",
        default="prompt_tokens",
        help="Prompt tokens column name (best effort; default: prompt_tokens)",
    )
    ap.add_argument(
        "--output-len-col",
        default="completion_tokens",
        help="Completion tokens column name (best effort; default: completion_tokens)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    rows: list[dict[str, str]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rng.shuffle(rows)
    rows = rows[: args.num_requests]

    # Normalize arrivals into offsets (seconds).
    ts_list: list[float] = []
    for r in rows:
        ts = _get_first(r, [args.arrival_time_col, "time", "arrival_time", "arrival_ts"], None)
        try:
            ts_list.append(float(ts))
        except Exception:
            ts_list.append(float(len(ts_list)))
    t0 = min(ts_list) if ts_list else 0.0

    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for r, ts in zip(rows, ts_list):
            try:
                in_len = int(float(_get_first(r, [args.input_len_col, "input_tokens", "prompt_len"], 0)))
            except Exception:
                in_len = 0
            try:
                out_len = int(float(_get_first(r, [args.output_len_col, "output_tokens", "response_tokens"], 0)))
            except Exception:
                out_len = 0

            record = {
                "request_id": str(uuid.uuid4()),
                "arrival_offset_s": float(ts - t0),
                "input_len_tokens": int(in_len),
                "output_len_tokens": int(out_len),
                "dataset": "burstgpt",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} workload records to {out_path}")


if __name__ == "__main__":
    main()

