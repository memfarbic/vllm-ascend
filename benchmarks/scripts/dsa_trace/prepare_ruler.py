from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Any


def _get_first(example: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in example and example[k] not in (None, ""):
            return example[k]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output prompts JSONL path")
    ap.add_argument("--dataset", default="allenai/ruler_data", help="Hugging Face dataset id")
    ap.add_argument("--split", default="test", help="HF split (best effort)")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--task-allowlist",
        default="",
        help="Comma-separated task_type allowlist (optional)",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("Please install `datasets` to prepare RULER prompts.") from e

    ds = load_dataset(args.dataset, split=args.split)
    items = list(ds)
    rng = random.Random(args.seed)
    rng.shuffle(items)

    allow = {x.strip() for x in args.task_allowlist.split(",") if x.strip()} if args.task_allowlist else None

    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for ex in items:
            if written >= args.num_prompts:
                break
            if not isinstance(ex, dict):
                continue

            task_type = _get_first(ex, ["task_type", "task", "type"])
            if allow is not None and task_type not in allow:
                continue

            text = _get_first(ex, ["text", "prompt", "input", "query"])
            if not isinstance(text, str) or not text.strip():
                continue

            record = {
                "request_id": str(uuid.uuid4()),
                "prompt": text.strip(),
                "max_tokens": int(args.max_tokens),
                "dataset": "ruler",
                "task_type": task_type,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} prompts to {out_path}")


if __name__ == "__main__":
    main()

