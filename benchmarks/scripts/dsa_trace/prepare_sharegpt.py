from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Any


def _extract_prompt(conv: dict[str, Any]) -> str | None:
    # ShareGPT format varies; best effort extraction.
    if "conversations" in conv and isinstance(conv["conversations"], list):
        parts: list[str] = []
        for msg in conv["conversations"]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("from", msg.get("role", "user")))
            text = msg.get("value", msg.get("content", ""))
            if text is None:
                continue
            parts.append(f"{role}: {str(text).strip()}")
        prompt = "\n".join([p for p in parts if p])
        return prompt.strip() if prompt.strip() else None
    if "text" in conv and isinstance(conv["text"], str):
        return conv["text"].strip() or None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to ShareGPT JSON file")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        items = data["data"]
    else:
        items = data

    if not isinstance(items, list):
        raise ValueError("Unsupported ShareGPT format: expected a list.")

    rng.shuffle(items)
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for conv in items:
            if written >= args.num_prompts:
                break
            if not isinstance(conv, dict):
                continue
            prompt = _extract_prompt(conv)
            if not prompt:
                continue
            record = {
                "request_id": str(uuid.uuid4()),
                "prompt": prompt,
                "max_tokens": int(args.max_tokens),
                "dataset": "sharegpt",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} prompts to {out_path}")


if __name__ == "__main__":
    main()

