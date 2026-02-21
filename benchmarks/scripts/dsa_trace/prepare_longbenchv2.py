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


def _format_mcq(context: str, question: str, choices: list[str]) -> str:
    lines = []
    if context.strip():
        lines.append("Context:")
        lines.append(context.strip())
        lines.append("")
    lines.append("Question:")
    lines.append(question.strip())
    lines.append("")
    lines.append("Choices:")
    for i, c in enumerate(choices):
        label = chr(ord("A") + i)
        lines.append(f"{label}. {c}")
    lines.append("")
    lines.append("Answer with the choice letter only.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output prompts JSONL path")
    ap.add_argument("--dataset", default="zai-org/LongBench-v2", help="Hugging Face dataset id")
    ap.add_argument("--split", default="test", help="HF split (best effort)")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("Please install `datasets` to prepare LongBench v2 prompts.") from e

    ds = load_dataset(args.dataset, split=args.split)
    items = list(ds)
    rng = random.Random(args.seed)
    rng.shuffle(items)

    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for ex in items:
            if written >= args.num_prompts:
                break
            if not isinstance(ex, dict):
                continue

            context = _get_first(ex, ["context", "article", "document", "input", "passage"])
            question = _get_first(ex, ["question", "query", "instruction"])
            choices = _get_first(ex, ["choices", "options"])

            if not isinstance(context, str) or not isinstance(question, str):
                continue
            if isinstance(choices, dict) and "text" in choices:
                choices = choices["text"]
            if not isinstance(choices, list) or not choices:
                # Fallback: treat as open-ended QA.
                prompt = f"Context:\n{context.strip()}\n\nQuestion:\n{question.strip()}\n\nAnswer:"
            else:
                choices_str = [str(c).strip() for c in choices if str(c).strip()]
                prompt = _format_mcq(context, question, choices_str)

            record = {
                "request_id": str(uuid.uuid4()),
                "prompt": prompt,
                "max_tokens": int(args.max_tokens),
                "dataset": "longbenchv2",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} prompts to {out_path}")


if __name__ == "__main__":
    main()

