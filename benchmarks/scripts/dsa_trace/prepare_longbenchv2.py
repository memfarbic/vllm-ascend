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
    choices_lines = "\n".join([f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)])
    return (
        "Context:\n"
        f"{context.strip()}\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Choices:\n"
        f"{choices_lines}\n\n"
        "Answer:"
    )


def _choose_split(ds_dict: Any) -> str:
    # Prefer eval-like splits, fallback to train, then first available.
    for name in ["test", "validation", "val", "train"]:
        if name in ds_dict:
            return name
    return next(iter(ds_dict.keys()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output prompts JSONL path")
    ap.add_argument("--dataset", default="zai-org/LongBench-v2", help="Hugging Face dataset id")
    ap.add_argument(
        "--split",
        default="auto",
        help="HF split. Use 'auto' to pick an available split (default).",
    )

    ap.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="How many requests to sample and write (use -1 for all).",
    )

    # This controls output length (max new tokens), NOT input/context length.
    ap.add_argument(
        "--max-new-tokens",
        "--max-tokens",
        dest="max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens to generate per request (completion length).",
    )

    ap.add_argument(
        "--min-context-chars",
        type=int,
        default=0,
        help="Only keep examples with context length >= this many characters.",
    )
    ap.add_argument(
        "--min-prompt-chars",
        type=int,
        default=0,
        help="Only keep examples with final prompt length >= this many characters.",
    )

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("Please install `datasets` to prepare LongBench v2 prompts.") from e

    split = str(args.split)
    if split == "auto":
        ds_dict = load_dataset(args.dataset)
        split = _choose_split(ds_dict)
        ds = ds_dict[split]
    else:
        try:
            ds = load_dataset(args.dataset, split=split)
        except Exception as e:
            # Common failure: unknown split. Provide available splits.
            try:
                ds_dict = load_dataset(args.dataset)
                available = list(ds_dict.keys())
            except Exception:
                available = []
            hint = f" Available splits: {available}" if available else ""
            raise RuntimeError(f"Failed to load dataset split '{split}'.{hint}") from e

    items = list(ds)
    rng = random.Random(args.seed)
    rng.shuffle(items)

    max_to_write = None if args.num_prompts < 0 else int(args.num_prompts)

    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for ex in items:
            if max_to_write is not None and written >= max_to_write:
                break
            if not isinstance(ex, dict):
                continue

            context = _get_first(ex, ["context", "article", "document", "input", "passage"])
            question = _get_first(ex, ["question", "query", "instruction"])
            choices = _get_first(ex, ["choices", "options"])

            if not isinstance(context, str) or not isinstance(question, str):
                continue
            if args.min_context_chars and len(context) < int(args.min_context_chars):
                continue

            if isinstance(choices, dict) and "text" in choices:
                choices = choices["text"]

            if not isinstance(choices, list) or not choices:
                prompt = f"Context:\n{context.strip()}\n\nQuestion:\n{question.strip()}\n\nAnswer:"
            else:
                choices_str = [str(c).strip() for c in choices if str(c).strip()]
                prompt = _format_mcq(context, question, choices_str)

            if args.min_prompt_chars and len(prompt) < int(args.min_prompt_chars):
                continue

            record = {
                "request_id": str(uuid.uuid4()),
                "prompt": prompt,
                "max_tokens": int(args.max_new_tokens),
                "dataset": "longbenchv2",
                "meta": {
                    "split": split,
                    "min_context_chars": int(args.min_context_chars),
                    "min_prompt_chars": int(args.min_prompt_chars),
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} prompts to {out_path}")


if __name__ == "__main__":
    main()
