from __future__ import annotations

import argparse
import json
import os
import random
import tarfile
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve


def _iter_ruler_jsonl_from_archive(
    archive_path: str,
    context_len: int,
    task_allowlist: set[str] | None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield (task_type, example) pairs from a ruler_data tgz archive."""

    target_suffix = f"validation_{context_len}.jsonl"

    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if not name.endswith(target_suffix):
                continue

            # Expected structure:
            # data/ruler/<task_type>/validation_<context_len>.jsonl
            parts = name.split("/")
            task_type = parts[-2] if len(parts) >= 2 else "unknown"
            if task_allowlist is not None and task_type not in task_allowlist:
                continue

            f = tf.extractfile(member)
            if f is None:
                continue
            for raw in f:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                if isinstance(ex, dict):
                    yield task_type, ex


def _download_ruler_archive(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download ruler_data archive via HTTPS and return local path."""

    cache = Path(os.path.expanduser(cache_dir))
    cache.mkdir(parents=True, exist_ok=True)

    safe_repo = repo_id.replace("/", "__")
    dst = cache / f"{safe_repo}__{filename}"
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst)

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}?download=true"
    try:
        urlretrieve(url, str(dst))
    except Exception as e:
        raise RuntimeError(
            "Failed to download RULER archive from HuggingFace. "
            "Please ensure you can access https://huggingface.co (and its CDN). "
            f"Tried URL: {url}"
        ) from e

    return str(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output prompts JSONL path")
    ap.add_argument(
        "--dataset",
        default="allenai/ruler_data",
        help="Dataset id. For RULER, keep default 'allenai/ruler_data'.",
    )
    ap.add_argument(
        "--archive",
        default="data_100_samples.tgz",
        help="Archive filename inside 'allenai/ruler_data' (e.g. data_debug.tgz, data_100_samples.tgz)",
    )
    ap.add_argument(
        "--cache-dir",
        default="~/.cache/vllm-ascend/datasets",
        help="Where to cache downloaded tgz archives",
    )
    ap.add_argument(
        "--context-len",
        type=int,
        default=4096,
        help="Pick validation_<context_len>.jsonl from the archive (default: 4096)",
    )
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--task-allowlist",
        default="",
        help="Comma-separated task allowlist (optional). Examples: qa_1,vt,niah_single_1",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    allow = {x.strip() for x in args.task_allowlist.split(",") if x.strip()} if args.task_allowlist else None

    if args.dataset == "allenai/ruler_data":
        archive_path = _download_ruler_archive(args.dataset, args.archive, args.cache_dir)
        items: list[tuple[str, dict[str, Any]]] = list(
            _iter_ruler_jsonl_from_archive(archive_path, args.context_len, allow)
        )
    else:
        # Best-effort compatibility path (for custom datasets).
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Please install `datasets` if you want to load non-archive datasets via --dataset."
            ) from e

        ds = load_dataset(args.dataset, split="test")
        items = []
        for ex in ds:
            if not isinstance(ex, dict):
                continue
            task_type = ex.get("task_type") or ex.get("task") or ex.get("type")
            if allow is not None and task_type not in allow:
                continue
            items.append((str(task_type) if task_type else "unknown", ex))

    rng = random.Random(args.seed)
    rng.shuffle(items)

    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for task_type, ex in items:
            if written >= args.num_prompts:
                break

            prompt = ex.get("input") or ex.get("text") or ex.get("prompt") or ex.get("query")
            if not isinstance(prompt, str) or not prompt.strip():
                continue

            record = {
                "request_id": str(uuid.uuid4()),
                "prompt": prompt.strip(),
                "max_tokens": int(args.max_tokens),
                "dataset": "ruler",
                "task_type": task_type,
                "meta": {
                    "source": args.dataset,
                    "archive": args.archive if args.dataset == "allenai/ruler_data" else None,
                    "context_len": int(args.context_len) if args.dataset == "allenai/ruler_data" else None,
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} prompts to {out_path}")


if __name__ == "__main__":
    main()
