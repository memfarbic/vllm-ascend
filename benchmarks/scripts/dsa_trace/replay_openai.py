from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptReq:
    request_id: str
    prompt: str
    max_tokens: int


@dataclass
class WorkloadReq:
    request_id: str
    arrival_offset_s: float
    input_len_tokens: int
    output_len_tokens: int


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any] | None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"HTTPError {e.code}: {detail}") from e


def _load_prompts_jsonl(path: Path) -> list[PromptReq]:
    out: list[PromptReq] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                PromptReq(
                    request_id=str(obj.get("request_id", "")),
                    prompt=str(obj.get("prompt", "")),
                    max_tokens=int(obj.get("max_tokens", 256)),
                )
            )
    return out


def _load_workload_jsonl(path: Path) -> list[WorkloadReq]:
    out: list[WorkloadReq] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                WorkloadReq(
                    request_id=str(obj.get("request_id", "")),
                    arrival_offset_s=float(obj.get("arrival_offset_s", 0.0)),
                    input_len_tokens=int(obj.get("input_len_tokens", 0)),
                    output_len_tokens=int(obj.get("output_len_tokens", 0)),
                )
            )
    out.sort(key=lambda x: x.arrival_offset_s)
    return out


def _synthetic_prompt(target_tokens: int) -> str:
    # Best-effort: approximate token length without tokenizer.
    # This keeps prompts deterministic and ASCII-only.
    n = max(1, target_tokens)
    return ("a " * n).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--model", required=True, help="Model name served by vLLM")
    ap.add_argument("--prompts-jsonl", default="", help="Prepared prompts.jsonl (ShareGPT/RULER/LongBench v2)")
    ap.add_argument("--workload-jsonl", default="", help="Prepared workload.jsonl (BurstGPT arrival trace)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--override-max-tokens",
        "--override-max-new-tokens",
        dest="override_max_tokens",
        type=int,
        default=0,
        help="If >0, override max_tokens for all requests (prompts/workload).",
    )
    args = ap.parse_args()

    if not args.prompts_jsonl and not args.workload_jsonl:
        raise SystemExit("Please provide either --prompts-jsonl or --workload-jsonl.")

    rng = random.Random(args.seed)
    lock = threading.Lock()
    ok = 0
    fail = 0

    def _send(prompt: str, max_tokens: int) -> None:
        nonlocal ok, fail
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(args.temperature),
            "stream": False,
        }
        try:
            _http_post_json(args.endpoint, payload, args.timeout_s)
            with lock:
                ok += 1
        except Exception:
            with lock:
                fail += 1

    if args.prompts_jsonl:
        prompts = _load_prompts_jsonl(Path(args.prompts_jsonl))
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(_send, p.prompt, (args.override_max_tokens if args.override_max_tokens > 0 else p.max_tokens)) for p in prompts]
            for _ in as_completed(futs):
                pass
        print(f"Done. ok={ok} fail={fail}")
        return

    workload = _load_workload_jsonl(Path(args.workload_jsonl))
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = []
        for r in workload:
            now = time.perf_counter()
            target = start + float(r.arrival_offset_s)
            if target > now:
                time.sleep(target - now)
            prompt = _synthetic_prompt(r.input_len_tokens)
            max_tokens = int(args.override_max_tokens) if args.override_max_tokens > 0 else int(max(1, r.output_len_tokens))
            futs.append(ex.submit(_send, prompt, max_tokens))
        for _ in as_completed(futs):
            pass
    print(f"Done. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()

