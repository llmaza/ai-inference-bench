from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


SMALL_LLM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.labor_sample.jsonl"
RESULTS_DIR = SMALL_LLM_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "message" not in row:
                raise ValueError("Each JSONL row must contain a 'message' field")
            rows.append(row)
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_stage(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 4),
        "p50": round(percentile(values, 0.50), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def send_generate(session: requests.Session, url: str, message: str, timeout_sec: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = session.post(url, json={"message": message}, timeout=timeout_sec)
        wall_latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            return {
                "success": False,
                "status_code": response.status_code,
                "wall_latency_ms": wall_latency_ms,
                "error_type": f"HTTP_{response.status_code}",
                "response": None,
            }
        return {
            "success": True,
            "status_code": response.status_code,
            "wall_latency_ms": wall_latency_ms,
            "error_type": None,
            "response": response.json(),
        }
    except requests.Timeout:
        return {
            "success": False,
            "status_code": None,
            "wall_latency_ms": (time.perf_counter() - started) * 1000,
            "error_type": "timeout",
            "response": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "status_code": None,
            "wall_latency_ms": (time.perf_counter() - started) * 1000,
            "error_type": type(exc).__name__,
            "response": None,
        }


def benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = load_prompts(Path(args.prompts))
    expanded: list[dict[str, Any]] = []
    for repeat_index in range(args.repeats):
        for prompt in prompts:
            expanded.append(
                {
                    "prompt_id": prompt.get("prompt_id"),
                    "message": prompt["message"],
                    "repeat_index": repeat_index,
                }
            )

    run_id = str(uuid.uuid4())
    timestamp = utc_now_iso()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    endpoint = args.base_url.rstrip("/") + "/generate"

    def task(request_index: int, prompt: dict[str, Any]) -> None:
        with requests.Session() as session:
            result = send_generate(session, endpoint, prompt["message"], args.timeout_sec)
        payload = result.get("response") or {}
        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "mode": "baseline_fastapi",
            "request_index": request_index,
            "prompt_id": prompt.get("prompt_id"),
            "repeat_index": prompt["repeat_index"],
            "message": prompt["message"],
            "success": result["success"],
            "status_code": result["status_code"],
            "error_type": result["error_type"],
            "wall_latency_ms": round(result["wall_latency_ms"], 4),
            "api_latency_ms": payload.get("latency_ms"),
            "generation_ms": payload.get("generation_ms"),
            "input_tokens": payload.get("input_tokens"),
            "output_tokens": payload.get("output_tokens"),
            "tokens_per_sec": payload.get("tokens_per_sec"),
            "ttft_ms": payload.get("ttft_ms"),
            "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb"),
            "model_key": payload.get("model_key"),
            "model_name": payload.get("model_name"),
            "device": payload.get("device"),
            "text_preview": (payload.get("text") or "")[:200],
        }
        with rows_lock:
            rows.append(row)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(task, idx, prompt) for idx, prompt in enumerate(expanded)]
        for future in as_completed(futures):
            future.result()

    total_wall_sec = time.perf_counter() - started
    rows = sorted(rows, key=lambda row: row["request_index"])
    success_rows = [row for row in rows if row["success"]]
    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "mode": "baseline_fastapi",
        "base_url": args.base_url,
        "prompts_path": str(Path(args.prompts).resolve()),
        "prompt_count": len(prompts),
        "repeats": args.repeats,
        "total_requests": len(rows),
        "concurrency": args.concurrency,
        "timeout_sec": args.timeout_sec,
        "success_rate": round(len(success_rows) / len(rows), 4) if rows else 0.0,
        "throughput_rps": round(len(rows) / total_wall_sec, 4) if total_wall_sec > 0 else None,
        "latency_ms": summarize_stage(success_rows, "wall_latency_ms"),
        "api_latency_ms": summarize_stage(success_rows, "api_latency_ms"),
        "generation_ms": summarize_stage(success_rows, "generation_ms"),
        "tokens_per_sec": summarize_stage(success_rows, "tokens_per_sec"),
        "output_tokens": summarize_stage(success_rows, "output_tokens"),
        "ttft_ms": summarize_stage(success_rows, "ttft_ms"),
        "peak_gpu_memory_mb": summarize_stage(success_rows, "peak_gpu_memory_mb"),
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the local small_llm FastAPI baseline.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = benchmark(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{stamp}_baseline_fastapi_{summary['run_id']}"
    raw_path = RESULTS_DIR / f"{base_name}.jsonl"
    summary_path = RESULTS_DIR / f"{base_name}_summary.json"
    write_jsonl(raw_path, rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"raw_results": str(raw_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

