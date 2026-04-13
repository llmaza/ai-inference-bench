from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.small_llm.app.backends.trtllm_direct import (  # noqa: E402
    TensorRTLLMUnavailable,
    get_backend,
    resolve_trtllm_config,
)


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


def load_prompts(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({"prompt_id": row.get("prompt_id"), "message": row["message"]})
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_stage(rows: list[dict], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 4),
        "p50": round(percentile(values, 0.50), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def run_once(backend, prompt: str) -> dict:
    started = time.perf_counter()
    stats = backend.generate_with_stats(prompt)
    wall_latency_ms = (time.perf_counter() - started) * 1000
    return {
        "wall_latency_ms": round(wall_latency_ms, 4),
        **stats.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT-LLM direct inference for small_llm.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model-key", default="qwen_1_5b_instruct")
    parser.add_argument("--single-prompt", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = resolve_trtllm_config(args.model_key)
        backend = get_backend(args.model_key)
    except (TensorRTLLMUnavailable, FileNotFoundError) as exc:
        raise SystemExit(str(exc))

    if args.single_prompt:
        result = run_once(backend, args.single_prompt)
        print(
            json.dumps(
                {"mode": "trtllm_direct_single", "model_key": config.model_key, **result},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    prompts = load_prompts(Path(args.prompts))
    run_id = str(uuid.uuid4())
    timestamp = utc_now_iso()
    rows = []
    for repeat_index in range(args.repeats):
        for prompt in prompts:
            result = run_once(backend, prompt["message"])
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "mode": "trtllm_direct",
                    "request_index": len(rows),
                    "prompt_id": prompt.get("prompt_id"),
                    "repeat_index": repeat_index,
                    "message": prompt["message"],
                    "success": True,
                    "status_code": None,
                    "error_type": None,
                    "model_key": config.model_key,
                    "model_name": config.model_name,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "text_preview": result["text"][:200],
                    **result,
                }
            )

    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "mode": "trtllm_direct",
        "model_key": config.model_key,
        "model_name": config.model_name,
        "engine_dir": str(config.engine_dir),
        "prompts_path": str(Path(args.prompts).resolve()),
        "prompt_count": len(prompts),
        "repeats": args.repeats,
        "total_requests": len(rows),
        "success_rate": 1.0,
        "latency_ms": summarize_stage(rows, "wall_latency_ms"),
        "generation_ms": summarize_stage(rows, "generation_ms"),
        "tokens_per_sec": summarize_stage(rows, "tokens_per_sec"),
        "output_tokens": summarize_stage(rows, "output_tokens"),
        "ttft_ms": summarize_stage(rows, "ttft_ms"),
        "peak_gpu_memory_mb": summarize_stage(rows, "peak_gpu_memory_mb"),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{stamp}_trtllm_direct_{run_id}"
    raw_path = RESULTS_DIR / f"{base_name}.jsonl"
    summary_path = RESULTS_DIR / f"{base_name}_summary.json"
    write_jsonl(raw_path, rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"raw_results": str(raw_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
