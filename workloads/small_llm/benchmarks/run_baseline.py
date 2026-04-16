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
import sys
from typing import Any

from workloads.small_llm.benchmarks.sweep_runtime import sleep_for_offered_load

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.small_llm.app.logging_utils import (
    append_jsonl_record,
    request_record,
    run_summary_record,
    utc_now_iso,
    write_jsonl_records,
    write_summary_json,
)

SMALL_LLM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts" / "prompts_rag_medium_tk_rk_baseline.jsonl"
STAGE_RESULTS_DIR = SMALL_LLM_ROOT / "results" / "stage_a_baseline"
RUNS_DIR = STAGE_RESULTS_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


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


def send_generate(
    session: requests.Session,
    url: str,
    message: str,
    timeout_sec: float,
    max_new_tokens: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {"message": message}
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    try:
        response = session.post(url, json=payload, timeout=timeout_sec)
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


def send_generate_batch(
    session: requests.Session,
    url: str,
    items: list[dict[str, Any]],
    timeout_sec: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {"items": items}
    try:
        response = session.post(url, json=payload, timeout=timeout_sec)
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
    endpoint = args.base_url.rstrip("/")
    max_new_tokens = getattr(args, "max_new_tokens", None)
    offered_load_rps = getattr(args, "offered_load_rps", None)
    batch_size = max(1, int(getattr(args, "batch_size", 1) or 1))

    def append_row(prompt: dict[str, Any], request_index: int, result: dict[str, Any], batch_latency_ms: float | None = None) -> None:
        payload = result.get("response") or {}
        row = request_record(
            run_id=run_id,
            timestamp=timestamp,
            stage="baseline_fastapi",
            backend="baseline_fastapi",
            model_key=payload.get("model_key"),
            model_name=payload.get("model_name"),
            prompt_name=prompt.get("prompt_id"),
            prompt_file=str(Path(args.prompts).resolve()),
            repeat_index=prompt["repeat_index"],
            request_index=request_index,
            input_tokens=payload.get("input_tokens"),
            generated_tokens=payload.get("output_tokens"),
            total_tokens=payload.get("total_tokens"),
            max_new_tokens=max_new_tokens,
            temperature=None,
            do_sample=False,
            wall_latency_ms=round(batch_latency_ms if batch_latency_ms is not None else result["wall_latency_ms"], 4),
            generation_ms=payload.get("generation_ms"),
            ttft_ms=payload.get("ttft_ms"),
            tokens_per_sec=payload.get("tokens_per_sec"),
            success=result["success"],
            error_type=result["error_type"],
            error_message=None,
            finish_reason="unknown" if result["success"] else "error",
            device=payload.get("device"),
            dtype=None,
            gpu_memory_mb=None,
            peak_gpu_memory_mb=payload.get("peak_gpu_memory_mb"),
            generated_text=payload.get("text"),
            mode="baseline_fastapi",
            prompt_id=prompt.get("prompt_id"),
            message=prompt["message"],
            status_code=result["status_code"],
            api_latency_ms=payload.get("latency_ms"),
            output_tokens=payload.get("output_tokens"),
            text_preview=(payload.get("text") or "")[:200],
        )
        with rows_lock:
            rows.append(row)

    if batch_size > 1:
        batch_endpoint = endpoint + "/generate_batch"
        batches = [expanded[i : i + batch_size] for i in range(0, len(expanded), batch_size)]

        def task(batch_index: int, batch_prompts: list[dict[str, Any]]) -> None:
            batch_items = [
                {
                    "message": prompt["message"],
                    "prompt_name": prompt.get("prompt_id"),
                    "prompt_file": str(Path(args.prompts).resolve()),
                    "request_index": global_index,
                    "repeat_index": prompt["repeat_index"],
                    "max_new_tokens": max_new_tokens,
                }
                for global_index, prompt in enumerate(batch_prompts, start=batch_index * batch_size)
            ]
            sleep_for_offered_load(started, batch_prompts[0].get("request_index", batch_index * batch_size), offered_load_rps)
            with requests.Session() as session:
                result = send_generate_batch(session, batch_endpoint, batch_items, args.timeout_sec)
            if not result["success"]:
                for global_index, prompt in enumerate(batch_prompts, start=batch_index * batch_size):
                    append_row(prompt, global_index, result, batch_latency_ms=result["wall_latency_ms"])
                return
            payload = result.get("response") or {}
            items = payload.get("items") or []
            for global_index, prompt, item in zip(range(batch_index * batch_size, batch_index * batch_size + len(batch_prompts)), batch_prompts, items):
                append_row(prompt, global_index, {**result, "response": item}, batch_latency_ms=result["wall_latency_ms"])

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(task, idx, batch) for idx, batch in enumerate(batches)]
            for future in as_completed(futures):
                future.result()
    else:
        def task(request_index: int, prompt: dict[str, Any]) -> None:
            sleep_for_offered_load(started, request_index, offered_load_rps)
            with requests.Session() as session:
                result = send_generate(session, endpoint + "/generate", prompt["message"], args.timeout_sec, max_new_tokens)
            append_row(prompt, request_index, result)

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(task, idx, prompt) for idx, prompt in enumerate(expanded)]
            for future in as_completed(futures):
                future.result()

    total_wall_sec = time.perf_counter() - started
    rows = sorted(rows, key=lambda row: row["request_index"])
    success_rows = [row for row in rows if row["success"]]
    summary = run_summary_record(
        run_id=run_id,
        timestamp=timestamp,
        stage="baseline_fastapi",
        backend="baseline_fastapi",
        model_key=success_rows[0].get("model_key") if success_rows else None,
        model_name=success_rows[0].get("model_name") if success_rows else None,
        prompt_file=str(Path(args.prompts).resolve()),
        concurrency=args.concurrency,
        num_requests=len(rows),
        success_rate=round(len(success_rows) / len(rows), 4) if rows else 0.0,
        mean_latency_ms=summarize_stage(success_rows, "wall_latency_ms")["mean"],
        p50_latency_ms=summarize_stage(success_rows, "wall_latency_ms")["p50"],
        p95_latency_ms=summarize_stage(success_rows, "wall_latency_ms")["p95"],
        mean_generation_ms=summarize_stage(success_rows, "generation_ms")["mean"],
        mean_tokens_per_sec=summarize_stage(success_rows, "tokens_per_sec")["mean"],
        total_input_tokens=sum(int(row["input_tokens"]) for row in success_rows if row.get("input_tokens") is not None),
        total_generated_tokens=sum(
            int(row["generated_tokens"]) for row in success_rows if row.get("generated_tokens") is not None
        ),
        notes="Stage A local FastAPI baseline benchmark.",
        mode="baseline_fastapi",
        base_url=args.base_url,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else success_rows[0].get("max_new_tokens") if success_rows else None,
        temperature=success_rows[0].get("temperature") if success_rows else None,
        do_sample=False,
        prompts_path=str(Path(args.prompts).resolve()),
        prompt_count=len(prompts),
        repeats=args.repeats,
        total_requests=len(rows),
        timeout_sec=args.timeout_sec,
        throughput_rps=round(len(rows) / total_wall_sec, 4) if total_wall_sec > 0 else None,
        latency_ms=summarize_stage(success_rows, "wall_latency_ms"),
        api_latency_ms=summarize_stage(success_rows, "api_latency_ms"),
        generation_ms=summarize_stage(success_rows, "generation_ms"),
        tokens_per_sec=summarize_stage(success_rows, "tokens_per_sec"),
        output_tokens=summarize_stage(success_rows, "output_tokens"),
        ttft_ms=summarize_stage(success_rows, "ttft_ms"),
        peak_gpu_memory_mb=summarize_stage(success_rows, "peak_gpu_memory_mb"),
    )
    return rows, summary


def build_baseline_sweep_row(
    *,
    summary: dict[str, Any],
    sweep_type: str,
    scenario_name: str,
    experiment_name: str,
    prompt_len: int | None,
    gen_len: int | None,
    concurrency: int | None,
    batch_size: int | None,
    offered_load_rps: float | int | None,
    model_key: str | None = None,
    backend_variant: str = "cpu",
    precision: str | None = "float32",
    notes: str = "",
) -> dict[str, Any]:
    """Map a completed baseline benchmark summary into the shared sweep schema.

    The baseline FastAPI service does not currently enforce batching or offered-load
    pacing internally. Those knobs are recorded as sweep metadata so the sweep runner
    can keep a consistent contract, but the actual request execution still follows the
    existing concurrency-driven benchmark path.
    """

    caveats: list[str] = []
    if batch_size not in (None, 1):
        caveats.append("batch_size is applied through the FastAPI /generate_batch path.")
    if offered_load_rps is not None:
        caveats.append("offered_load_rps pacing is applied by the benchmark runner.")

    note_parts = [part.strip() for part in (notes, *caveats) if part and part.strip()]
    combined_notes = " ".join(note_parts).strip()

    resolved_model_key = model_key or summary.get("model_key")
    return {
        "timestamp": summary.get("timestamp"),
        "run_id": summary.get("run_id"),
        "experiment_name": experiment_name,
        "sweep_type": sweep_type,
        "scenario_name": scenario_name,
        "backend": summary.get("backend") or "baseline_fastapi",
        "backend_variant": backend_variant,
        "model_name": summary.get("model_name") or resolved_model_key,
        "precision": precision,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "concurrency": concurrency,
        "batch_size": batch_size,
        "offered_load_rps": offered_load_rps,
        "ttft_ms": summary.get("ttft_ms", {}).get("mean") if isinstance(summary.get("ttft_ms"), dict) else summary.get("ttft_ms"),
        "latency_ms": summary.get("mean_latency_ms"),
        "p50_ms": summary.get("p50_latency_ms"),
        "p95_ms": summary.get("p95_latency_ms"),
        "decode_toks_per_s": summary.get("mean_tokens_per_sec"),
        "throughput_req_per_s": summary.get("throughput_rps"),
        "throughput_toks_per_s": summary.get("mean_tokens_per_sec"),
        "total_input_tokens": summary.get("total_input_tokens"),
        "total_generated_tokens": summary.get("total_generated_tokens"),
        "peak_vram_mb": summary.get("peak_gpu_memory_mb", {}).get("mean") if isinstance(summary.get("peak_gpu_memory_mb"), dict) else summary.get("peak_gpu_memory_mb"),
        "gpu_util_pct": None,
        "error_count": 0 if summary.get("success_rate", 0) == 1 else None,
        "oom_flag": False,
        "notes": combined_notes or "Baseline FastAPI sweep point.",
    }



def run_baseline_sweep_point(
    *,
    sweep_type: str,
    scenario_name: str,
    experiment_name: str,
    prompt_len: int | None,
    gen_len: int | None,
    concurrency: int,
    batch_size: int,
    offered_load_rps: float | int | None = None,
    model_key: str | None = None,
    repeats: int = 3,
    base_url: str = "http://127.0.0.1:8010",
    prompts: str | Path = DEFAULT_PROMPTS_PATH,
    timeout_sec: float = 180.0,
    max_new_tokens: int | None = None,
    backend_variant: str = "cpu",
    precision: str | None = "float32",
    notes: str = "",
) -> dict[str, Any]:
    """Run one sweep point with the existing baseline benchmark implementation.

    This keeps the CLI benchmark behavior intact while giving the sweep runner a
    stable callable contract that returns both the raw benchmark artifacts and the
    sweep-schema row.
    """

    args = argparse.Namespace(
        base_url=base_url,
        prompts=str(prompts),
        repeats=repeats,
        concurrency=concurrency,
        timeout_sec=timeout_sec,
        max_new_tokens=max_new_tokens,
        offered_load_rps=offered_load_rps,
        batch_size=batch_size,
    )
    rows, summary = benchmark(args)
    sweep_row = build_baseline_sweep_row(
        summary=summary,
        sweep_type=sweep_type,
        scenario_name=scenario_name,
        experiment_name=experiment_name,
        prompt_len=prompt_len,
        gen_len=gen_len,
        concurrency=concurrency,
        batch_size=batch_size,
        offered_load_rps=offered_load_rps,
        model_key=model_key,
        backend_variant=backend_variant,
        precision=precision,
        notes=notes,
    )
    return {
        "rows": rows,
        "summary": summary,
        "sweep_row": sweep_row,
    }


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
    raw_path = RUNS_DIR / f"{base_name}.jsonl"
    summary_path = RUNS_DIR / f"{base_name}_summary.json"
    write_jsonl_records(raw_path, rows)
    write_summary_json(summary_path, summary)
    print(json.dumps({"raw_results": str(raw_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
