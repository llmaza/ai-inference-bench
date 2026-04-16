from __future__ import annotations

import argparse
import json
import statistics
import threading
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workloads.small_llm.benchmarks.sweep_runtime import sleep_for_offered_load

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.small_llm.app.backends.triton_client import (  # noqa: E402
    TritonOpenAIBackend,
    resolve_triton_config,
)
from workloads.small_llm.app.logging_utils import (  # noqa: E402
    append_jsonl_record,
    artifact_metadata_record,
    request_record,
    run_summary_record,
    utc_now_iso,
    write_jsonl_records,
    write_summary_json,
)


SMALL_LLM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = (
    Path(__file__).resolve().parent / "prompts" / "prompts_rag_medium_tk_rk_baseline.jsonl"
)
STAGE_RESULTS_DIR = SMALL_LLM_ROOT / "results" / "stage_c_triton_trtllm"
RUNS_DIR = STAGE_RESULTS_DIR / "runs"
RAW_LOG_DIR = STAGE_RESULTS_DIR / "requests"
ARTIFACTS_DIR = STAGE_RESULTS_DIR / "artifacts"
REQUEST_LOG_PATH = RAW_LOG_DIR / "requests_triton_trtllm.jsonl"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


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
    rows: list[dict[str, Any]] = []
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


def run_once(backend: TritonOpenAIBackend, prompt: str, max_new_tokens: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    stats = backend.generate_with_stats(prompt, max_new_tokens=max_new_tokens)
    wall_latency_ms = (time.perf_counter() - started) * 1000
    stats_dict = stats.to_dict()
    return {
        "wall_latency_ms": round(wall_latency_ms, 4),
        "generated_text": stats_dict["text"],
        "input_tokens": stats_dict["input_tokens"],
        "generated_tokens": stats_dict["output_tokens"],
        "total_tokens": stats_dict["total_tokens"],
        "generation_ms": stats_dict["generation_ms"],
        "tokens_per_sec": stats_dict["tokens_per_sec"],
        "ttft_ms": stats_dict["ttft_ms"],
        "peak_gpu_memory_mb": stats_dict["peak_gpu_memory_mb"],
        "finish_reason": stats_dict["finish_reason"],
    }


def build_request_log_record(
    *,
    config,
    request_id: str,
    run_id: str | None,
    timestamp: str,
    prompt_name: str | None,
    prompt_file: str | None,
    request_index: int | None,
    repeat_index: int | None,
    prompt: str,
    result: dict[str, Any] | None,
    error: Exception | None = None,
) -> dict[str, Any]:
    dtype = config.runtime_precision
    if error is not None:
        return request_record(
            request_id=request_id,
            run_id=run_id,
            timestamp=timestamp,
            stage="stage_c_triton_trtllm",
            backend="triton_trtllm",
            model_key=config.model_key,
            model_name=config.model_name,
            prompt_name=prompt_name,
            prompt_file=prompt_file,
            repeat_index=repeat_index,
            request_index=request_index,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=False,
            wall_latency_ms=None,
            generation_ms=None,
            ttft_ms=None,
            tokens_per_sec=None,
            success=False,
            error_type=type(error).__name__,
            error_message=str(error),
            finish_reason=None,
            device="cuda",
            dtype=dtype,
            gpu_memory_mb=None,
            peak_gpu_memory_mb=None,
            generated_text=None,
            message=prompt,
            mode="triton_trtllm",
            generated_text_preview=None,
        )
    assert result is not None
    return request_record(
        request_id=request_id,
        run_id=run_id,
        timestamp=timestamp,
        stage="stage_c_triton_trtllm",
        backend="triton_trtllm",
        model_key=config.model_key,
        model_name=config.model_name,
        prompt_name=prompt_name,
        prompt_file=prompt_file,
        repeat_index=repeat_index,
        request_index=request_index,
        input_tokens=result["input_tokens"],
        generated_tokens=result["generated_tokens"],
        total_tokens=result["total_tokens"],
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=False,
        wall_latency_ms=result["wall_latency_ms"],
        generation_ms=result["generation_ms"],
        ttft_ms=result["ttft_ms"],
        tokens_per_sec=result["tokens_per_sec"],
        success=True,
        error_type=None,
        error_message=None,
        finish_reason=result["finish_reason"],
        device="cuda",
        dtype=dtype,
        gpu_memory_mb=None,
        peak_gpu_memory_mb=result["peak_gpu_memory_mb"],
        generated_text=result["generated_text"],
        message=prompt,
        mode="triton_trtllm",
        generated_text_preview=result["generated_text"][:200],
    )


def build_artifact_metadata(config) -> dict[str, Any]:
    source_metadata = {}
    if config.artifact_metadata_path.exists():
        try:
            source_metadata = json.loads(config.artifact_metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            source_metadata = {}
    engine_path = source_metadata.get("engine_path") or str(config.engine_path)
    engine_file = Path(engine_path)
    engine_size_bytes = engine_file.stat().st_size if engine_file.exists() else None
    timing_cache_path = source_metadata.get("timing_cache_path")
    if timing_cache_path is None:
        cache_guess = REPO_ROOT / "model.cache"
        timing_cache_path = str(cache_guess) if cache_guess.exists() else None
    return artifact_metadata_record(
        timestamp=utc_now_iso(),
        stage="stage_c_triton_trtllm",
        backend="triton_trtllm",
        model_key=config.model_key,
        engine_path=str(engine_file) if engine_file.exists() else None,
        engine_size_bytes=engine_size_bytes,
        build_time_sec=source_metadata.get("build_time_sec"),
        build_config={
            "source_artifact_metadata_path": str(config.artifact_metadata_path),
            "triton_base_url": config.base_url,
            "triton_chat_path": config.chat_path,
            "served_model_name": config.served_model_name,
            "tokenizer_path": str(config.tokenizer_path),
            "engine_path": str(config.engine_path),
            "max_input_len": config.max_input_len,
            "max_output_len": config.max_new_tokens,
            "temperature": config.temperature,
            "timeout_sec": config.timeout_sec,
        },
        timing_cache_path=timing_cache_path,
        runtime_precision=config.runtime_precision,
        max_input_len=config.max_input_len,
        max_output_len=config.max_new_tokens,
        source_artifact_metadata_path=str(config.artifact_metadata_path),
        triton_base_url=config.base_url,
        served_model_name=config.served_model_name,
        tokenizer_path=str(config.tokenizer_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton TRT-LLM OpenAI-compatible inference for small_llm.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--model-key", default="qwen_1_5b_instruct")
    parser.add_argument("--single-prompt", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(resolve_triton_config(args.model_key), timeout_sec=args.timeout_sec)
    backend = TritonOpenAIBackend(config)
    artifact_metadata_path = ARTIFACTS_DIR / f"{config.model_key}_triton_server_metadata.json"
    write_summary_json(artifact_metadata_path, build_artifact_metadata(config))

    if args.single_prompt:
        request_id = str(uuid.uuid4())
        timestamp = utc_now_iso()
        try:
            result = run_once(backend, args.single_prompt)
            append_jsonl_record(
                REQUEST_LOG_PATH,
                build_request_log_record(
                    config=config,
                    request_id=request_id,
                    run_id=None,
                    timestamp=timestamp,
                    prompt_name=None,
                    prompt_file=None,
                    request_index=None,
                    repeat_index=None,
                    prompt=args.single_prompt,
                    result=result,
                ),
            )
        except Exception as exc:
            append_jsonl_record(
                REQUEST_LOG_PATH,
                build_request_log_record(
                    config=config,
                    request_id=request_id,
                    run_id=None,
                    timestamp=timestamp,
                    prompt_name=None,
                    prompt_file=None,
                    request_index=None,
                    repeat_index=None,
                    prompt=args.single_prompt,
                    result=None,
                    error=exc,
                ),
            )
            raise
        print(
            json.dumps(
                {
                    "benchmark_schema_version": "small_llm.v1",
                    "mode": "triton_trtllm_single",
                    "stage": "stage_c_triton_trtllm",
                    "backend": "triton_trtllm",
                    "model_key": config.model_key,
                    **result,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    prompts = load_prompts(Path(args.prompts))
    run_id = str(uuid.uuid4())
    timestamp = utc_now_iso()
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
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()

    def task(request_index: int, prompt: dict[str, Any]) -> None:
        sleep_for_offered_load(started, request_index, offered_load_rps)
        request_id = str(uuid.uuid4())
        try:
            result = run_once(backend, prompt["message"], max_new_tokens=max_new_tokens)
            log_record = build_request_log_record(
                config=config,
                request_id=request_id,
                run_id=run_id,
                timestamp=timestamp,
                prompt_name=prompt.get("prompt_id"),
                prompt_file=str(Path(args.prompts).resolve()),
                request_index=request_index,
                repeat_index=prompt["repeat_index"],
                prompt=prompt["message"],
                result=result,
            )
            append_jsonl_record(REQUEST_LOG_PATH, log_record)
            with rows_lock:
                rows.append(
                    {
                        **log_record,
                        "mode": "triton_trtllm",
                        "prompt_id": prompt.get("prompt_id"),
                        "message": prompt["message"],
                        "status_code": None,
                        **result,
                    }
                )
        except Exception as exc:
            append_jsonl_record(
                REQUEST_LOG_PATH,
                build_request_log_record(
                    config=config,
                    request_id=request_id,
                    run_id=run_id,
                    timestamp=timestamp,
                    prompt_name=prompt.get("prompt_id"),
                    prompt_file=str(Path(args.prompts).resolve()),
                    request_index=request_index,
                    repeat_index=prompt["repeat_index"],
                    prompt=prompt["message"],
                    result=None,
                    error=exc,
                ),
            )
            raise

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(task, idx, prompt) for idx, prompt in enumerate(expanded)]
        for future in as_completed(futures):
            future.result()

    summary = run_summary_record(
        run_id=run_id,
        timestamp=timestamp,
        stage="stage_c_triton_trtllm",
        backend="triton_trtllm",
        model_key=config.model_key,
        model_name=config.model_name,
        prompt_file=str(Path(args.prompts).resolve()),
        concurrency=args.concurrency,
        num_requests=len(rows),
        success_rate=1.0,
        mean_latency_ms=summarize_stage(rows, "wall_latency_ms")["mean"],
        p50_latency_ms=summarize_stage(rows, "wall_latency_ms")["p50"],
        p95_latency_ms=summarize_stage(rows, "wall_latency_ms")["p95"],
        mean_generation_ms=summarize_stage(rows, "generation_ms")["mean"],
        mean_tokens_per_sec=summarize_stage(rows, "tokens_per_sec")["mean"],
        total_input_tokens=sum(
            int(row["input_tokens"]) for row in rows if row.get("input_tokens") is not None
        ),
        total_generated_tokens=sum(
            int(row["generated_tokens"]) for row in rows if row.get("generated_tokens") is not None
        ),
        notes="Stage C Triton OpenAI-compatible TRT-LLM benchmark.",
        mode="triton_trtllm",
        triton_base_url=config.base_url,
        served_model_name=config.served_model_name,
        runtime_precision=config.runtime_precision,
        artifact_metadata_path=str(artifact_metadata_path) if artifact_metadata_path.exists() else None,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=False,
        prompts_path=str(Path(args.prompts).resolve()),
        prompt_count=len(prompts),
        repeats=args.repeats,
        total_requests=len(rows),
        generated_tokens=summarize_stage(rows, "generated_tokens"),
        ttft_ms=summarize_stage(rows, "ttft_ms"),
        peak_gpu_memory_mb=summarize_stage(rows, "peak_gpu_memory_mb"),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{stamp}_triton_trtllm_{run_id}"
    raw_path = RUNS_DIR / f"{base_name}.jsonl"
    summary_path = RUNS_DIR / f"{base_name}_summary.json"
    write_jsonl_records(raw_path, rows)
    write_summary_json(summary_path, summary)
    print(json.dumps({"raw_results": str(raw_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
