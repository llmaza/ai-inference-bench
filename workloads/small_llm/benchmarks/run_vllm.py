from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workloads.small_llm.benchmarks.sweep_runtime import sleep_for_offered_load

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parents[1] / "app"
BACKENDS_DIR = APP_DIR / "backends"


def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


logging_utils = _load_local_module("small_llm_logging_utils", APP_DIR / "logging_utils.py")
append_jsonl_record = logging_utils.append_jsonl_record
request_record = logging_utils.request_record
run_summary_record = logging_utils.run_summary_record
utc_now_iso = logging_utils.utc_now_iso
write_jsonl_records = logging_utils.write_jsonl_records
write_summary_json = logging_utils.write_summary_json


def load_vllm_client():
    return _load_local_module("small_llm_vllm_client", BACKENDS_DIR / "vllm_client.py")


SMALL_LLM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts" / "prompts_rag_medium_tk_rk_baseline.jsonl"
STAGE_RESULTS_DIR = SMALL_LLM_ROOT / "results" / "stage_d_vllm"
RUNS_DIR = STAGE_RESULTS_DIR / "runs"
RAW_LOG_DIR = STAGE_RESULTS_DIR / "requests"
REQUEST_LOG_PATH = RAW_LOG_DIR / "requests_vllm.jsonl"
def ensure_output_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)


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


def run_once(backend: Any, prompt: str, max_new_tokens: int | None = None) -> dict[str, Any]:
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
            stage="stage_d_vllm",
            backend="vllm",
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
            mode="vllm",
            generated_text_preview=None,
        )
    assert result is not None
    return request_record(
        request_id=request_id,
        run_id=run_id,
        timestamp=timestamp,
        stage="stage_d_vllm",
        backend="vllm",
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
        finish_reason=result.get("finish_reason"),
        device="cuda",
        dtype=dtype,
        gpu_memory_mb=None,
        peak_gpu_memory_mb=result["peak_gpu_memory_mb"],
        generated_text=result["generated_text"],
        message=prompt,
        mode="vllm",
        generated_text_preview=result["generated_text"][:200],
    )


def build_sweep_row(
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
    backend_variant: str = "openai",
    precision: str | None = "fp16",
    notes: str = "",
) -> dict[str, Any]:
    caveats: list[str] = []
    if batch_size not in (None, 1):
        caveats.append("batch_size is applied through the vLLM OpenAI completions batch path.")
    if offered_load_rps is not None:
        caveats.append("offered_load_rps pacing is applied by the benchmark runner.")

    note_parts = [part.strip() for part in (notes, *caveats) if part and part.strip()]
    combined_notes = " ".join(note_parts).strip()
    resolved_model_key = model_key or summary.get("model_key")
    ttft = summary.get("ttft_ms")
    peak_vram = summary.get("peak_gpu_memory_mb")
    return {
        "timestamp": summary.get("timestamp"),
        "run_id": summary.get("run_id"),
        "experiment_name": experiment_name,
        "sweep_type": sweep_type,
        "scenario_name": scenario_name,
        "backend": summary.get("backend") or "vllm",
        "backend_variant": backend_variant,
        "model_name": summary.get("model_name") or resolved_model_key,
        "precision": precision,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "concurrency": concurrency,
        "batch_size": batch_size,
        "offered_load_rps": offered_load_rps,
        "ttft_ms": ttft.get("mean") if isinstance(ttft, dict) else ttft,
        "latency_ms": summary.get("mean_latency_ms"),
        "p50_ms": summary.get("p50_latency_ms"),
        "p95_ms": summary.get("p95_latency_ms"),
        "decode_toks_per_s": summary.get("mean_tokens_per_sec"),
        "throughput_req_per_s": summary.get("throughput_rps"),
        "throughput_toks_per_s": summary.get("mean_tokens_per_sec"),
        "total_input_tokens": summary.get("total_input_tokens"),
        "total_generated_tokens": summary.get("total_generated_tokens"),
        "peak_vram_mb": peak_vram.get("mean") if isinstance(peak_vram, dict) else peak_vram,
        "gpu_util_pct": None,
        "error_count": 0 if summary.get("success_rate", 0) == 1 else None,
        "oom_flag": False,
        "notes": combined_notes or "vLLM sweep point.",
    }


def benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ensure_output_dirs()
    vllm_client = load_vllm_client()
    config = vllm_client.resolve_vllm_config(args.model_key)
    backend = vllm_client.VLLMOpenAIBackend(config)
    prompts = load_prompts(Path(args.prompts))
    max_new_tokens = getattr(args, "max_new_tokens", None)
    offered_load_rps = getattr(args, "offered_load_rps", None)
    batch_size = max(1, int(getattr(args, "batch_size", 1) or 1))
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

    if batch_size > 1:
        batches = [expanded[i : i + batch_size] for i in range(0, len(expanded), batch_size)]

        def task(batch_index: int, batch_prompts: list[dict[str, Any]]) -> None:
            sleep_for_offered_load(started, batch_index * batch_size, offered_load_rps)
            request_id = str(uuid.uuid4())
            messages_batch = [
                [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": prompt["message"]},
                ]
                for prompt in batch_prompts
            ]
            try:
                batch_results = backend.generate_batch_with_stats(messages_batch, max_new_tokens=max_new_tokens)
                for offset, (prompt, result_obj) in enumerate(zip(batch_prompts, batch_results)):
                    request_index = batch_index * batch_size + offset
                    result = result_obj.to_dict()
                    log_record = build_request_log_record(
                        config=config,
                        request_id=f"{request_id}:{offset}",
                        run_id=run_id,
                        timestamp=timestamp,
                        prompt_name=prompt.get("prompt_id"),
                        prompt_file=str(Path(args.prompts).resolve()),
                        request_index=request_index,
                        repeat_index=prompt["repeat_index"],
                        prompt=prompt["message"],
                        result={
                            "wall_latency_ms": result["generation_ms"],
                            "generated_text": result["text"],
                            "input_tokens": result["input_tokens"],
                            "generated_tokens": result["output_tokens"],
                            "total_tokens": result["total_tokens"],
                            "generation_ms": result["generation_ms"],
                            "tokens_per_sec": result["tokens_per_sec"],
                            "ttft_ms": result["ttft_ms"],
                            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                            "finish_reason": result["finish_reason"],
                        },
                    )
                    append_jsonl_record(REQUEST_LOG_PATH, log_record)
                    with rows_lock:
                        rows.append(
                            {
                                **log_record,
                                "mode": "vllm",
                                "prompt_id": prompt.get("prompt_id"),
                                "message": prompt["message"],
                                "status_code": None,
                                "wall_latency_ms": result["generation_ms"],
                                **result,
                            }
                        )
            except Exception as exc:
                for offset, prompt in enumerate(batch_prompts):
                    request_index = batch_index * batch_size + offset
                    append_jsonl_record(
                        REQUEST_LOG_PATH,
                        build_request_log_record(
                            config=config,
                            request_id=f"{request_id}:{offset}",
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
            futures = [executor.submit(task, idx, batch_prompts) for idx, batch_prompts in enumerate(batches)]
            for future in as_completed(futures):
                future.result()
    else:
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
                            "mode": "vllm",
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

    total_wall_sec = time.perf_counter() - started
    summary = run_summary_record(
        run_id=run_id,
        timestamp=timestamp,
        stage="stage_d_vllm",
        backend="vllm",
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
        total_input_tokens=sum(int(row["input_tokens"]) for row in rows if row.get("input_tokens") is not None),
        total_generated_tokens=sum(int(row["generated_tokens"]) for row in rows if row.get("generated_tokens") is not None),
        notes="Stage D vLLM OpenAI-compatible benchmark.",
        mode="vllm",
        vllm_base_url=config.base_url,
        served_model_name=config.served_model_name,
        runtime_precision=config.runtime_precision,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=False,
        prompts_path=str(Path(args.prompts).resolve()),
        prompt_count=len(prompts),
        repeats=args.repeats,
        total_requests=len(rows),
        throughput_rps=round(len(rows) / total_wall_sec, 4) if total_wall_sec > 0 else None,
        generated_tokens=summarize_stage(rows, "generated_tokens"),
        ttft_ms=summarize_stage(rows, "ttft_ms"),
        peak_gpu_memory_mb=summarize_stage(rows, "peak_gpu_memory_mb"),
    )
    return rows, summary


def run_vllm_sweep_point(
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
    prompts: str | Path = DEFAULT_PROMPTS_PATH,
    notes: str = "",
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    args = argparse.Namespace(
        prompts=str(prompts),
        repeats=repeats,
        concurrency=concurrency,
        model_key=model_key,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else gen_len,
        offered_load_rps=offered_load_rps,
        batch_size=batch_size,
    )
    rows, summary = benchmark(args)
    sweep_row = build_sweep_row(
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
        notes=notes,
    )
    return {"rows": rows, "summary": summary, "sweep_row": sweep_row}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM OpenAI-compatible inference for small_llm.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model-key", default="qwen_1_5b_instruct")
    parser.add_argument("--single-prompt", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_prompt:
        ensure_output_dirs()
        vllm_client = load_vllm_client()
        config = vllm_client.resolve_vllm_config(args.model_key)
        backend = vllm_client.VLLMOpenAIBackend(config)
        request_id = str(uuid.uuid4())
        timestamp = utc_now_iso()
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
        print(
            json.dumps(
                {
                    "benchmark_schema_version": "small_llm.v1",
                    "mode": "vllm_single",
                    "stage": "stage_d_vllm",
                    "backend": "vllm",
                    "model_key": config.model_key,
                    **result,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    rows, summary = benchmark(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{stamp}_vllm_{summary['run_id']}"
    raw_path = RUNS_DIR / f"{base_name}.jsonl"
    summary_path = RUNS_DIR / f"{base_name}_summary.json"
    write_jsonl_records(raw_path, rows)
    write_summary_json(summary_path, summary)
    print(json.dumps({"raw_results": str(raw_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
