from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from workloads.small_llm.benchmarks.sweep_runtime import sleep_for_offered_load

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.small_llm.app.logging_utils import (  # noqa: E402
    append_jsonl_record,
    request_record,
    run_summary_record,
    utc_now_iso,
    write_jsonl_records,
    write_summary_json,
)
from workloads.small_llm.app.backends.trtllm_direct import (  # noqa: E402
    TensorRTLLMUnavailable,
    get_backend,
    resolve_trtllm_config,
)


SMALL_LLM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts" / "prompts_rag_medium_tk_rk_baseline.jsonl"
STAGE_RESULTS_DIR = SMALL_LLM_ROOT / "results" / "stage_b_trtllm_direct"
RUNS_DIR = STAGE_RESULTS_DIR / "runs"
RAW_LOG_DIR = STAGE_RESULTS_DIR / "requests"
REQUEST_LOG_PATH = RAW_LOG_DIR / "requests_trtllm.jsonl"
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


def run_once(backend, prompt: str, max_new_tokens: int | None = None) -> dict:
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
    result: dict | None,
    error: Exception | None = None,
) -> dict:
    if error is not None:
        return request_record(
            request_id=request_id,
            run_id=run_id,
            timestamp=timestamp,
            stage="trtllm_direct",
            backend="trtllm_direct",
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
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=None,
            gpu_memory_mb=None,
            peak_gpu_memory_mb=None,
            generated_text=None,
            message=prompt,
            mode="trtllm_direct",
            generated_text_preview=None,
        )
    assert result is not None
    return request_record(
        request_id=request_id,
        run_id=run_id,
        timestamp=timestamp,
        stage="trtllm_direct",
        backend="trtllm_direct",
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
        finish_reason=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=None,
        gpu_memory_mb=None,
        peak_gpu_memory_mb=result["peak_gpu_memory_mb"],
        generated_text=result["generated_text"],
        message=prompt,
        mode="trtllm_direct",
        generated_text_preview=result["generated_text"][:200],
    )



def build_trtllm_sweep_row(
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
    backend_variant: str = "gpu",
    precision: str | None = "fp16",
    notes: str = "",
) -> dict[str, Any]:
    """Map a completed TRT-LLM direct benchmark into the shared sweep schema.

    TRT-LLM direct does not currently receive per-sweep prompt synthesis, batching,
    or offered-load pacing from this runner. Those values are recorded as sweep
    metadata so the new sweep runner can keep a consistent contract across backends.
    """

    caveats: list[str] = []
    if batch_size not in (None, 1):
        caveats.append("batch_size is applied through the batched TRT-LLM execution path.")
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
        "backend": summary.get("backend") or "trtllm_direct",
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
        "notes": combined_notes or "TRT-LLM direct sweep point.",
    }



def run_trtllm_sweep_point(
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
) -> dict[str, Any]:
    """Run one sweep point through the existing TRT-LLM direct benchmark path."""

    resolved_model_key = model_key or "qwen_1_5b_instruct"
    args = argparse.Namespace(
        prompts=str(prompts),
        repeats=repeats,
        model_key=resolved_model_key,
        single_prompt=None,
        max_new_tokens=gen_len,
        offered_load_rps=offered_load_rps,
        batch_size=batch_size,
    )
    rows, summary = benchmark(args)
    sweep_row = build_trtllm_sweep_row(
        summary=summary,
        sweep_type=sweep_type,
        scenario_name=scenario_name,
        experiment_name=experiment_name,
        prompt_len=prompt_len,
        gen_len=gen_len,
        concurrency=concurrency,
        batch_size=batch_size,
        offered_load_rps=offered_load_rps,
        model_key=resolved_model_key,
        notes=notes,
    )
    return {
        "rows": rows,
        "summary": summary,
        "sweep_row": sweep_row,
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
                    "mode": "trtllm_direct_single",
                    "stage": "trtllm_direct",
                    "backend": "trtllm_direct",
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
    artifact_metadata_path = STAGE_RESULTS_DIR / "artifacts" / f"{config.model_key}_metadata.json"
    max_new_tokens = getattr(args, "max_new_tokens", None) or config.max_new_tokens
    offered_load_rps = getattr(args, "offered_load_rps", None)
    batch_size = max(1, int(getattr(args, "batch_size", 1) or 1))
    rows = []
    started = time.perf_counter()
    if batch_size > 1:
        batches = []
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
        for idx in range(0, len(expanded), batch_size):
            batches.append(expanded[idx : idx + batch_size])
        for batch_index, batch_prompts in enumerate(batches):
            sleep_for_offered_load(started, batch_index * batch_size, offered_load_rps)
            request_index_start = batch_index * batch_size
            request_id = str(uuid.uuid4())
            try:
                batch_results = backend.generate_batch_with_stats(
                    [prompt["message"] for prompt in batch_prompts],
                    max_new_tokens=max_new_tokens,
                )
                for offset, (prompt, result_obj) in enumerate(zip(batch_prompts, batch_results)):
                    request_index = request_index_start + offset
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
                        },
                    )
                    append_jsonl_record(REQUEST_LOG_PATH, log_record)
                    rows.append(
                        {
                            **log_record,
                            "mode": "trtllm_direct",
                            "prompt_id": prompt.get("prompt_id"),
                            "message": prompt["message"],
                            "status_code": None,
                            "wall_latency_ms": result["generation_ms"],
                            **result,
                        }
                    )
            except Exception as exc:
                for offset, prompt in enumerate(batch_prompts):
                    request_index = request_index_start + offset
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
    else:
        for repeat_index in range(args.repeats):
            for prompt in prompts:
                request_index = len(rows)
                request_id = str(uuid.uuid4())
                try:
                    sleep_for_offered_load(started, request_index, offered_load_rps)
                    result = run_once(backend, prompt["message"], max_new_tokens=max_new_tokens)
                    log_record = build_request_log_record(
                        config=config,
                        request_id=request_id,
                        run_id=run_id,
                        timestamp=timestamp,
                        prompt_name=prompt.get("prompt_id"),
                        prompt_file=str(Path(args.prompts).resolve()),
                        request_index=request_index,
                        repeat_index=repeat_index,
                        prompt=prompt["message"],
                        result=result,
                    )
                    append_jsonl_record(REQUEST_LOG_PATH, log_record)
                    rows.append(
                        {
                            **log_record,
                            "mode": "trtllm_direct",
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
                            repeat_index=repeat_index,
                            prompt=prompt["message"],
                            result=None,
                            error=exc,
                        ),
                    )
                    raise

    summary = run_summary_record(
        run_id=run_id,
        timestamp=timestamp,
        stage="trtllm_direct",
        backend="trtllm_direct",
        model_key=config.model_key,
        model_name=config.model_name,
        prompt_file=str(Path(args.prompts).resolve()),
        concurrency=1,
        num_requests=len(rows),
        success_rate=1.0,
        mean_latency_ms=summarize_stage(rows, "wall_latency_ms")["mean"],
        p50_latency_ms=summarize_stage(rows, "wall_latency_ms")["p50"],
        p95_latency_ms=summarize_stage(rows, "wall_latency_ms")["p95"],
        mean_generation_ms=summarize_stage(rows, "generation_ms")["mean"],
        mean_tokens_per_sec=summarize_stage(rows, "tokens_per_sec")["mean"],
        total_input_tokens=sum(int(row["input_tokens"]) for row in rows if row.get("input_tokens") is not None),
        total_generated_tokens=sum(
            int(row["generated_tokens"]) for row in rows if row.get("generated_tokens") is not None
        ),
        notes="Stage B TensorRT-LLM direct benchmark.",
        mode="trtllm_direct",
        engine_dir=str(config.engine_dir),
        artifact_metadata_path=str(artifact_metadata_path) if artifact_metadata_path.exists() else None,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=False,
        prompts_path=str(Path(args.prompts).resolve()),
        prompt_count=len(prompts),
        repeats=args.repeats,
        total_requests=len(rows),
        latency_ms=summarize_stage(rows, "wall_latency_ms"),
        generation_ms=summarize_stage(rows, "generation_ms"),
        tokens_per_sec=summarize_stage(rows, "tokens_per_sec"),
        generated_tokens=summarize_stage(rows, "generated_tokens"),
        ttft_ms=summarize_stage(rows, "ttft_ms"),
        peak_gpu_memory_mb=summarize_stage(rows, "peak_gpu_memory_mb"),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{stamp}_trtllm_direct_{run_id}"
    raw_path = RUNS_DIR / f"{base_name}.jsonl"
    summary_path = RUNS_DIR / f"{base_name}_summary.json"
    write_jsonl_records(raw_path, rows)
    write_summary_json(summary_path, summary)
    print(json.dumps({"raw_results": str(raw_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
