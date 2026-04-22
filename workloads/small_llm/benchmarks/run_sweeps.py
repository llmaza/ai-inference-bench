from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SWEEP_SCHEMA_PATH = Path(__file__).resolve().with_name("sweep_schema.py")
sweep_schema_spec = importlib.util.spec_from_file_location("small_llm_sweep_schema", SWEEP_SCHEMA_PATH)
assert sweep_schema_spec is not None and sweep_schema_spec.loader is not None
sweep_schema = importlib.util.module_from_spec(sweep_schema_spec)
sys.modules[sweep_schema_spec.name] = sweep_schema
sweep_schema_spec.loader.exec_module(sweep_schema)

append_sweep_row = sweep_schema.append_sweep_row
get_sweep_definition = sweep_schema.get_sweep_definition
get_sweep_csv_path = sweep_schema.get_sweep_csv_path
from workloads.small_llm.benchmarks.sweep_runtime import materialize_sweep_prompts, sleep_for_offered_load

SWEEP_ALIAS_TO_TYPE = {
    "A": "prefill",
    "B": "decode",
    "C": "concurrency",
    "D": "long_context_concurrency",
    "E": "batching",
    "F": "overload",
}

BACKEND_CHOICES = ("baseline_fastapi", "trtllm_direct", "triton", "vllm")
DEFAULT_PROMPTS_PATH = (
    Path(__file__).resolve().parent / "prompts" / "prompts_rag_medium_tk_rk_baseline.jsonl"
)
DEFAULT_EXPERIMENT_NAME = "small_llm_sweeps"


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


def resolve_sweep_type(sweep_alias: str) -> str:
    try:
        return SWEEP_ALIAS_TO_TYPE[sweep_alias.upper()]
    except KeyError as exc:
        raise KeyError(f"Unknown sweep alias: {sweep_alias!r}") from exc


def resolve_override_params(args: argparse.Namespace, varied_param: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for name in ("prompt_len", "gen_len", "concurrency", "batch_size", "offered_load_rps"):
        value = getattr(args, name)
        if value is not None and name != varied_param:
            overrides[name] = value
    return overrides


def sweep_points_for_definition(definition, args: argparse.Namespace):
    overrides = resolve_override_params(args, definition.varied_param)
    varied_override = getattr(args, definition.varied_param, None)
    if varied_override is not None:
        grid_values = (varied_override,)
    else:
        grid_values = definition.grid_values
    for varied_value in grid_values:
        params = definition.build_run_params(varied_value, overrides=overrides)
        yield varied_value, params


def load_backend_runner(backend: str) -> Callable[..., dict[str, Any]]:
    if backend == "baseline_fastapi":
        from workloads.small_llm.benchmarks.run_baseline import run_baseline_sweep_point

        return run_baseline_sweep_point
    if backend == "trtllm_direct":
        from workloads.small_llm.benchmarks.run_trtllm import run_trtllm_sweep_point

        return run_trtllm_sweep_point
    if backend == "triton":
        return run_triton_sweep_point
    if backend == "vllm":
        run_vllm_path = Path(__file__).resolve().with_name("run_vllm.py")
        spec = importlib.util.spec_from_file_location("small_llm_run_vllm", run_vllm_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.run_vllm_sweep_point
    raise KeyError(f"Unsupported backend: {backend!r}")


def run_triton_sweep_point(
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
    """Run one sweep point against the existing Triton OpenAI-compatible backend.

    The Triton path uses single chat requests for batch_size=1 and OpenAI
    completions prompt-list requests for batch_size>1.
    """

    import workloads.small_llm.benchmarks.run_triton as triton_runner

    resolved_model_key = model_key or "qwen_1_5b_instruct"
    config = triton_runner.resolve_triton_config(resolved_model_key)
    backend = triton_runner.TritonOpenAIBackend(config)
    prompt_rows = triton_runner.load_prompts(Path(prompts))
    max_new_tokens = max_new_tokens if max_new_tokens is not None else gen_len

    expanded: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        for prompt in prompt_rows:
            expanded.append(
                {
                    "prompt_id": prompt.get("prompt_id"),
                    "message": prompt["message"],
                    "repeat_index": repeat_index,
                }
            )

    run_id = str(uuid.uuid4())
    timestamp = triton_runner.utc_now_iso()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()

    if batch_size > 1:
        batches = [expanded[i : i + batch_size] for i in range(0, len(expanded), batch_size)]

        def task(batch_index: int, batch_prompts: list[dict[str, Any]]) -> None:
            sleep_for_offered_load(started, batch_index * batch_size, offered_load_rps)
            messages_batch = [
                [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": prompt["message"]},
                ]
                for prompt in batch_prompts
            ]
            batch_results = backend.generate_batch_with_stats(messages_batch, max_new_tokens=max_new_tokens)
            with rows_lock:
                for offset, result_obj in enumerate(batch_results):
                    request_index = batch_index * batch_size + offset
                    result = result_obj.to_dict()
                    rows.append(
                        {
                            "request_index": request_index,
                            "success": True,
                            "wall_latency_ms": result["generation_ms"],
                            "generation_ms": result["generation_ms"],
                            "ttft_ms": result["ttft_ms"],
                            "tokens_per_sec": result["tokens_per_sec"],
                            "input_tokens": result["input_tokens"],
                            "generated_tokens": result["output_tokens"],
                            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                        }
                    )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(task, idx, batch) for idx, batch in enumerate(batches)]
            for future in as_completed(futures):
                future.result()
    else:
        def task(request_index: int, prompt: dict[str, Any]) -> None:
            sleep_for_offered_load(started, request_index, offered_load_rps)
            result = triton_runner.run_once(backend, prompt["message"], max_new_tokens=max_new_tokens)
            row = {
                "request_index": request_index,
                "success": True,
                "wall_latency_ms": result["wall_latency_ms"],
                "generation_ms": result["generation_ms"],
                "ttft_ms": result["ttft_ms"],
                "tokens_per_sec": result["tokens_per_sec"],
                "input_tokens": result["input_tokens"],
                "generated_tokens": result["generated_tokens"],
                "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            }
            with rows_lock:
                rows.append(row)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(task, idx, prompt) for idx, prompt in enumerate(expanded)]
            for future in as_completed(futures):
                future.result()

    total_wall_sec = time.perf_counter() - started
    rows = sorted(rows, key=lambda row: row["request_index"])
    success_rows = [row for row in rows if row.get("success")]
    summary = {
        "timestamp": timestamp,
        "run_id": run_id,
        "stage": "stage_c_triton_trtllm",
        "backend": "triton",
        "model_key": config.model_key,
        "model_name": getattr(config, "model_name", config.model_key),
        "prompt_file": str(Path(prompts).resolve()),
        "concurrency": concurrency,
        "num_requests": len(rows),
        "success_rate": round(len(success_rows) / len(rows), 4) if rows else 0.0,
        "mean_latency_ms": summarize_stage(success_rows, "wall_latency_ms")["mean"],
        "p50_latency_ms": summarize_stage(success_rows, "wall_latency_ms")["p50"],
        "p95_latency_ms": summarize_stage(success_rows, "wall_latency_ms")["p95"],
        "mean_generation_ms": summarize_stage(success_rows, "generation_ms")["mean"],
        "mean_tokens_per_sec": summarize_stage(success_rows, "tokens_per_sec")["mean"],
        "total_input_tokens": sum(int(row["input_tokens"]) for row in success_rows if row.get("input_tokens") is not None),
        "total_generated_tokens": sum(
            int(row["generated_tokens"]) for row in success_rows if row.get("generated_tokens") is not None
        ),
        "notes": notes or "Triton sweep point.",
        "throughput_rps": round(len(rows) / total_wall_sec, 4) if total_wall_sec > 0 else None,
        "ttft_ms": summarize_stage(success_rows, "ttft_ms"),
        "peak_gpu_memory_mb": summarize_stage(success_rows, "peak_gpu_memory_mb"),
    }

    caveats: list[str] = []
    if batch_size not in (None, 1):
        caveats.append("batch_size is applied through Triton OpenAI completions prompt-list batching.")
    if offered_load_rps is not None:
        caveats.append("offered_load_rps pacing is applied by the benchmark runner.")

    note_parts = [part.strip() for part in (notes, *caveats) if part and part.strip()]
    combined_notes = " ".join(note_parts).strip()
    sweep_row = {
        "timestamp": summary["timestamp"],
        "run_id": summary["run_id"],
        "experiment_name": experiment_name,
        "sweep_type": sweep_type,
        "scenario_name": scenario_name,
        "backend": summary["backend"],
        "backend_variant": "gpu",
        "model_name": summary["model_name"],
        "precision": getattr(config, "runtime_precision", None),
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "concurrency": concurrency,
        "batch_size": batch_size,
        "offered_load_rps": offered_load_rps,
        "ttft_ms": summary["ttft_ms"]["mean"],
        "latency_ms": summary["mean_latency_ms"],
        "p50_ms": summary["p50_latency_ms"],
        "p95_ms": summary["p95_latency_ms"],
        "decode_toks_per_s": summary["mean_tokens_per_sec"],
        "throughput_req_per_s": summary["throughput_rps"],
        "throughput_toks_per_s": summary["mean_tokens_per_sec"],
        "total_input_tokens": summary["total_input_tokens"],
        "total_generated_tokens": summary["total_generated_tokens"],
        "peak_vram_mb": summary["peak_gpu_memory_mb"]["mean"],
        "gpu_util_pct": None,
        "error_count": 0 if summary["success_rate"] == 1 else None,
        "oom_flag": False,
        "notes": combined_notes or "Triton sweep point.",
    }
    return {"rows": rows, "summary": summary, "sweep_row": sweep_row}


def run_one_sweep_point(
    backend: str,
    sweep_type: str,
    scenario_name: str,
    experiment_name: str,
    params: dict[str, Any],
    *,
    model_key: str | None,
    repeats: int,
    notes: str,
    warmup_requests: int,
) -> dict[str, Any]:
    runner = load_backend_runner(backend)
    runner_kwargs = {
        "sweep_type": sweep_type,
        "scenario_name": scenario_name,
        "experiment_name": experiment_name,
        "model_key": model_key,
        "repeats": repeats,
        "notes": notes,
        **params,
    }
    if backend in {"baseline_fastapi", "vllm"}:
        runner_kwargs["warmup_requests"] = warmup_requests
    return runner(**runner_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small_llm sweeps.")
    parser.add_argument("sweep", choices=tuple(SWEEP_ALIAS_TO_TYPE))
    parser.add_argument("--backend", required=True, choices=BACKEND_CHOICES)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--notes", default="")
    parser.add_argument("--results-mode", choices=("debug", "main"), default="debug")
    parser.add_argument("--prompt-len", type=int, default=None)
    parser.add_argument("--gen-len", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--offered-load-rps", type=float, default=None)
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_type = resolve_sweep_type(args.sweep)
    definition = get_sweep_definition(sweep_type)
    experiment_name = DEFAULT_EXPERIMENT_NAME
    results_root = REPO_ROOT / "workloads" / "small_llm" / "results"
    if args.results_mode == "debug":
        results_root = results_root / "debug"
    master_csv_path = results_root / "tables" / "sweep_runs.csv"
    sweep_csv_path = results_root / "sweeps" / definition.csv_path.name

    varied_override = getattr(args, definition.varied_param, None)
    grid_values = (varied_override,) if varied_override is not None else definition.grid_values
    overrides = resolve_override_params(args, definition.varied_param)

    written = []
    with tempfile.TemporaryDirectory(prefix="small_llm_sweeps_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for varied_value in grid_values:
            sweep_params = definition.build_run_params(varied_value, overrides=overrides)
            prompt_len_target = sweep_params.get("prompt_len")
            prompts_path = args.prompts
            if prompt_len_target is not None:
                prompts_path, prompt_rows = materialize_sweep_prompts(
                    model_key=args.model_key or "qwen_1_5b_instruct",
                    target_prompt_len=int(prompt_len_target),
                    prompts_source=Path(args.prompts),
                    output_dir=temp_dir_path,
                )
                sweep_params["prompt_len"] = prompt_rows[0]["actual_prompt_len"]
            result = run_one_sweep_point(
                args.backend,
                sweep_type,
                scenario_name=f"{definition.short_id}_{definition.varied_param}_{varied_value}",
                experiment_name=experiment_name,
                params={
                    **sweep_params,
                    "prompts": str(prompts_path),
                    "max_new_tokens": sweep_params.get("gen_len"),
                    "offered_load_rps": sweep_params.get("offered_load_rps"),
                    "batch_size": sweep_params.get("batch_size"),
                },
                model_key=args.model_key,
                repeats=args.repeats,
                notes=args.notes,
                warmup_requests=args.warmup_requests,
            )
            master_csv_path, sweep_csv_path = append_sweep_row(
                result["sweep_row"],
                master_csv_path=master_csv_path,
                per_sweep_csv_path=sweep_csv_path,
            )
            written.append(
                {
                    "sweep_type": sweep_type,
                    "scenario_name": f"{definition.short_id}_{definition.varied_param}_{varied_value}",
                    "master_csv": str(master_csv_path),
                    "sweep_csv": str(sweep_csv_path),
                    "run_id": result["summary"]["run_id"],
                }
            )

    print(
        json.dumps(
            {
                "backend": args.backend,
                "sweep": args.sweep,
                "sweep_type": sweep_type,
                "points": written,
                "master_csv": str(master_csv_path),
                "sweep_csv": str(sweep_csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
