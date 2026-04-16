from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK_SCHEMA_VERSION = "small_llm.v1"

REQUEST_RECORD_KEYS = (
    "benchmark_schema_version",
    "record_type",
    "request_id",
    "run_id",
    "timestamp",
    "stage",
    "backend",
    "model_key",
    "model_name",
    "prompt_name",
    "prompt_file",
    "repeat_index",
    "request_index",
    "input_tokens",
    "generated_tokens",
    "total_tokens",
    "max_new_tokens",
    "temperature",
    "do_sample",
    "wall_latency_ms",
    "generation_ms",
    "ttft_ms",
    "tokens_per_sec",
    "success",
    "error_type",
    "error_message",
    "finish_reason",
    "device",
    "dtype",
    "gpu_memory_mb",
    "peak_gpu_memory_mb",
    "generated_text",
)

RUN_SUMMARY_KEYS = (
    "benchmark_schema_version",
    "record_type",
    "run_id",
    "timestamp",
    "stage",
    "backend",
    "model_key",
    "model_name",
    "prompt_file",
    "concurrency",
    "num_requests",
    "success_rate",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_generation_ms",
    "mean_tokens_per_sec",
    "total_input_tokens",
    "total_generated_tokens",
    "notes",
)

ARTIFACT_METADATA_KEYS = (
    "benchmark_schema_version",
    "record_type",
    "timestamp",
    "stage",
    "backend",
    "model_key",
    "engine_path",
    "engine_size_bytes",
    "build_time_sec",
    "build_config",
    "timing_cache_path",
    "runtime_precision",
    "max_input_len",
    "max_output_len",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def request_record(**overrides: Any) -> dict[str, Any]:
    record = {key: None for key in REQUEST_RECORD_KEYS}
    record["benchmark_schema_version"] = BENCHMARK_SCHEMA_VERSION
    record["record_type"] = "request"
    record.update(overrides)
    return record


def run_summary_record(**overrides: Any) -> dict[str, Any]:
    record = {key: None for key in RUN_SUMMARY_KEYS}
    record["benchmark_schema_version"] = BENCHMARK_SCHEMA_VERSION
    record["record_type"] = "run_summary"
    record.update(overrides)
    return record


def artifact_metadata_record(**overrides: Any) -> dict[str, Any]:
    record = {key: None for key in ARTIFACT_METADATA_KEYS}
    record["benchmark_schema_version"] = BENCHMARK_SCHEMA_VERSION
    record["record_type"] = "artifact_metadata"
    record.update(overrides)
    return record
