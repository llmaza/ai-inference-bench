from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
SMALL_LLM_RESULTS_ROOT = REPO_ROOT / "workloads" / "small_llm" / "results"
SWEEP_TABLES_DIR = SMALL_LLM_RESULTS_ROOT / "tables"
SWEEP_RESULTS_DIR = SMALL_LLM_RESULTS_ROOT / "sweeps"

SWEEP_MASTER_CSV = SWEEP_TABLES_DIR / "sweep_runs.csv"
SWEEP_PREFILL_CSV = SWEEP_RESULTS_DIR / "prefill.csv"
SWEEP_DECODE_CSV = SWEEP_RESULTS_DIR / "decode.csv"
SWEEP_CONCURRENCY_CSV = SWEEP_RESULTS_DIR / "concurrency.csv"
SWEEP_LONGCTX_CONCURRENCY_CSV = SWEEP_RESULTS_DIR / "long_context_concurrency.csv"
SWEEP_BATCHING_CSV = SWEEP_RESULTS_DIR / "batching.csv"
SWEEP_OVERLOAD_CSV = SWEEP_RESULTS_DIR / "overload.csv"

SWEEP_COLUMNS = (
    "timestamp",
    "run_id",
    "experiment_name",
    "sweep_type",
    "scenario_name",
    "backend",
    "backend_variant",
    "model_name",
    "precision",
    "prompt_len",
    "gen_len",
    "concurrency",
    "batch_size",
    "offered_load_rps",
    "ttft_ms",
    "latency_ms",
    "p50_ms",
    "p95_ms",
    "decode_toks_per_s",
    "throughput_req_per_s",
    "throughput_toks_per_s",
    "total_input_tokens",
    "total_generated_tokens",
    "peak_vram_mb",
    "gpu_util_pct",
    "error_count",
    "oom_flag",
    "notes",
)

SWEEP_TYPE_TO_CSV = {
    "prefill": SWEEP_PREFILL_CSV,
    "decode": SWEEP_DECODE_CSV,
    "concurrency": SWEEP_CONCURRENCY_CSV,
    "long_context_concurrency": SWEEP_LONGCTX_CONCURRENCY_CSV,
    "batching": SWEEP_BATCHING_CSV,
    "overload": SWEEP_OVERLOAD_CSV,
}


@dataclass(frozen=True)
class SweepDefinition:
    """Static metadata for one sweep family.

    The runner can use this structure directly and merge CLI overrides on top of
    ``fixed_params`` without having to duplicate sweep-specific logic.
    """

    sweep_type: str
    short_id: str
    label: str
    csv_path: Path
    varied_param: str
    grid_values: tuple[Any, ...]
    fixed_params: Mapping[str, Any]

    def build_run_params(
        self,
        varied_value: Any,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = dict(self.fixed_params)
        params[self.varied_param] = varied_value
        if overrides:
            params.update(overrides)
        return params


SWEEP_DEFINITIONS: dict[str, SweepDefinition] = {
    "prefill": SweepDefinition(
        sweep_type="prefill",
        short_id="A",
        label="Prefill",
        csv_path=SWEEP_PREFILL_CSV,
        varied_param="prompt_len",
        grid_values=(128, 256, 512, 1024, 2048),
        fixed_params={
            "gen_len": 128,
            "concurrency": 1,
            "batch_size": 1,
        },
    ),
    "decode": SweepDefinition(
        sweep_type="decode",
        short_id="B",
        label="Decode",
        csv_path=SWEEP_DECODE_CSV,
        varied_param="gen_len",
        grid_values=(64, 128, 256),
        fixed_params={
            "prompt_len": 512,
            "concurrency": 1,
            "batch_size": 1,
        },
    ),
    "concurrency": SweepDefinition(
        sweep_type="concurrency",
        short_id="C",
        label="Concurrency",
        csv_path=SWEEP_CONCURRENCY_CSV,
        varied_param="concurrency",
        grid_values=(1, 2, 4, 8, 16),
        fixed_params={
            "prompt_len": 512,
            "gen_len": 128,
            "batch_size": 1,
        },
    ),
    "long_context_concurrency": SweepDefinition(
        sweep_type="long_context_concurrency",
        short_id="D",
        label="Long-context concurrency",
        csv_path=SWEEP_LONGCTX_CONCURRENCY_CSV,
        varied_param="concurrency",
        grid_values=(1, 2, 4, 8),
        fixed_params={
            "prompt_len": 2048,
            "gen_len": 128,
            "batch_size": 1,
        },
    ),
    "batching": SweepDefinition(
        sweep_type="batching",
        short_id="E",
        label="Batching",
        csv_path=SWEEP_BATCHING_CSV,
        varied_param="batch_size",
        grid_values=(1, 2, 4, 8),
        fixed_params={
            "prompt_len": 512,
            "gen_len": 128,
            "concurrency": 1,
        },
    ),
    "overload": SweepDefinition(
        sweep_type="overload",
        short_id="F",
        label="Overload",
        csv_path=SWEEP_OVERLOAD_CSV,
        varied_param="offered_load_rps",
        grid_values=(1, 2, 4, 8),
        fixed_params={
            "prompt_len": 512,
            "gen_len": 128,
            "concurrency": 1,
            "batch_size": 1,
        },
    ),
}

SWEEP_ORDER: tuple[str, ...] = tuple(SWEEP_DEFINITIONS)


def _normalize_cell(value: Any) -> Any:
    if value is None:
        return ""
    return value


def normalize_sweep_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {column: "" for column in SWEEP_COLUMNS}
    for column in SWEEP_COLUMNS:
        normalized[column] = _normalize_cell(row.get(column))
    return normalized


def _write_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalize_sweep_row(row))


def get_sweep_csv_path(sweep_type: str) -> Path:
    try:
        return SWEEP_TYPE_TO_CSV[sweep_type]
    except KeyError as exc:
        raise KeyError(f"Unknown sweep_type: {sweep_type!r}") from exc


def get_sweep_definition(sweep_type: str) -> SweepDefinition:
    try:
        return SWEEP_DEFINITIONS[sweep_type]
    except KeyError as exc:
        raise KeyError(f"Unknown sweep_type: {sweep_type!r}") from exc


def iter_sweep_points(
    sweep_type: str,
    *,
    overrides: Mapping[str, Any] | None = None,
):
    definition = get_sweep_definition(sweep_type)
    for varied_value in definition.grid_values:
        yield varied_value, definition.build_run_params(varied_value, overrides=overrides)


def append_sweep_row(
    row: Mapping[str, Any],
    *,
    master_csv_path: Path = SWEEP_MASTER_CSV,
    per_sweep_csv_path: Path | None = None,
) -> tuple[Path, Path]:
    sweep_type = row.get("sweep_type")
    if not sweep_type:
        raise ValueError("row must include sweep_type")

    resolved_per_sweep = per_sweep_csv_path or get_sweep_csv_path(str(sweep_type))
    _write_csv_row(master_csv_path, row)
    _write_csv_row(resolved_per_sweep, row)
    return master_csv_path, resolved_per_sweep
