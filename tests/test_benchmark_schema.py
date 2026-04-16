from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "workloads" / "small_llm" / "benchmarks" / "sweep_schema.py"
spec = importlib.util.spec_from_file_location("small_llm_sweep_schema", MODULE_PATH)
assert spec is not None and spec.loader is not None
sweep_schema = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sweep_schema
spec.loader.exec_module(sweep_schema)

SWEEP_COLUMNS = sweep_schema.SWEEP_COLUMNS
SWEEP_DEFINITIONS = sweep_schema.SWEEP_DEFINITIONS
append_sweep_row = sweep_schema.append_sweep_row
get_sweep_csv_path = sweep_schema.get_sweep_csv_path
normalize_sweep_row = sweep_schema.normalize_sweep_row


class SweepSchemaIOTest(unittest.TestCase):
    def test_field_order_is_stable(self) -> None:
        self.assertEqual(
            list(SWEEP_COLUMNS),
            [
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
            ],
        )

    def test_normalize_preserves_columns_and_fills_missing(self) -> None:
        row = normalize_sweep_row(
            {
                "run_id": "r1",
                "sweep_type": "prefill",
                "prompt_len": 128,
            }
        )
        self.assertEqual(list(row.keys()), list(SWEEP_COLUMNS))
        self.assertEqual(row["run_id"], "r1")
        self.assertEqual(row["prompt_len"], 128)
        self.assertEqual(row["gen_len"], "")
        self.assertEqual(row["notes"], "")

    def test_get_sweep_csv_path_maps_known_sweeps(self) -> None:
        self.assertTrue(str(get_sweep_csv_path("prefill")).endswith("results/sweeps/prefill.csv"))
        self.assertTrue(str(get_sweep_csv_path("decode")).endswith("results/sweeps/decode.csv"))
        self.assertTrue(str(get_sweep_csv_path("overload")).endswith("results/sweeps/overload.csv"))

    def test_append_creates_headers_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "results" / "tables" / "sweep_runs.csv"
            sweep = root / "results" / "sweeps" / "prefill.csv"

            row = {
                "timestamp": "2026-04-15T00:00:00Z",
                "run_id": "run-1",
                "experiment_name": "exp-a",
                "sweep_type": "prefill",
                "scenario_name": "A_prompt_len_128",
                "backend": "baseline_fastapi",
                "backend_variant": "cpu",
                "model_name": "qwen_1_5b_instruct",
                "precision": "float32",
                "prompt_len": 128,
                "gen_len": 128,
                "concurrency": 1,
                "batch_size": 1,
                "offered_load_rps": "",
                "ttft_ms": 10.0,
                "latency_ms": 12.3,
                "p50_ms": 12.0,
                "p95_ms": 15.0,
                "decode_toks_per_s": 7.8,
                "throughput_req_per_s": 1.0,
                "throughput_toks_per_s": 7.8,
                "peak_vram_mb": "",
                "gpu_util_pct": "",
                "error_count": 0,
                "oom_flag": False,
                "notes": "ok",
            }

            append_sweep_row(row, master_csv_path=master, per_sweep_csv_path=sweep)

            for path in (master, sweep):
                self.assertTrue(path.exists())
                self.assertTrue(path.parent.exists())
                with path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    self.assertEqual(reader.fieldnames, list(SWEEP_COLUMNS))
                    rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["run_id"], "run-1")
                self.assertEqual(rows[0]["sweep_type"], "prefill")

    def test_append_handles_default_per_sweep_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "results" / "tables" / "sweep_runs.csv"
            per_sweep = root / "results" / "sweeps" / "decode.csv"
            original = sweep_schema.SWEEP_TYPE_TO_CSV["decode"]
            sweep_schema.SWEEP_TYPE_TO_CSV["decode"] = per_sweep
            try:
                row = {
                    "timestamp": "2026-04-15T00:00:00Z",
                    "run_id": "run-2",
                    "experiment_name": "exp-b",
                    "sweep_type": "decode",
                    "scenario_name": "B_gen_len_128",
                    "backend": "trtllm_direct",
                    "backend_variant": "gpu",
                    "model_name": "qwen_1_5b_instruct",
                    "precision": "fp16",
                    "prompt_len": 512,
                    "gen_len": 128,
                    "concurrency": 1,
                    "batch_size": 1,
                    "offered_load_rps": "",
                    "ttft_ms": 20.0,
                    "latency_ms": 25.0,
                    "p50_ms": 25.0,
                    "p95_ms": 27.0,
                    "decode_toks_per_s": 10.0,
                    "throughput_req_per_s": 1.0,
                    "throughput_toks_per_s": 10.0,
                    "peak_vram_mb": 1024.0,
                    "gpu_util_pct": "",
                    "error_count": 0,
                    "oom_flag": False,
                    "notes": "ok",
                }

                master_path, per_sweep_path = append_sweep_row(row, master_csv_path=master)
                self.assertEqual(master_path, master)
                self.assertEqual(per_sweep_path, per_sweep)
                self.assertTrue(per_sweep_path.exists())
                with per_sweep_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["backend"], "trtllm_direct")
            finally:
                sweep_schema.SWEEP_TYPE_TO_CSV["decode"] = original

    def test_definitions_cover_expected_sweeps(self) -> None:
        self.assertEqual(
            set(SWEEP_DEFINITIONS),
            {"prefill", "decode", "concurrency", "long_context_concurrency", "batching", "overload"},
        )


if __name__ == "__main__":
    unittest.main()
