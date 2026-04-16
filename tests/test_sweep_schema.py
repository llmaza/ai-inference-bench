from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "workloads" / "small_llm" / "benchmarks" / "sweep_schema.py"
spec = importlib.util.spec_from_file_location("sweep_schema", MODULE_PATH)
assert spec is not None and spec.loader is not None
sweep_schema = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = sweep_schema
spec.loader.exec_module(sweep_schema)

SWEEP_COLUMNS = sweep_schema.SWEEP_COLUMNS
SWEEP_DEFINITIONS = sweep_schema.SWEEP_DEFINITIONS
append_sweep_row = sweep_schema.append_sweep_row
get_sweep_definition = sweep_schema.get_sweep_definition
normalize_sweep_row = sweep_schema.normalize_sweep_row


class SweepSchemaTest(unittest.TestCase):
    def test_normalize_fills_all_columns(self) -> None:
        row = normalize_sweep_row(
            {
                "timestamp": "2026-04-15T00:00:00Z",
                "run_id": "abc",
                "sweep_type": "prefill",
                "backend": "baseline_fastapi",
                "prompt_len": 128,
                "latency_ms": 12.3,
            }
        )
        self.assertEqual(list(row.keys()), list(SWEEP_COLUMNS))
        self.assertEqual(row["run_id"], "abc")
        self.assertEqual(row["prompt_len"], 128)
        self.assertEqual(row["gen_len"], "")

    def test_definitions_cover_all_sweeps(self) -> None:
        expected = {
            "prefill",
            "decode",
            "concurrency",
            "long_context_concurrency",
            "batching",
            "overload",
        }
        self.assertEqual(set(SWEEP_DEFINITIONS), expected)
        prefill = get_sweep_definition("prefill")
        self.assertEqual(prefill.short_id, "A")
        self.assertEqual(prefill.varied_param, "prompt_len")
        self.assertEqual(prefill.grid_values, (128, 512, 1024, 2048))
        self.assertEqual(prefill.fixed_params["gen_len"], 128)
        self.assertEqual(prefill.fixed_params["concurrency"], 1)
        self.assertEqual(prefill.fixed_params["batch_size"], 1)

    def test_append_writes_master_and_sweep_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "results" / "tables" / "sweep_runs.csv"
            sweep = root / "results" / "sweeps" / "prefill.csv"
            row = {
                "timestamp": "2026-04-15T00:00:00Z",
                "run_id": "run-1",
                "experiment_name": "exp-a",
                "sweep_type": "prefill",
                "scenario_name": "scenario-a",
                "backend": "baseline_fastapi",
                "backend_variant": "cpu",
                "model_name": "qwen",
                "precision": "float32",
                "prompt_len": 128,
                "gen_len": 32,
                "concurrency": 1,
                "batch_size": 1,
                "offered_load_rps": "",
                "ttft_ms": "",
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
                with path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    self.assertEqual(reader.fieldnames, list(SWEEP_COLUMNS))
                    rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["run_id"], "run-1")
                self.assertEqual(rows[0]["sweep_type"], "prefill")


if __name__ == "__main__":
    unittest.main()
