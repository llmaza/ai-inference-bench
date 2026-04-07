import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

from benchmark.load_test import SUMMARY_TABLE_PATH, append_summary_row, write_request_results
from benchmark.metrics import summarize_run
from benchmark.schema import BenchmarkRequestResult
from workloads.bert_classifier.triton.client import TritonBertClient


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "workloads" / "bert_classifier" / "artifacts" / "model"
DEFAULT_SCENARIO_PATH = REPO_ROOT / "benchmark" / "scenarios" / "bert_inputs.jsonl"
DEFAULT_TRITON_URL = "http://127.0.0.1:8000/v2/models/bert_classifier/infer"
RAW_RESULTS_DIR = REPO_ROOT / "results" / "raw"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rows(path: Path, num_samples: int) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            message = row.get("message", "").strip()
            if not message:
                continue
            rows.append(
                {
                    "message": message,
                    "topic": row.get("topic"),
                    "source_row": row.get("source_row"),
                }
            )
            if len(rows) >= num_samples:
                break
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Triton single-item benchmark without dynamic batching")
    parser.add_argument("--triton-url", default=DEFAULT_TRITON_URL)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--scenario-path", default=str(DEFAULT_SCENARIO_PATH))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--workload", default="bert_classifier")
    parser.add_argument("--serving-mode", default="triton_onnx_no_batch")
    parser.add_argument("--model-name", default="rubert_topic_classifier_triton_onnx")
    parser.add_argument("--hardware", default="unknown")
    parser.add_argument("--notes", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    scenario_path = Path(args.scenario_path).resolve()
    run_id = str(uuid.uuid4())
    timestamp = utc_now_iso()

    client = TritonBertClient(
        triton_url=args.triton_url,
        model_dir=model_dir,
        max_length=args.max_length,
        timeout_sec=args.timeout_sec,
    )

    warmup_rows = load_rows(scenario_path, args.warmup_samples) if args.warmup_samples > 0 else []
    measured_rows = load_rows(scenario_path, args.num_samples)

    for row in warmup_rows:
        client.predict([row["message"]])

    started_total = time.perf_counter()
    request_rows = []

    for idx, row in enumerate(measured_rows):
        started = time.perf_counter()
        predictions = client.predict([row["message"]])
        latency_ms = (time.perf_counter() - started) * 1000
        prediction = predictions[0]

        request_row = BenchmarkRequestResult(
            run_id=run_id,
            timestamp=timestamp,
            workload=args.workload,
            serving_mode=args.serving_mode,
            model_name=args.model_name,
            hardware=args.hardware,
            batch_size=1,
            concurrency=1,
            request_index=idx,
            success=True,
            status_code=200,
            wall_latency_ms=latency_ms,
            api_latency_ms=None,
            predicted_topic=prediction["topic"],
            confidence=prediction["confidence"],
            error_type=None,
            input_topic=row.get("topic"),
            source_row=row.get("source_row"),
            message_len_chars=len(row["message"]),
            message_preview=row["message"][:120],
        )
        request_rows.append(request_row)

        if not args.quiet:
            print(
                "sample=%d latency_ms=%.2f pred_id=%d"
                % (idx, latency_ms, prediction["class_id"])
            )

    total_wall_sec = time.perf_counter() - started_total
    result_dicts = [row.to_dict() for row in request_rows]
    summary = summarize_run(
        run_id=run_id,
        timestamp=timestamp,
        workload=args.workload,
        serving_mode=args.serving_mode,
        model_name=args.model_name,
        hardware=args.hardware,
        batch_size=1,
        concurrency=1,
        total_requests=len(result_dicts),
        total_http_requests=len(result_dicts),
        notes=args.notes,
        endpoint="triton_http_v2_infer",
        dataset_path=str(scenario_path),
        timeout_sec=args.timeout_sec,
        warmup_requests=args.warmup_samples,
        total_wall_sec=total_wall_sec,
        results=result_dicts,
    ).to_dict()

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_RESULTS_DIR / f"{run_id}_requests.jsonl"
    write_request_results(raw_path, request_rows)
    append_summary_row(SUMMARY_TABLE_PATH, summary)

    print("\n=== TRITON NO-BATCH SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nRaw request results: {raw_path}")
    print(f"Summary table: {SUMMARY_TABLE_PATH}")


if __name__ == "__main__":
    main()
