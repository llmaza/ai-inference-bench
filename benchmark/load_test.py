import argparse
import csv
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from benchmark.metrics import summarize_run
from benchmark.schema import BenchmarkRequestResult, BenchmarkRunSummary


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_PATH = REPO_ROOT / "benchmark" / "scenarios" / "bert_inputs.jsonl"
RAW_RESULTS_DIR = REPO_ROOT / "results" / "raw"
TABLES_DIR = REPO_ROOT / "results" / "tables"
SUMMARY_TABLE_PATH = TABLES_DIR / "benchmark_runs.csv"
SUMMARY_FIELDNAMES = list(BenchmarkRunSummary.__dataclass_fields__.keys())

RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_messages(dataset_path: Optional[str], num_requests: int) -> List[dict]:
    path = Path(dataset_path) if dataset_path else DEFAULT_SCENARIO_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "message" not in row:
                    raise ValueError("Each JSONL row must contain a 'message' field")
                records.append(
                    {
                        "message": row["message"],
                        "topic": row.get("topic"),
                        "source_row": row.get("source_row"),
                    }
                )
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    records.append({"message": text, "topic": None, "source_row": None})

    if not records:
        raise ValueError("Dataset is empty")

    return [records[i % len(records)] for i in range(num_requests)]


def chunk_records(records: List[dict], batch_size: int) -> List[List[dict]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def send_request(session: requests.Session, url: str, message: str, timeout_sec: float) -> dict:
    wall_started = time.perf_counter()
    try:
        response = session.post(url, json={"message": message}, timeout=timeout_sec)
        wall_ms = (time.perf_counter() - wall_started) * 1000

        result = {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "wall_latency_ms": wall_ms,
            "api_latency_ms": None,
            "predicted_topic": None,
            "confidence": None,
            "error_type": None,
        }

        if response.status_code == 200:
            payload = response.json()
            result["api_latency_ms"] = payload.get("latency_ms")
            result["predicted_topic"] = payload.get("topic")
            result["confidence"] = payload.get("confidence")
        else:
            try:
                payload = response.json()
                result["error_type"] = payload.get("detail", f"HTTP_{response.status_code}")
            except Exception:
                result["error_type"] = f"HTTP_{response.status_code}"

        return result
    except requests.Timeout:
        return {
            "success": False,
            "status_code": None,
            "wall_latency_ms": (time.perf_counter() - wall_started) * 1000,
            "api_latency_ms": None,
            "predicted_topic": None,
            "confidence": None,
            "error_type": "timeout",
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "wall_latency_ms": (time.perf_counter() - wall_started) * 1000,
            "api_latency_ms": None,
            "predicted_topic": None,
            "confidence": None,
            "error_type": type(e).__name__,
        }


def send_batch_request(
    session: requests.Session,
    url: str,
    messages: List[str],
    timeout_sec: float,
) -> dict:
    wall_started = time.perf_counter()
    try:
        response = session.post(url, json={"messages": messages}, timeout=timeout_sec)
        wall_ms = (time.perf_counter() - wall_started) * 1000

        result = {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "wall_latency_ms": wall_ms,
            "api_latency_ms": None,
            "predictions": None,
            "error_type": None,
        }

        if response.status_code == 200:
            payload = response.json()
            result["api_latency_ms"] = payload.get("latency_ms")
            result["predictions"] = payload.get("predictions", [])
        else:
            try:
                payload = response.json()
                result["error_type"] = payload.get("detail", f"HTTP_{response.status_code}")
            except Exception:
                result["error_type"] = f"HTTP_{response.status_code}"

        return result
    except requests.Timeout:
        return {
            "success": False,
            "status_code": None,
            "wall_latency_ms": (time.perf_counter() - wall_started) * 1000,
            "api_latency_ms": None,
            "predictions": None,
            "error_type": "timeout",
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "wall_latency_ms": (time.perf_counter() - wall_started) * 1000,
            "api_latency_ms": None,
            "predictions": None,
            "error_type": type(e).__name__,
        }


def write_request_results(path: Path, rows: List[BenchmarkRequestResult]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def append_summary_row(path: Path, row: dict) -> None:
    file_exists = path.exists()
    if file_exists:
        with path.open("r", encoding="utf-8", newline="") as f:
            first_line = f.readline().strip()
        existing_header = first_line.split(",") if first_line else []
        if existing_header != SUMMARY_FIELDNAMES:
            path.unlink()
            file_exists = False

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def replay_requests(messages: List[dict], args, include_results: bool, run_id: str, timestamp: str):
    results = []  # type: List[BenchmarkRequestResult]
    results_lock = threading.Lock()
    wall_started = time.perf_counter()
    batches = chunk_records(messages, args.batch_size)
    use_batch_endpoint = args.endpoint == "/predict_batch"

    def task(batch_index: int, batch_records: List[dict]) -> None:
        with requests.Session() as session:
            if not use_batch_endpoint:
                response_result = send_request(
                    session=session,
                    url=args.base_url.rstrip("/") + args.endpoint,
                    message=batch_records[0]["message"],
                    timeout_sec=args.timeout_sec,
                )
            else:
                response_result = send_batch_request(
                    session=session,
                    url=args.base_url.rstrip("/") + args.endpoint,
                    messages=[record["message"] for record in batch_records],
                    timeout_sec=args.timeout_sec,
                )

        if not include_results:
            return

        predictions = response_result.get("predictions") or []
        for item_index, record in enumerate(batch_records):
            prediction = predictions[item_index] if item_index < len(predictions) else {}
            row = BenchmarkRequestResult(
                run_id=run_id,
                timestamp=timestamp,
                workload=args.workload,
                serving_mode=args.serving_mode,
                model_name=args.model_name,
                hardware=args.hardware,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                request_index=(batch_index * args.batch_size) + item_index,
                success=response_result["success"],
                status_code=response_result["status_code"],
                wall_latency_ms=response_result["wall_latency_ms"],
                api_latency_ms=response_result["api_latency_ms"],
                predicted_topic=prediction.get("topic") if response_result["success"] else None,
                confidence=prediction.get("confidence") if response_result["success"] else None,
                error_type=response_result["error_type"],
                input_topic=record.get("topic"),
                source_row=record.get("source_row"),
                message_len_chars=len(record["message"]),
                message_preview=record["message"][:120],
            )

            with results_lock:
                results.append(row)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(task, idx, batch) for idx, batch in enumerate(batches)]
        for future in as_completed(futures):
            future.result()

    total_wall_sec = time.perf_counter() - wall_started
    return sorted(results, key=lambda item: item.request_index), total_wall_sec, len(batches)


def run_benchmark(args) -> Tuple[Path, Path, dict]:
    run_id = str(uuid.uuid4())
    timestamp = utc_now_iso()
    dataset_path = str(Path(args.dataset_path).resolve()) if args.dataset_path else str(DEFAULT_SCENARIO_PATH.resolve())
    if args.batch_size > 1 and args.endpoint != "/predict_batch":
        raise ValueError("Batch size > 1 requires --endpoint /predict_batch")
    messages = load_messages(args.dataset_path, args.num_requests)
    warmup_messages = messages[: args.warmup_requests] if args.warmup_requests > 0 else []

    if warmup_messages:
        _, _, _ = replay_requests(
            messages=warmup_messages,
            args=args,
            include_results=False,
            run_id=run_id,
            timestamp=timestamp,
        )

    results, total_wall_sec, total_http_requests = replay_requests(
        messages=messages,
        args=args,
        include_results=True,
        run_id=run_id,
        timestamp=timestamp,
    )

    result_dicts = [row.to_dict() for row in results]
    summary = summarize_run(
        run_id=run_id,
        timestamp=timestamp,
        workload=args.workload,
        serving_mode=args.serving_mode,
        model_name=args.model_name,
        hardware=args.hardware,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        total_requests=len(result_dicts),
        total_http_requests=total_http_requests,
        notes=args.notes,
        endpoint=args.endpoint,
        dataset_path=dataset_path,
        timeout_sec=args.timeout_sec,
        warmup_requests=args.warmup_requests,
        total_wall_sec=total_wall_sec,
        results=result_dicts,
    ).to_dict()

    raw_path = RAW_RESULTS_DIR / f"{run_id}_requests.jsonl"
    table_path = SUMMARY_TABLE_PATH
    write_request_results(raw_path, results)
    append_summary_row(table_path, summary)
    return raw_path, table_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BERT benchmark runner for FastAPI baseline")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", type=str, default="/predict")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_SCENARIO_PATH))
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--workload", type=str, default="bert_classifier")
    parser.add_argument("--serving-mode", type=str, default="fastapi_baseline")
    parser.add_argument("--model-name", type=str, default="rubert_topic_classifier")
    parser.add_argument("--hardware", type=str, default="unknown")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--notes", type=str, default="")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    raw_path, table_path, summary = run_benchmark(args)

    print("=== BENCHMARK SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nRaw request results: {raw_path}")
    print(f"Summary table: {table_path}")


if __name__ == "__main__":
    main()
