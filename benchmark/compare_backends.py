import argparse
import csv
import json
import math
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark.load_test import DEFAULT_SCENARIO_PATH, load_messages, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "workloads" / "bert_classifier" / "artifacts" / "model"
DEFAULT_ONNX_PATH = (
    REPO_ROOT / "workloads" / "bert_classifier" / "onnx_export" / "bert_classifier.onnx"
)
TABLES_DIR = REPO_ROOT / "results" / "tables"
RAW_DIR = REPO_ROOT / "results" / "raw"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BackendBenchRow:
    run_id: str
    timestamp: str
    backend: str
    provider: str
    model_name: str
    max_length: int
    batch_size: int
    num_texts: int
    warmup_batches: int
    timed_batches: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_items_per_sec: float
    notes: str


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def load_texts(dataset_path: str, num_requests: int) -> List[str]:
    messages = load_messages(dataset_path, num_requests)
    return [row["message"] for row in messages]


def measure_pytorch(
    *,
    texts: List[str],
    model_dir: Path,
    batch_size: int,
    max_length: int,
    warmup_batches: int,
    device_name: str,
) -> BackendBenchRow:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device(device_name)
    model.to(device)
    model.eval()

    batches = batched(texts, batch_size)
    run_id = str(uuid.uuid4())
    timestamp = utc_now_iso()

    def run_batch(batch_texts: List[str]) -> float:
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            _ = model(**encoded).logits
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000

    for batch in batches[:warmup_batches]:
        run_batch(batch)

    latencies = [run_batch(batch) for batch in batches]
    latencies_sorted = sorted(latencies)
    total_items = len(texts)
    total_sec = sum(latencies) / 1000.0

    return BackendBenchRow(
        run_id=run_id,
        timestamp=timestamp,
        backend="pytorch",
        provider=device.type,
        model_name="rubert_topic_classifier",
        max_length=max_length,
        batch_size=batch_size,
        num_texts=total_items,
        warmup_batches=warmup_batches,
        timed_batches=len(batches),
        mean_ms=round(statistics.mean(latencies), 4),
        p50_ms=round(percentile(latencies_sorted, 50), 4),
        p95_ms=round(percentile(latencies_sorted, 95), 4),
        p99_ms=round(percentile(latencies_sorted, 99), 4),
        min_ms=round(min(latencies_sorted), 4),
        max_ms=round(max(latencies_sorted), 4),
        throughput_items_per_sec=round(total_items / total_sec, 4) if total_sec else 0.0,
        notes="direct model benchmark",
    )


def measure_onnx(
    *,
    texts: List[str],
    model_dir: Path,
    onnx_path: Path,
    batch_size: int,
    max_length: int,
    warmup_batches: int,
    provider: str,
) -> BackendBenchRow:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required: pip install onnxruntime") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise SystemExit(
            "Requested ONNX provider %s is not available. Available providers: %s"
            % (provider, ", ".join(available_providers))
        )

    session = ort.InferenceSession(str(onnx_path), providers=[provider])
    input_names = [item.name for item in session.get_inputs()]
    batches = batched(texts, batch_size)
    run_id = str(uuid.uuid4())
    timestamp = utc_now_iso()

    def run_batch(batch_texts: List[str]) -> float:
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        ort_inputs = {name: encoded[name].cpu().numpy() for name in input_names}
        started = time.perf_counter()
        _ = session.run(["logits"], ort_inputs)
        return (time.perf_counter() - started) * 1000

    for batch in batches[:warmup_batches]:
        run_batch(batch)

    latencies = [run_batch(batch) for batch in batches]
    latencies_sorted = sorted(latencies)
    total_items = len(texts)
    total_sec = sum(latencies) / 1000.0

    return BackendBenchRow(
        run_id=run_id,
        timestamp=timestamp,
        backend="onnxruntime",
        provider=provider,
        model_name="rubert_topic_classifier",
        max_length=max_length,
        batch_size=batch_size,
        num_texts=total_items,
        warmup_batches=warmup_batches,
        timed_batches=len(batches),
        mean_ms=round(statistics.mean(latencies), 4),
        p50_ms=round(percentile(latencies_sorted, 50), 4),
        p95_ms=round(percentile(latencies_sorted, 95), 4),
        p99_ms=round(percentile(latencies_sorted, 99), 4),
        min_ms=round(min(latencies_sorted), 4),
        max_ms=round(max(latencies_sorted), 4),
        throughput_items_per_sec=round(total_items / total_sec, 4) if total_sec else 0.0,
        notes="direct model benchmark",
    )


def append_rows(path: Path, rows: List[BackendBenchRow]) -> None:
    fieldnames = list(BackendBenchRow.__dataclass_fields__.keys())
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_raw_json(path: Path, rows: List[BackendBenchRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare direct PyTorch vs ONNX backend inference")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--onnx-path", type=str, default=str(DEFAULT_ONNX_PATH))
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_SCENARIO_PATH))
    parser.add_argument("--num-texts", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup-batches", type=int, default=4)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["pytorch_cpu", "pytorch_cuda", "onnx_cpu"],
        help="Choose from pytorch_cpu, pytorch_cuda, onnx_cpu, onnx_cuda",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    onnx_path = Path(args.onnx_path).expanduser().resolve()
    texts = load_texts(args.dataset_path, args.num_texts)

    rows = []
    for batch_size in args.batch_sizes:
        if "pytorch_cpu" in args.backends:
            rows.append(
                measure_pytorch(
                    texts=texts,
                    model_dir=model_dir,
                    batch_size=batch_size,
                    max_length=args.max_length,
                    warmup_batches=args.warmup_batches,
                    device_name="cpu",
                )
            )

        if "pytorch_cuda" in args.backends and torch.cuda.is_available():
            rows.append(
                measure_pytorch(
                    texts=texts,
                    model_dir=model_dir,
                    batch_size=batch_size,
                    max_length=args.max_length,
                    warmup_batches=args.warmup_batches,
                    device_name="cuda",
                )
            )

        if "onnx_cpu" in args.backends:
            rows.append(
                measure_onnx(
                    texts=texts,
                    model_dir=model_dir,
                    onnx_path=onnx_path,
                    batch_size=batch_size,
                    max_length=args.max_length,
                    warmup_batches=args.warmup_batches,
                    provider="CPUExecutionProvider",
                )
            )

        if "onnx_cuda" in args.backends:
            rows.append(
                measure_onnx(
                    texts=texts,
                    model_dir=model_dir,
                    onnx_path=onnx_path,
                    batch_size=batch_size,
                    max_length=args.max_length,
                    warmup_batches=args.warmup_batches,
                    provider="CUDAExecutionProvider",
                )
            )

    table_path = TABLES_DIR / "backend_comparison.csv"
    raw_path = RAW_DIR / ("backend_comparison_%s.jsonl" % str(uuid.uuid4()))
    append_rows(table_path, rows)
    write_raw_json(raw_path, rows)

    print("=== BACKEND COMPARISON ===")
    for row in rows:
        print(
            "%s | provider=%s | batch=%d | mean_ms=%.4f | p50_ms=%.4f | p95_ms=%.4f | throughput_items_per_sec=%.4f"
            % (
                row.backend,
                row.provider,
                row.batch_size,
                row.mean_ms,
                row.p50_ms,
                row.p95_ms,
                row.throughput_items_per_sec,
            )
        )

    print("\nRaw results:", raw_path)
    print("Table:", table_path)


if __name__ == "__main__":
    main()
