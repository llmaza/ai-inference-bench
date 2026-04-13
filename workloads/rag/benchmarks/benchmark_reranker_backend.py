#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
DIRECT_DIR = RAG_ROOT / "direct"
if str(DIRECT_DIR) not in sys.path:
    sys.path.append(str(DIRECT_DIR))
if str(RAG_ROOT) not in sys.path:
    sys.path.append(str(RAG_ROOT))

from rag_pipeline import (  # noqa: E402
    CHUNKS_PATH,
    EMBEDDING_MODEL_DEFAULT,
    RERANKER_DEFAULT,
    TOP_K_EACH,
    _load_dotenv,
    elapsed_ms,
    load_chunks,
    merge_candidates,
    now_timestamp,
    retrieve_bm25,
    retrieve_dense,
    tokenize_ru,
)
from qdrant_client import QdrantClient  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from reranker.backend import score_pairs  # noqa: E402
from benchmark_direct_rag import DEFAULT_QUERIES, load_queries, pctl  # noqa: E402

BENCHMARK_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_ROOT / "results"


def summarize(records: list[dict], meta: dict) -> dict:
    successful = [r for r in records if r["success"]]
    summary = {
        **meta,
        "num_queries": len(records),
        "successful_queries": len(successful),
        "success_rate": (len(successful) / len(records)) if records else 0.0,
    }
    if not successful:
        return summary
    for backend in sorted({r["backend"] for r in successful}):
        rows = [r for r in successful if r["backend"] == backend]
        vals = [r["timings_ms"]["rerank_ms"] for r in rows]
        summary.setdefault("backend_stats_ms", {})[backend] = {
            "mean": round(statistics.mean(vals), 3),
            "p50": round(statistics.median(vals), 3),
            "p95": round(pctl(vals, 0.95), 3),
            "max": round(max(vals), 3),
        }
    return summary


def run_benchmark(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    _load_dotenv()
    queries = load_queries(args.queries)
    chunks = load_chunks(args.chunks)
    tokenized = [tokenize_ru(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    qdrant = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key or None)
    run_id = str(uuid.uuid4())
    meta = {
        "run_id": run_id,
        "timestamp": now_timestamp(),
        "mode": "reranker_backend",
        "queries_path": str(args.queries.resolve()),
        "chunks_path": str(args.chunks.resolve()),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "backends": list(args.backends),
        "repeats": args.repeats,
        "top_each": args.top_each,
    }
    records: list[dict] = []
    for repeat_index in range(1, args.repeats + 1):
        for item in queries:
            query = item["query"]
            bm25_hits = retrieve_bm25(bm25, chunks, query, args.top_each)
            dense_hits = retrieve_dense(qdrant, args.collection, args.embedding_model, query, args.top_each)
            merged = merge_candidates(bm25_hits, dense_hits)
            texts = [c["text"][:6000] for c in merged]
            for backend in args.backends:
                start = time.perf_counter()
                record = {
                    "run_id": run_id,
                    "timestamp": now_timestamp(),
                    "mode": "reranker_backend",
                    "backend": backend,
                    "repeat_index": repeat_index,
                    "query_id": item["query_id"],
                    "query": query,
                    "notes": item["notes"],
                    "success": False,
                    "error": None,
                    "candidate_count": len(texts),
                    "timings_ms": {},
                }
                try:
                    scores = score_pairs(
                        query,
                        texts,
                        model_name=args.reranker_model,
                        batch_size=args.batch_size,
                        backend_name=backend,
                    )
                    if len(scores) != len(texts):
                        raise RuntimeError(f"Backend {backend} returned {len(scores)} scores for {len(texts)} passages")
                    record["success"] = True
                    record["top_score"] = max(scores) if scores else None
                    record["timings_ms"] = {"rerank_ms": elapsed_ms(start)}
                except Exception as exc:
                    record["error"] = str(exc)
                    record["timings_ms"] = {"rerank_ms": elapsed_ms(start)}
                records.append(record)
    return meta, records


def write_results(meta: dict, records: list[dict]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_reranker_backend_{meta['run_id']}"
    raw_path = RESULTS_DIR / f"{stem}.jsonl"
    summary_path = RESULTS_DIR / f"{stem}_summary.json"
    with raw_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summarize(records, meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw_path, summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Benchmark torch vs ONNX reranker backends")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    ap.add_argument("--collection", default="labor_code_tk_e5")
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT)
    ap.add_argument("--reranker-model", default=RERANKER_DEFAULT)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-api-key", default="")
    ap.add_argument("--top-each", type=int, default=TOP_K_EACH)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--backends", nargs="+", default=["torch", "onnx"])
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    meta, records = run_benchmark(args)
    raw_path, summary_path = write_results(meta, records)
    print(json.dumps({"raw_results": str(raw_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
