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
WORKLOADS_ROOT = RAG_ROOT.parent
if str(DIRECT_DIR) not in sys.path:
    sys.path.append(str(DIRECT_DIR))
if str(WORKLOADS_ROOT) not in sys.path:
    sys.path.append(str(WORKLOADS_ROOT))
if str(RAG_ROOT) not in sys.path:
    sys.path.append(str(RAG_ROOT))

from rag_pipeline import (  # noqa: E402
    CHUNKS_PATH,
    EMBEDDING_MODEL_DEFAULT,
    LLM_DEFAULT,
    RERANK_BACKEND_DEFAULT,
    RERANKER_DEFAULT,
    TOP_K_AFTER_RERANK,
    TOP_K_EACH,
    _load_dotenv,
    build_context,
    build_context_limited,
    elapsed_ms,
    load_chunks,
    merge_candidates,
    now_timestamp,
    resolve_reranker_backend,
    retrieve_bm25,
    retrieve_dense,
    rerank_hf,
    run_llm,
    tokenize_ru,
)
from qdrant_client import QdrantClient  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from workloads.small_llm.llm_inference import get_local_chat_backend  # noqa: E402

BENCHMARK_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_ROOT / "results"
DEFAULT_QUERIES = BENCHMARK_ROOT / "queries.sample.jsonl"


def pctl(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("No values for percentile")
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))
    return vals[idx]


def load_queries(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query = (row.get("query") or "").strip()
        if not query:
            raise ValueError(f"Missing 'query' in benchmark input row: {row}")
        rows.append(
            {
                "query_id": row.get("query_id") or f"q{len(rows)+1}",
                "query": query,
                "notes": row.get("notes", ""),
            }
        )
    if not rows:
        raise ValueError(f"No queries found in {path}")
    return rows


def summarize(records: list[dict], run_meta: dict) -> dict:
    successful = [r for r in records if r["success"]]
    summary = {
        **run_meta,
        "num_queries": len(records),
        "unique_queries": len({r["query_id"] for r in records}),
        "successful_queries": len(successful),
        "success_rate": (len(successful) / len(records)) if records else 0.0,
    }
    if not successful:
        return summary

    total_latencies = [r["timings_ms"]["total_ms"] for r in successful]
    summary["latency_ms"] = {
        "mean": round(statistics.mean(total_latencies), 3),
        "p50": round(statistics.median(total_latencies), 3),
        "p95": round(pctl(total_latencies, 0.95), 3),
        "max": round(max(total_latencies), 3),
    }

    stage_keys = sorted(
        {k for r in successful for k in r["timings_ms"].keys() if k != "total_ms"}
    )
    summary["stage_stats_ms"] = {}
    for key in stage_keys:
        vals = [r["timings_ms"][key] for r in successful if key in r["timings_ms"]]
        summary["stage_stats_ms"][key] = {
            "mean": round(statistics.mean(vals), 3),
            "p50": round(statistics.median(vals), 3),
            "p95": round(pctl(vals, 0.95), 3),
            "max": round(max(vals), 3),
        }
    return summary


def run_benchmark(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    _load_dotenv()

    embed_model = args.embedding_model
    reranker_model = args.reranker_model
    reranker_backend = resolve_reranker_backend(args.reranker_backend)
    llm_model = args.llm_model
    qdrant = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key or None)
    llm_backend = None
    if args.mode == "full":
        llm_backend = get_local_chat_backend(model_name=llm_model)

    chunks = load_chunks(args.chunks)
    tokenized = [tokenize_ru(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    queries = load_queries(args.queries)

    run_id = str(uuid.uuid4())
    run_meta = {
        "run_id": run_id,
        "timestamp": now_timestamp(),
        "mode": args.mode,
        "queries_path": str(args.queries.resolve()),
        "chunks_path": str(args.chunks.resolve()),
        "collection": args.collection,
        "embedding_model": embed_model,
        "reranker_model": reranker_model,
        "reranker_backend": reranker_backend,
        "llm_model": llm_model if args.mode == "full" else None,
        "top_each": args.top_each,
        "top_final": args.top_final,
        "max_context_chars": args.max_context_chars,
        "llm_max_tokens": args.llm_max_tokens,
        "qdrant_url": args.qdrant_url,
    }

    records: list[dict] = []
    for repeat_index in range(1, args.repeats + 1):
        for item in queries:
            query = item["query"]
            total_start = time.perf_counter()
            record = {
                "run_id": run_id,
                "timestamp": now_timestamp(),
                "mode": args.mode,
                "repeat_index": repeat_index,
                "query_id": item["query_id"],
                "query": query,
                "notes": item["notes"],
                "success": False,
                "error": None,
                "timings_ms": {},
            }
            try:
                bm25_start = time.perf_counter()
                bm25_hits = retrieve_bm25(bm25, chunks, query, args.top_each)
                bm25_ms = elapsed_ms(bm25_start)

                dense_start = time.perf_counter()
                dense_hits = retrieve_dense(qdrant, args.collection, embed_model, query, args.top_each)
                dense_ms = elapsed_ms(dense_start)

                merge_start = time.perf_counter()
                merged = merge_candidates(bm25_hits, dense_hits)
                merge_ms = elapsed_ms(merge_start)

                timings = {
                    "bm25_ms": bm25_ms,
                    "dense_ms": dense_ms,
                    "merge_ms": merge_ms,
                }
                top = merged

                if args.mode in {"retrieval_rerank", "full"}:
                    rerank_start = time.perf_counter()
                    reranked = rerank_hf(reranker_model, query, merged, backend_name=reranker_backend)
                    timings["rerank_ms"] = elapsed_ms(rerank_start)
                    top = [c for c, _ in reranked[: args.top_final]]

                if args.mode == "full":
                    context_start = time.perf_counter()
                    context = (
                        build_context_limited(top, args.max_context_chars)
                        if args.max_context_chars > 0
                        else build_context(top)
                    )
                    timings["context_build_ms"] = elapsed_ms(context_start)

                    llm_start = time.perf_counter()
                    answer = run_llm(llm_backend, llm_model, query, context, args.llm_max_tokens)
                    timings["llm_ms"] = elapsed_ms(llm_start)
                    record["answer_preview"] = answer[:300]
                    record["context_chars"] = len(context)

                timings["total_ms"] = elapsed_ms(total_start)
                record["timings_ms"] = timings
                record["success"] = True
                record["num_bm25_hits"] = len(bm25_hits)
                record["num_dense_hits"] = len(dense_hits)
                record["num_merged_hits"] = len(merged)
                record["top_chunk_ids"] = [int(c["chunk_id"]) for c in top[: args.top_final]]
            except Exception as exc:
                record["error"] = str(exc)
                record["timings_ms"] = {"total_ms": elapsed_ms(total_start)}
            records.append(record)

    return run_meta, records


def write_results(run_meta: dict, records: list[dict]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_{run_meta['mode']}_{run_meta['run_id']}"
    raw_path = RESULTS_DIR / f"{stem}.jsonl"
    summary_path = RESULTS_DIR / f"{stem}_summary.json"

    with raw_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(records, run_meta)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw_path, summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Benchmark the local direct RAG workload")
    ap.add_argument(
        "--mode",
        choices=("retrieval", "retrieval_rerank", "full"),
        required=True,
        help="Benchmark retrieval only, retrieval plus rerank, or the full direct RAG path.",
    )
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    ap.add_argument("--collection", default="labor_code_tk_e5")
    ap.add_argument("--top-each", type=int, default=TOP_K_EACH)
    ap.add_argument("--top-final", type=int, default=TOP_K_AFTER_RERANK)
    ap.add_argument("--max-context-chars", type=int, default=0)
    ap.add_argument("--llm-max-tokens", type=int, default=512)
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT)
    ap.add_argument("--reranker-model", default=RERANKER_DEFAULT)
    ap.add_argument("--reranker-backend", default=RERANK_BACKEND_DEFAULT)
    ap.add_argument("--llm-model", default=LLM_DEFAULT)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-api-key", default="")
    ap.add_argument("--repeats", type=int, default=1)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_meta, records = run_benchmark(args)
    raw_path, summary_path = write_results(run_meta, records)
    print(json.dumps({"raw_results": str(raw_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
