#!/usr/bin/env python3
"""
RAG по Трудовому кодексу (RU):
  1) BM25 — топ-50 по лексике
  2) Косинус / Qdrant — топ-50 по E5 (query:)
  3) Объединение кандидатов по chunk_id
  4) Локальный реранкер BGE — только его скоры → топ-25
  5) Ответ через локальную instruct-модель

.env: QDRANT_URL, QDRANT_API_KEY (опц.), HF_RERANKER_MODEL, HF_LLM_MODEL,
      EMBEDDING_MODEL, QDRANT_COLLECTION

Запуск CLI:
  .venv/bin/python rag_pipeline.py "Какой срок испытательного срока?"
  .venv/bin/python rag_pipeline.py "..." --verbose --max-context-chars 24000

Веб-UI (Streamlit):
  .venv/bin/streamlit run rag_ui.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

from qdrant_client import QdrantClient

# --- константы (переопределяются через .env) ---
TOP_K_EACH = 50
TOP_K_AFTER_RERANK = 25
MAX_PASSAGE_CHARS_RERANK = 6000
RERANK_BATCH = 12
CHUNKS_PATH = Path(__file__).resolve().parent / "assets" / "Трудовой_Кодекс_chunks.jsonl"
RAG_ROOT = Path(__file__).resolve().parents[1]
WORKLOADS_ROOT = RAG_ROOT.parent
if str(WORKLOADS_ROOT) not in sys.path:
    sys.path.append(str(WORKLOADS_ROOT))
if str(RAG_ROOT) not in sys.path:
    sys.path.append(str(RAG_ROOT))

from embed.local_backend import (  # noqa: E402
    EMBEDDING_MODEL_DEFAULT,
    VECTOR_SIZE_DEFAULT,
    encode_query_e5_local,
)
from reranker.backend import (  # noqa: E402
    RERANK_BACKEND_DEFAULT,
    RERANKER_DEFAULT,
    resolve_reranker_backend,
    score_pairs as score_reranker_pairs,
)
from workloads.small_llm.llm_inference import (  # noqa: E402
    LOCAL_LLM_DEFAULT,
    get_local_chat_backend,
)

VECTOR_DIM = VECTOR_SIZE_DEFAULT
LLM_DEFAULT = LOCAL_LLM_DEFAULT
BENCHMARK_LOG_PATH = RAG_ROOT / "benchmarks" / "direct_rag_runs.jsonl"

SYSTEM_PROMPT_RU = """Ты юридический ассистент по трудовому праву Республики Казахстан.
Отвечай на русском языке, опираясь только на предоставленный контекст — фрагменты Трудового кодекса РК.
Если в контексте нет достаточных оснований для ответа, так и скажи и укажи, какие нормы обычно нужно проверить.
Не выдумывай статьи и формулировки: при ссылках указывай номер статьи из контекста (метаданные/текст).
Ответ структурируй кратко: сначала вывод, при необходимости перечисли релевантные статьи.
Формулируй сразу итог для пользователя; не уводи ответ в длинные пошаговые рассуждения."""


def _load_dotenv() -> None:
    env = RAG_ROOT / ".env"
    if load_dotenv is not None and env.is_file():
        load_dotenv(env)


def now_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def append_benchmark_log(record: dict) -> None:
    BENCHMARK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BENCHMARK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tokenize_ru(s: str) -> list[str]:
    return re.findall(r"[\w\d]+", s.lower(), flags=re.UNICODE)


def load_chunks(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def chunk_record_from_payload(payload: dict) -> dict:
    md_keys = {
        "document",
        "part",
        "section",
        "chapter",
        "article_number",
        "article_title",
        "point_index",
        "point_kind",
        "point_label",
        "breadcrumb",
        "chunk_in_point",
        "chunks_in_point",
        "tokenizer",
        "max_tokens",
        "overlap_tokens",
    }
    meta = {k: payload[k] for k in md_keys if k in payload}
    return {
        "chunk_id": payload["chunk_id"],
        "text": payload.get("text", ""),
        "metadata": meta,
    }


def embed_query_e5(query: str, model: str) -> list[float]:
    vec = encode_query_e5_local(query, model_name=model)
    if len(vec) != VECTOR_DIM:
        raise RuntimeError(f"Размерность эмбеддинга запроса {len(vec)}, ожидалось {VECTOR_DIM}")
    return vec


def retrieve_bm25(bm25: BM25Okapi, chunks: list[dict], query: str, k: int) -> list[dict]:
    scores = bm25.get_scores(tokenize_ru(query))
    idx = np.argsort(scores)[::-1][:k]
    return [chunks[int(i)] for i in idx]


def retrieve_dense(
    qclient: QdrantClient,
    collection: str,
    embed_model: str,
    query: str,
    k: int,
) -> list[dict]:
    qvec = embed_query_e5(query, embed_model)
    # qdrant-client ≥1.12: вместо search() используется query_points()
    resp = qclient.query_points(
        collection_name=collection,
        query=qvec,
        limit=k,
        with_payload=True,
    )
    out: list[dict] = []
    for h in resp.points:
        pl = h.payload
        if not pl:
            continue
        payload_dict = dict(pl) if not isinstance(pl, dict) else pl
        out.append(chunk_record_from_payload(payload_dict))
    return out


def merge_candidates(bm25_hits: list[dict], dense_hits: list[dict]) -> list[dict]:
    by_id: dict[int, dict] = {}
    for c in bm25_hits + dense_hits:
        cid = int(c["chunk_id"])
        if cid not in by_id:
            by_id[cid] = c
    return list(by_id.values())


def rerank_hf(
    reranker_model: str,
    query: str,
    candidates: list[dict],
    backend_name: str | None = None,
) -> list[tuple[dict, float]]:
    """Скоры локального реранкера для пар query-passsage."""
    texts = [c["text"][:MAX_PASSAGE_CHARS_RERANK] for c in candidates]
    scores_all = score_reranker_pairs(
        query,
        texts,
        model_name=reranker_model,
        batch_size=RERANK_BATCH,
        backend_name=backend_name,
    )
    if len(scores_all) != len(candidates):
        raise RuntimeError(
            f"Реранкер вернул {len(scores_all)} скоров при {len(candidates)} документах"
        )
    scored = list(zip(candidates, scores_all))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _chunk_block(i: int, c: dict) -> str:
    md = c.get("metadata") or {}
    head = md.get("breadcrumb") or (
        f"Статья {md.get('article_number', '?')}. {md.get('article_title', '')}"
    )
    return f"[{i}] {head}\n{c['text']}"


def build_context(top_chunks: list[dict]) -> str:
    parts = [_chunk_block(i, c) for i, c in enumerate(top_chunks, 1)]
    return "\n\n---\n\n".join(parts)


def build_context_limited(top_chunks: list[dict], max_chars: int) -> str:
    """Собирает контекст из чанков по порядку реранка, не превышая max_chars (целые блоки; последний может обрезаться)."""
    if max_chars <= 0:
        return build_context(top_chunks)
    sep = "\n\n---\n\n"
    parts: list[str] = []
    total = 0
    for i, c in enumerate(top_chunks, 1):
        block = _chunk_block(i, c)
        need = len(block) + (len(sep) if parts else 0)
        if total + need <= max_chars:
            parts.append(block)
            total += need
            continue
        room = max_chars - total - (len(sep) if parts else 0)
        if room > 200:
            parts.append(block[:room] + "\n… [фрагмент обрезан по лимиту]")
        break
    return sep.join(parts)


def run_llm_chat(
    llm_backend,
    llm_model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.15,
) -> str:
    """Произвольный список сообщений (system + история + user) для режима чата."""
    return llm_backend.generate(
        messages=messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )


def build_chat_messages_for_rag(
    history: list[dict],
    current_query: str,
    context: str,
    max_prior_messages: int,
) -> list[dict]:
    """
    history — только прошлые user/assistant (без текущего вопроса).
    max_prior_messages: сколько последних сообщений истории включить (0 = только текущий вопрос с RAG).
    """
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT_RU}]
    tail = history[-max_prior_messages:] if max_prior_messages > 0 else []
    for m in tail:
        r = m.get("role")
        if r not in ("user", "assistant"):
            continue
        msgs.append({"role": r, "content": m.get("content", "")})
    msgs.append(
        {
            "role": "user",
            "content": (
                "Фрагменты Трудового кодекса РК для текущего вопроса:\n\n"
                f"{context}\n\n---\n\nТекущий вопрос: {current_query}\n\n"
                "Ответь с опорой на контекст; при необходимости учти предыдущие реплики диалога."
            ),
        }
    )
    return msgs


def retrieve_top_chunks(
    bm25: BM25Okapi,
    chunks: list[dict],
    qdrant: QdrantClient,
    query: str,
    collection: str,
    embed_model: str,
    reranker_model: str,
    reranker_backend: str | None,
    top_each: int,
    top_final: int,
) -> tuple[list[dict], list[tuple[dict, float]], dict[str, float]]:
    bm25_start = time.perf_counter()
    bm25_hits = retrieve_bm25(bm25, chunks, query, top_each)
    bm25_ms = elapsed_ms(bm25_start)

    dense_start = time.perf_counter()
    dense_hits = retrieve_dense(qdrant, collection, embed_model, query, top_each)
    dense_ms = elapsed_ms(dense_start)

    merge_start = time.perf_counter()
    merged = merge_candidates(bm25_hits, dense_hits)
    merge_ms = elapsed_ms(merge_start)

    rerank_start = time.perf_counter()
    reranked = rerank_hf(reranker_model, query, merged, backend_name=reranker_backend)
    rerank_ms = elapsed_ms(rerank_start)

    top = [c for c, _ in reranked[:top_final]]
    timings = {
        "bm25_ms": bm25_ms,
        "dense_ms": dense_ms,
        "merge_ms": merge_ms,
        "rerank_ms": rerank_ms,
    }
    return top, reranked, timings


def run_llm(
    llm_backend,
    llm_model: str,
    question: str,
    context: str,
    max_tokens: int,
) -> str:
    user_msg = (
        "Ниже фрагменты Трудового кодекса РК (контекст для ответа).\n\n"
        f"{context}\n\n"
        f"Вопрос пользователя: {question}\n\n"
        "Дай ответ с опорой на контекст."
    )
    return run_llm_chat(
        llm_backend,
        llm_model,
        [
            {"role": "system", "content": SYSTEM_PROMPT_RU},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.15,
    )


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="RAG: BM25 + Qdrant → HF rerank → Qwen")
    ap.add_argument("question", help="Вопрос на русском")
    ap.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    ap.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "labor_code_tk_e5"),
    )
    ap.add_argument("--top-each", type=int, default=TOP_K_EACH)
    ap.add_argument("--top-final", type=int, default=TOP_K_AFTER_RERANK)
    ap.add_argument("--llm-max-tokens", type=int, default=2048)
    ap.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help="Лимит символов контекста кодекса (0 = без лимита)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    total_start = time.perf_counter()

    embed_model = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT)
    reranker_model = os.environ.get("HF_RERANKER_MODEL", RERANKER_DEFAULT)
    reranker_backend = resolve_reranker_backend(os.environ.get("RERANK_BACKEND", RERANK_BACKEND_DEFAULT))
    llm_model = os.environ.get("HF_LLM_MODEL", LLM_DEFAULT)

    q_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    q_key = os.environ.get("QDRANT_API_KEY") or None

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise SystemExit(f"Пусто: {args.chunks}")

    tokenized = [tokenize_ru(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    qdrant = QdrantClient(url=q_url, api_key=q_key)
    llm_backend = get_local_chat_backend(model_name=llm_model)

    q = args.question
    log_record = {
        "timestamp": now_timestamp(),
        "query": q,
        "chunks_path": str(args.chunks.resolve()),
        "collection": args.collection,
        "retrieval_params": {
            "top_each": args.top_each,
            "top_final": args.top_final,
            "llm_max_tokens": args.llm_max_tokens,
            "max_context_chars": args.max_context_chars,
            "embedding_model": embed_model,
            "reranker_model": reranker_model,
            "reranker_backend": reranker_backend,
            "llm_model": llm_model,
            "qdrant_url": q_url,
        },
    }

    try:
        top, reranked, stage_timings = retrieve_top_chunks(
            bm25,
            chunks,
            qdrant,
            q,
            args.collection,
            embed_model,
            reranker_model,
            reranker_backend,
            args.top_each,
            args.top_final,
        )

        context_start = time.perf_counter()
        context = (
            build_context_limited(top, args.max_context_chars)
            if args.max_context_chars > 0
            else build_context(top)
        )
        context_build_ms = elapsed_ms(context_start)

        if args.verbose:
            print("Реранкер (топ-10):", [(c["chunk_id"], round(s, 4)) for c, s in reranked[:10]])
            print(f"Символов в контексте для LLM: {len(context)}")

        llm_start = time.perf_counter()
        answer = run_llm(llm_backend, llm_model, q, context, args.llm_max_tokens)
        llm_ms = elapsed_ms(llm_start)

        total_ms = elapsed_ms(total_start)
        log_record.update(
            {
                "success": True,
                "error": None,
                "top_chunk_ids": [int(c["chunk_id"]) for c in top],
                "context_chars": len(context),
                "timings_ms": {
                    **stage_timings,
                    "context_build_ms": context_build_ms,
                    "llm_ms": llm_ms,
                    "total_ms": total_ms,
                },
            }
        )
        append_benchmark_log(log_record)

        print("\n========== ОТВЕТ ==========\n")
        print(answer)
        print("\n========== (конец) ==========\n")
    except Exception as exc:
        log_record.update(
            {
                "success": False,
                "error": str(exc),
                "timings_ms": {
                    "total_ms": elapsed_ms(total_start),
                },
            }
        )
        append_benchmark_log(log_record)
        raise


if __name__ == "__main__":
    main()
