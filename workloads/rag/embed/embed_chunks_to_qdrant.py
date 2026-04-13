#!/usr/bin/env python3
"""
Локальные эмбеддинги intfloat/multilingual-e5-large
и загрузка чанков в Qdrant.

Для E5 при индексации к тексту добавляется префикс «passage: » (для поиска — «query: »).

Переменные в .env:
  QDRANT_URL            — по умолчанию http://localhost:6333
  QDRANT_API_KEY        — для Qdrant Cloud (если нужен)
  EMBEDDING_MODEL       — локальная HF-модель эмбеддингов
  EMBEDDING_VECTOR_SIZE — размерность вектора
  LOCAL_EMBED_DEVICE    — cpu / cuda
  LOCAL_EMBED_BATCH_SIZE — локальный batch size

Qdrant локально:
  docker run -p 6333:6333 qdrant/qdrant

Запуск:
  .venv/bin/python embed_chunks_to_qdrant.py
  .venv/bin/python embed_chunks_to_qdrant.py --recreate --collection tk_ru_e5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.append(str(RAG_ROOT))

from embed.local_backend import (  # noqa: E402
    EMBEDDING_MODEL_DEFAULT,
    VECTOR_SIZE_DEFAULT,
    encode_passages_e5_local,
    get_local_backend,
)

DEFAULT_CHUNKS = RAG_ROOT / "direct" / "assets" / "Трудовой_Кодекс_chunks.jsonl"


def load_chunks(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def payload_for_point(chunk: dict) -> dict:
    md = chunk["metadata"]
    p: dict = {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
    }
    for k, v in md.items():
        p[k] = v
    return p


def ensure_collection(
    client: QdrantClient,
    name: str,
    *,
    vector_size: int,
    recreate: bool,
) -> None:
    exists = any(c.name == name for c in client.get_collections().collections)
    if recreate and exists:
        client.delete_collection(collection_name=name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=vector_size,
                distance=qm.Distance.COSINE,
            ),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Local E5 embeddings → Qdrant")
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--collection", type=str, default="labor_code_tk_e5")
    ap.add_argument(
        "--recreate",
        action="store_true",
        help="Удалить коллекцию и создать заново",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Сколько чанков залить в Qdrant за один upsert.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Пауза (сек) между шагами загрузки.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Один чанк: эмбеддинг + размер вектора, без Qdrant",
    )
    args = ap.parse_args()

    _env_file = RAG_ROOT / ".env"
    if load_dotenv is not None and _env_file.is_file():
        load_dotenv(_env_file)

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    model_name = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT).strip()
    vector_size = int(os.environ.get("EMBEDDING_VECTOR_SIZE", str(VECTOR_SIZE_DEFAULT)))
    local_batch_size = int(os.environ.get("LOCAL_EMBED_BATCH_SIZE", str(args.batch_size)))

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise SystemExit(f"Нет чанков в {args.chunks}")

    client: QdrantClient | None = None
    if not args.dry_run:
        client = QdrantClient(url=url, api_key=api_key)
        ensure_collection(
            client,
            args.collection,
            vector_size=vector_size,
            recreate=args.recreate,
        )
        print(f"Qdrant: {url}, коллекция: {args.collection}")

    backend = get_local_backend(model_name=model_name)
    print(f"Local embeddings, модель: {model_name}")
    print(f"Устройство: {backend.device}")
    print(f"Ожидаемая размерность вектора: {vector_size}")
    print(f"Размер локального батча эмбеддингов: {local_batch_size}")

    if args.dry_run:
        c = chunks[0]
        embed_start = time.perf_counter()
        vec = encode_passages_e5_local([c["text"]], model_name=model_name, batch_size=1)[0]
        embed_ms = (time.perf_counter() - embed_start) * 1000
        if len(vec) != vector_size:
            raise SystemExit(
                f"Размер вектора {len(vec)} ≠ {vector_size}. Проверьте модель и EMBEDDING_VECTOR_SIZE."
            )
        print(
            f"dry-run: chunk_id={c['chunk_id']}, dim={len(vec)}, "
            f"первые 3 значения: {vec[:3]}, embed_ms={embed_ms:.2f}"
        )
        print("dry-run завершён (Qdrant не трогали).")
        return

    assert client is not None
    total_uploaded = 0
    batch_chunks: list[dict] = []
    total_embed_ms = 0.0

    for global_i, c in enumerate(chunks):
        batch_chunks.append(c)

        flush = len(batch_chunks) >= args.batch_size or global_i == len(chunks) - 1
        if flush:
            texts = [bc["text"] for bc in batch_chunks]
            embed_start = time.perf_counter()
            batch_vecs = encode_passages_e5_local(
                texts,
                model_name=model_name,
                batch_size=local_batch_size,
            )
            batch_embed_ms = (time.perf_counter() - embed_start) * 1000
            total_embed_ms += batch_embed_ms
            for vec in batch_vecs:
                if len(vec) != vector_size:
                    raise SystemExit(
                        f"Размер вектора {len(vec)} ≠ {vector_size}. Проверьте модель и EMBEDDING_VECTOR_SIZE."
                    )
            points = [
                qm.PointStruct(
                    id=int(bc["chunk_id"]),
                    vector=batch_vecs[i],
                    payload=payload_for_point(bc),
                )
                for i, bc in enumerate(batch_chunks)
            ]
            client.upsert(collection_name=args.collection, points=points)
            total_uploaded += len(points)
            avg_embed_ms = batch_embed_ms / max(len(batch_chunks), 1)
            print(
                f"Загружено {total_uploaded} / {len(chunks)} "
                f"(batch_embed_ms={batch_embed_ms:.2f}, avg_embed_ms={avg_embed_ms:.2f})"
            )
            batch_chunks = []

        if args.sleep > 0 and global_i < len(chunks) - 1:
            time.sleep(args.sleep)

    info = client.get_collection(args.collection)
    print(
        f"Готово. Точек в коллекции (по данным кластера): "
        f"{getattr(info, 'points_count', '?')}"
    )
    print(
        f"Итог по эмбеддингам: total_embed_ms={total_embed_ms:.2f}, "
        f"avg_embed_ms={total_embed_ms / max(len(chunks), 1):.2f}"
    )


if __name__ == "__main__":
    main()
