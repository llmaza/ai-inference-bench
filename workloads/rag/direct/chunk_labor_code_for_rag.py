#!/usr/bin/env python3
"""
Чанкинг структурированного ТК для RAG.

Правило: если пункт (элемент points) ≤ max_tokens — один чанк; иначе — скользящие
окна с перекрытием. Счёт токенов: tiktoken cl100k_base (сопоставимо с OpenAI embeddings).

Рекомендуемые эмбеддинги (русский + юридический текст):
  • text-embedding-3-large — сильнее по качеству, мультиязычность, до 8191 токена на вход;
  • text-embedding-3-small — дешевле, часто достаточно для справочного поиска по НПА;
  • Локально / open-source: BAAI/bge-m3 (мультиязычный, длинный контекст),
    intfloat/multilingual-e5-large — хороший баланс для RU;
    для узкоспециализированного юр. RU иногда дообучают e5/bge на корпусе законов.

Запуск:
  .venv/bin/python chunk_labor_code_for_rag.py
  .venv/bin/python chunk_labor_code_for_rag.py --max-tokens 512 --overlap 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken

DEFAULT_INPUT = Path(__file__).resolve().parent / "assets" / "Трудовой_Кодекс_структурированный.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "assets" / "Трудовой_Кодекс_chunks.jsonl"


def split_tokens_with_overlap(
    tokens: list[int],
    max_tokens: int,
    overlap: int,
) -> list[list[int]]:
    if len(tokens) <= max_tokens:
        return [tokens]
    if overlap >= max_tokens:
        raise ValueError("overlap must be less than max_tokens")
    step = max_tokens - overlap
    chunks: list[list[int]] = []
    i = 0
    n = len(tokens)
    while i < n:
        piece = tokens[i : i + max_tokens]
        chunks.append(piece)
        if i + max_tokens >= n:
            break
        i += step
    return chunks


def point_breadcrumb(article: dict, p: dict) -> str:
    parts = [
        article["document"],
        f"ст. {article['article_number']}",
        article.get("article_title", "")[:120],
    ]
    kind = p.get("kind", "")
    label = p.get("label", "")
    if kind == "punkt" and label:
        parts.append(f"п. {label}.")
    elif kind == "podpunkt" and label:
        parts.append(f"подп. {label})")
    elif kind == "intro":
        parts.append("вводная часть")
    elif kind == "text":
        parts.append("текст статьи")
    return " | ".join(x for x in parts if x)


def chunk_corpus(
    articles: list[dict],
    enc: tiktoken.Encoding,
    max_tokens: int,
    overlap: int,
    prepend_breadcrumb_to_text: bool,
) -> list[dict]:
    out: list[dict] = []
    chunk_id = 0
    for art in articles:
        points = art.get("points") or []
        if not points and art.get("text"):
            points = [{"kind": "text", "label": "", "text": art["text"]}]

        for pi, p in enumerate(points):
            raw = (p.get("text") or "").strip()
            if not raw:
                continue
            crumb = point_breadcrumb(art, p)
            tokens = enc.encode(raw)
            token_lists = split_tokens_with_overlap(tokens, max_tokens, overlap)
            total_sub = len(token_lists)

            for si, toks in enumerate(token_lists):
                body = enc.decode(toks)
                if prepend_breadcrumb_to_text:
                    embed_text = f"{crumb}\n\n{body}"
                else:
                    embed_text = body

                rec = {
                    "chunk_id": chunk_id,
                    "text": embed_text,
                    "token_count": len(toks),
                    "metadata": {
                        "document": art["document"],
                        "part": art.get("part", ""),
                        "section": art.get("section", ""),
                        "chapter": art.get("chapter", ""),
                        "article_number": art["article_number"],
                        "article_title": art.get("article_title", ""),
                        "point_index": pi,
                        "point_kind": p.get("kind", ""),
                        "point_label": p.get("label", ""),
                        "breadcrumb": crumb,
                        "chunk_in_point": si,
                        "chunks_in_point": total_sub,
                        "tokenizer": "cl100k_base",
                        "max_tokens": max_tokens,
                        "overlap_tokens": overlap if total_sub > 1 else 0,
                    },
                }
                out.append(rec)
                chunk_id += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Чанкинг ТК для RAG (512 ток. + overlap).")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Перекрытие в токенах при разбиении длинного пункта (~12%% при 512).",
    )
    ap.add_argument(
        "--no-breadcrumb-in-text",
        action="store_true",
        help="Не добавлять breadcrumb в поле text (останется только в metadata).",
    )
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    enc = tiktoken.get_encoding("cl100k_base")

    chunks = chunk_corpus(
        data,
        enc,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        prepend_breadcrumb_to_text=not args.no_breadcrumb_in_text,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    multi = sum(1 for c in chunks if c["metadata"]["chunks_in_point"] > 1)
    print(f"Чанков: {len(chunks)} (из них с разбиением пункта: {multi})")
    print(f"Выход: {args.output.resolve()}")
    print(
        "Эмбеддинги: text-embedding-3-large или 3-small (OpenAI); "
        "локально — bge-m3 / multilingual-e5-large."
    )


if __name__ == "__main__":
    main()
