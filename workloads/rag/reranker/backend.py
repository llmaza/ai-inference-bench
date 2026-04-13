#!/usr/bin/env python3
from __future__ import annotations

import os

from reranker.local_backend import RERANKER_DEFAULT, get_local_reranker
from reranker.onnx_backend import get_onnx_reranker

RERANK_BACKEND_DEFAULT = "torch"


def resolve_reranker_backend(requested: str | None = None) -> str:
    value = (requested or os.environ.get("RERANK_BACKEND") or RERANK_BACKEND_DEFAULT).strip().lower()
    if value not in {"torch", "onnx"}:
        raise ValueError(f"Unsupported reranker backend: {value}")
    return value


def score_pairs(
    query: str,
    passages: list[str],
    model_name: str = RERANKER_DEFAULT,
    batch_size: int = 12,
    backend_name: str | None = None,
) -> list[float]:
    backend = resolve_reranker_backend(backend_name)
    if backend == "onnx":
        return get_onnx_reranker(model_name=model_name).score_pairs(
            query,
            passages,
            batch_size=batch_size,
        )
    return get_local_reranker(model_name=model_name).score_pairs(
        query,
        passages,
        batch_size=batch_size,
    )
