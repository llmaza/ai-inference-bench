#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL_DEFAULT = "intfloat/multilingual-e5-large"
VECTOR_SIZE_DEFAULT = 1024
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "


def resolve_device() -> str:
    requested = os.environ.get("LOCAL_EMBED_DEVICE", "").strip().lower()
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LocalEmbeddingBackend:
    model_name: str
    device: str
    max_length: int

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
            masked = token_embeddings * attention_mask.float()
            pooled = masked.sum(dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            normalized = F.normalize(pooled, p=2, dim=1)
        return normalized.detach().cpu().numpy()

    def encode_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            arr = self._encode_batch(batch)
            vectors.extend(arr.astype(float).tolist())
        return vectors


@lru_cache(maxsize=4)
def get_local_backend(
    model_name: str = EMBEDDING_MODEL_DEFAULT,
    device: str | None = None,
    max_length: int = 512,
) -> LocalEmbeddingBackend:
    return LocalEmbeddingBackend(
        model_name=model_name,
        device=device or resolve_device(),
        max_length=max_length,
    )


def encode_query_e5_local(query: str, model_name: str) -> list[float]:
    backend = get_local_backend(model_name=model_name)
    return backend.encode_texts([E5_QUERY_PREFIX + query], batch_size=1)[0]


def encode_passages_e5_local(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> list[list[float]]:
    backend = get_local_backend(model_name=model_name)
    prefixed = [E5_PASSAGE_PREFIX + text for text in texts]
    return backend.encode_texts(prefixed, batch_size=batch_size)
