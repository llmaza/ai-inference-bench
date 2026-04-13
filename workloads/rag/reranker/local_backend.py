#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RERANKER_DEFAULT = "BAAI/bge-reranker-v2-m3"


def resolve_device() -> str:
    requested = os.environ.get("LOCAL_RERANK_DEVICE", "").strip().lower()
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LocalRerankerBackend:
    model_name: str
    device: str
    max_length: int

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def score_pairs(
        self,
        query: str,
        passages: list[str],
        batch_size: int,
    ) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            encoded = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    batch_scores = logits[:, 0]
                elif logits.ndim == 1:
                    batch_scores = logits
                else:
                    batch_scores = logits[:, 0]
            scores.extend(batch_scores.detach().cpu().tolist())
        return [float(s) for s in scores]


@lru_cache(maxsize=4)
def get_local_reranker(
    model_name: str = RERANKER_DEFAULT,
    device: str | None = None,
    max_length: int = 1024,
) -> LocalRerankerBackend:
    return LocalRerankerBackend(
        model_name=model_name,
        device=device or resolve_device(),
        max_length=max_length,
    )
