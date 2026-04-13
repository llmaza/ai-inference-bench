#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RERANKER_DEFAULT = "BAAI/bge-reranker-v2-m3"
RERANKER_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = RERANKER_ROOT / "artifacts"


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def resolve_provider() -> str:
    requested = os.environ.get("LOCAL_RERANK_ONNX_PROVIDER", "").strip()
    available = ort.get_available_providers()
    if requested:
        if requested not in available:
            raise RuntimeError(f"Requested ONNX provider {requested!r} is not available: {available}")
        return requested
    if "CUDAExecutionProvider" in available:
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def export_reranker_to_onnx(model_name: str, max_length: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    dummy = tokenizer(
        ["query", "query"],
        ["passage", "passage"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_names = [name for name in tokenizer.model_input_names if name in dummy]
    dynamic_axes = {"logits": {0: "batch"}}
    for name in input_names:
        dynamic_axes[name] = {0: "batch", 1: "sequence"}

    class ForwardWrapper(torch.nn.Module):
        def __init__(self, base_model: AutoModelForSequenceClassification, names: list[str]) -> None:
            super().__init__()
            self.base_model = base_model
            self.names = names

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            kwargs = {name: tensor for name, tensor in zip(self.names, args)}
            return self.base_model(**kwargs).logits

    wrapper = ForwardWrapper(model, input_names)
    inputs = tuple(dummy[name] for name in input_names)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )


@dataclass
class OnnxRerankerBackend:
    model_name: str
    provider: str
    max_length: int

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_dir = ARTIFACTS_ROOT / sanitize_model_name(self.model_name)
        self.onnx_path = model_dir / "model.onnx"
        if not self.onnx_path.exists():
            export_reranker_to_onnx(self.model_name, self.max_length, self.onnx_path)
        session_providers = [self.provider]
        if self.provider != "CPUExecutionProvider":
            session_providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(self.onnx_path), providers=session_providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]

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
                return_tensors="np",
            )
            feeds = {}
            for name in self.input_names:
                arr = encoded.get(name)
                if arr is None:
                    continue
                if arr.dtype != np.int64:
                    arr = arr.astype(np.int64)
                feeds[name] = arr
            logits = self.session.run(["logits"], feeds)[0]
            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits[:, 0]
            elif logits.ndim == 1:
                batch_scores = logits
            else:
                batch_scores = logits[:, 0]
            scores.extend(float(x) for x in batch_scores.tolist())
        return scores


@lru_cache(maxsize=4)
def get_onnx_reranker(
    model_name: str = RERANKER_DEFAULT,
    provider: str | None = None,
    max_length: int = 1024,
) -> OnnxRerankerBackend:
    return OnnxRerankerBackend(
        model_name=model_name,
        provider=provider or resolve_provider(),
        max_length=max_length,
    )
