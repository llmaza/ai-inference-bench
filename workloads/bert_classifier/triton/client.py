import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "workloads" / "bert_classifier" / "artifacts" / "model"
DEFAULT_TRITON_URL = "http://127.0.0.1:8000/v2/models/bert_classifier/infer"


class TritonBertClient:
    def __init__(
        self,
        triton_url: str = DEFAULT_TRITON_URL,
        model_dir: Path = DEFAULT_MODEL_DIR,
        max_length: int = 256,
        timeout_sec: float = 30.0,
    ):
        self.triton_url = triton_url
        self.model_dir = Path(model_dir).resolve()
        self.max_length = max_length
        self.timeout_sec = timeout_sec
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.id2label = self._load_id2label(self.model_dir)

    @staticmethod
    def _load_id2label(model_dir: Path) -> Dict[str, str]:
        with (model_dir / "id2label.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def encode(self, messages: List[str]) -> Dict[str, np.ndarray]:
        encoded = self.tokenizer(
            messages,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "token_type_ids": encoded["token_type_ids"].astype(np.int64),
        }

    def build_payload(self, messages: List[str]) -> dict:
        encoded = self.encode(messages)
        return {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": list(encoded["input_ids"].shape),
                    "datatype": "INT64",
                    "data": encoded["input_ids"].tolist(),
                },
                {
                    "name": "attention_mask",
                    "shape": list(encoded["attention_mask"].shape),
                    "datatype": "INT64",
                    "data": encoded["attention_mask"].tolist(),
                },
                {
                    "name": "token_type_ids",
                    "shape": list(encoded["token_type_ids"].shape),
                    "datatype": "INT64",
                    "data": encoded["token_type_ids"].tolist(),
                },
            ],
            "outputs": [{"name": "logits"}],
        }

    def infer_logits(self, messages: List[str]) -> np.ndarray:
        payload = self.build_payload(messages)
        response = requests.post(self.triton_url, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        body = response.json()
        output = body["outputs"][0]
        shape = output["shape"]
        logits = np.array(output["data"], dtype=np.float32).reshape(shape)
        return logits

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(shifted)
        return probs / np.sum(probs, axis=1, keepdims=True)

    def predict(self, messages: List[str]) -> List[Dict[str, Optional[float]]]:
        logits = self.infer_logits(messages)
        probs = self.softmax(logits)
        predictions = []
        for row_index in range(probs.shape[0]):
            pred_id = int(np.argmax(probs[row_index]))
            predictions.append(
                {
                    "class_id": pred_id,
                    "topic": self.id2label.get(str(pred_id), str(pred_id)),
                    "confidence": float(probs[row_index][pred_id]),
                }
            )
        return predictions
