import json
import logging
import os
import platform
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model_loader import get_runtime
from .preprocess import validate_message, validate_messages


class PredictRequest(BaseModel):
    message: str


class PredictBatchRequest(BaseModel):
    messages: List[str]


class ClassScore(BaseModel):
    topic: str
    confidence: float


class PredictResponse(BaseModel):
    topic: str
    confidence: float
    latency_ms: float
    top3: List[ClassScore]


class PredictBatchResponseItem(BaseModel):
    topic: str
    confidence: float
    top3: List[ClassScore]


class PredictBatchResponse(BaseModel):
    predictions: List[PredictBatchResponseItem]
    batch_size: int
    latency_ms: float


SERVICE_NAME = os.getenv("SERVICE_NAME", "bert_triton_api")
SERVICE_TYPE = os.getenv("SERVICE_TYPE", "bert_triton")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("APP_ENV", "local")
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "raw" / "bert_classifier" / "triton_fastapi"
LOG_DIR = Path(os.getenv("BERT_LOG_DIR", str(DEFAULT_LOG_DIR)))
REQUEST_LOG_PATH = Path(
    os.getenv("BERT_REQUEST_LOG_PATH", str(LOG_DIR / "requests_bert_triton.jsonl"))
)

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

runtime = get_runtime()
app = FastAPI(title="BERT Triton Router MVP")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_startup() -> None:
    startup_record = {
        "event_type": "startup",
        "timestamp": utc_now_iso(),
        "service_name": SERVICE_NAME,
        "service_type": SERVICE_TYPE,
        "service_version": SERVICE_VERSION,
        "git_commit": GIT_COMMIT,
        "environment": ENVIRONMENT,
        "host_name": socket.gethostname(),
        "pid": os.getpid(),
        "model_type": "bert_topic_classifier_triton",
        "model_path": str(runtime.model_dir.resolve()),
        "triton_url": runtime.triton_url,
        "model_version": "unknown",
        "label_mapping_version": "id2label.json",
        "tokenizer_name": str(runtime.model_dir.resolve()),
        "tokenizer_version": "unknown",
        "max_length": runtime.max_length,
        "batch_size_default": 1,
        "python_version": platform.python_version(),
        "log_path": str(REQUEST_LOG_PATH.resolve()),
    }
    logger.info("Startup: %s", json.dumps(startup_record, ensure_ascii=False))
    write_jsonl(REQUEST_LOG_PATH, startup_record)


def build_top3_scores_for_row(probs: np.ndarray, row_index: int) -> List[ClassScore]:
    top_ids = np.argsort(-probs[row_index])[: min(3, probs.shape[1])]
    return [
        ClassScore(
            topic=runtime.id2label[str(int(class_id))],
            confidence=round(float(probs[row_index][class_id]), 4),
        )
        for class_id in top_ids
    ]


def run_inference(texts: List[str]):
    t0 = time.perf_counter()
    encoded = runtime.client.encode(texts)
    token_lengths = [int(mask.sum()) for mask in encoded["attention_mask"]]
    tokenize_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    logits = runtime.client.infer_logits(texts)
    model_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    probs = runtime.client.softmax(logits)
    postprocess_ms = (time.perf_counter() - t2) * 1000
    return probs, token_lengths, tokenize_ms, model_ms, postprocess_ms


log_startup()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "triton_url": runtime.triton_url,
        "service_name": SERVICE_NAME,
        "service_type": SERVICE_TYPE,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    request_id = str(uuid.uuid4())
    timestamp_start = utc_now_iso()
    total_started = time.perf_counter()

    text = validate_message(request.message)
    text_len_chars = len(text)
    if not text:
        raise HTTPException(status_code=400, detail="Message is empty")

    try:
        probs, token_lengths, tokenize_ms, model_ms, postprocess_ms = run_inference([text])
        pred_id = int(np.argmax(probs, axis=1)[0])
        pred_conf = float(probs[0][pred_id])
        top3 = build_top3_scores_for_row(probs, 0)
        total_latency_ms = (time.perf_counter() - total_started) * 1000

        write_jsonl(
            REQUEST_LOG_PATH,
            {
                "event_type": "request",
                "request_id": request_id,
                "timestamp_start": timestamp_start,
                "timestamp_end": utc_now_iso(),
                "service_name": SERVICE_NAME,
                "service_type": SERVICE_TYPE,
                "endpoint": "/predict",
                "worker_id": os.getpid(),
                "text_len_chars": text_len_chars,
                "text_len_tokens": token_lengths[0],
                "tokenize_ms": round_or_none(tokenize_ms, 2),
                "model_ms": round_or_none(model_ms, 2),
                "postprocess_ms": round_or_none(postprocess_ms, 2),
                "total_latency_ms": round_or_none(total_latency_ms, 2),
                "pred_label": runtime.id2label[str(pred_id)],
                "pred_conf": round_or_none(pred_conf, 4),
                "success_flag": True,
                "error_type": None,
                "triton_url": runtime.triton_url,
            },
        )

        return PredictResponse(
            topic=runtime.id2label[str(pred_id)],
            confidence=round(pred_conf, 4),
            latency_ms=round(total_latency_ms, 2),
            top3=top3,
        )
    except Exception:
        logger.exception("Prediction failed for request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest):
    request_id = str(uuid.uuid4())
    timestamp_start = utc_now_iso()
    total_started = time.perf_counter()

    texts = validate_messages(request.messages)
    batch_size = len(texts)
    if batch_size == 0:
        raise HTTPException(status_code=400, detail="Messages list is empty")
    if any(not text for text in texts):
        raise HTTPException(status_code=400, detail="Messages list contains empty items")

    try:
        probs, token_lengths, tokenize_ms, model_ms, postprocess_ms = run_inference(texts)
        predictions = []
        for row_index in range(probs.shape[0]):
            pred_id = int(np.argmax(probs[row_index]))
            pred_conf = float(probs[row_index][pred_id])
            predictions.append(
                PredictBatchResponseItem(
                    topic=runtime.id2label[str(pred_id)],
                    confidence=round(pred_conf, 4),
                    top3=build_top3_scores_for_row(probs, row_index),
                )
            )
        total_latency_ms = (time.perf_counter() - total_started) * 1000

        write_jsonl(
            REQUEST_LOG_PATH,
            {
                "event_type": "request_batch",
                "request_id": request_id,
                "timestamp_start": timestamp_start,
                "timestamp_end": utc_now_iso(),
                "service_name": SERVICE_NAME,
                "service_type": SERVICE_TYPE,
                "endpoint": "/predict_batch",
                "worker_id": os.getpid(),
                "batch_size": batch_size,
                "text_len_chars": sum(len(text) for text in texts),
                "text_len_tokens": token_lengths,
                "tokenize_ms": round_or_none(tokenize_ms, 2),
                "model_ms": round_or_none(model_ms, 2),
                "postprocess_ms": round_or_none(postprocess_ms, 2),
                "total_latency_ms": round_or_none(total_latency_ms, 2),
                "pred_label": [prediction.topic for prediction in predictions],
                "pred_conf": [prediction.confidence for prediction in predictions],
                "success_flag": True,
                "error_type": None,
                "triton_url": runtime.triton_url,
            },
        )

        return PredictBatchResponse(
            predictions=predictions,
            batch_size=batch_size,
            latency_ms=round(total_latency_ms, 2),
        )
    except Exception:
        logger.exception("Batch prediction failed for request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Internal prediction error")
