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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model_loader import get_gpu_metrics, get_runtime
from .preprocess import (
    build_top3_scores,
    build_top3_scores_for_index,
    prepare_inputs,
    validate_message,
    validate_messages,
)


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


SERVICE_NAME = os.getenv("SERVICE_NAME", "bert_api")
SERVICE_TYPE = os.getenv("SERVICE_TYPE", "bert")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("APP_ENV", "local")
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "raw" / "bert_classifier" / "baseline_fastapi"
LOG_DIR = Path(os.getenv("BERT_LOG_DIR", str(DEFAULT_LOG_DIR)))
REQUEST_LOG_PATH = Path(
    os.getenv("BERT_REQUEST_LOG_PATH", str(LOG_DIR / "requests_bert.jsonl"))
)

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

runtime = get_runtime()
app = FastAPI(title="BERT Router MVP")


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
        "model_type": "bert_topic_classifier",
        "model_path": str(runtime.model_dir.resolve()),
        "model_version": "unknown",
        "label_mapping_version": "id2label.json",
        "tokenizer_name": getattr(runtime.tokenizer, "name_or_path", "unknown"),
        "tokenizer_version": "unknown",
        "max_length": runtime.max_length,
        "batch_size_default": 1,
        "device_type": runtime.device.type,
        "gpu_name": runtime.gpu_name,
        "gpu_count": runtime.gpu_count,
        "gpu_total_memory_mb": runtime.gpu_total_memory_mb,
        "cuda_available": runtime.cuda_available,
        "cuda_version": runtime.cuda_version,
        "torch_version": runtime.torch_version,
        "python_version": platform.python_version(),
        "log_path": str(REQUEST_LOG_PATH.resolve()),
    }
    logger.info("Startup: %s", json.dumps(startup_record, ensure_ascii=False))
    write_jsonl(REQUEST_LOG_PATH, startup_record)


def run_inference(texts: List[str]):
    inputs, token_lengths, tokenize_ms, to_device_ms = prepare_inputs(
        texts=texts,
        tokenizer=runtime.tokenizer,
        device=runtime.device,
        max_length=runtime.max_length,
        torch_module=runtime.torch,
    )

    if runtime.device.type == "cuda":
        runtime.torch.cuda.reset_peak_memory_stats()
        runtime.torch.cuda.synchronize()

    t_model = time.perf_counter()
    with runtime.torch.no_grad():
        outputs = runtime.model(**inputs)
        probs = runtime.torch.softmax(outputs.logits, dim=1)
    if runtime.device.type == "cuda":
        runtime.torch.cuda.synchronize()
    model_ms = (time.perf_counter() - t_model) * 1000

    t_post = time.perf_counter()
    top_probs, top_ids = runtime.torch.topk(probs, k=min(3, probs.shape[1]), dim=1)
    postprocess_ms = (time.perf_counter() - t_post) * 1000
    return top_probs, top_ids, token_lengths, tokenize_ms, to_device_ms, model_ms, postprocess_ms


def build_batch_predictions(top_probs, top_ids) -> List[PredictBatchResponseItem]:
    predictions = []
    for row_index in range(top_probs.shape[0]):
        pred_id = top_ids[row_index, 0].item()
        pred_conf = top_probs[row_index, 0].item()
        predictions.append(
            PredictBatchResponseItem(
                topic=runtime.id2label[str(pred_id)],
                confidence=round(pred_conf, 4),
                top3=build_top3_scores_for_index(
                    top_probs=top_probs,
                    top_ids=top_ids,
                    row_index=row_index,
                    id2label=runtime.id2label,
                    class_score_cls=ClassScore,
                ),
            )
        )
    return predictions


log_startup()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(runtime.device),
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
        error_record = {
            "event_type": "request",
            "request_id": request_id,
            "timestamp_start": timestamp_start,
            "timestamp_end": utc_now_iso(),
            "service_name": SERVICE_NAME,
            "service_type": SERVICE_TYPE,
            "endpoint": "/predict",
            "worker_id": os.getpid(),
            "text_len_chars": text_len_chars,
            "text_len_tokens": None,
            "tokenize_ms": None,
            "to_device_ms": None,
            "model_ms": None,
            "postprocess_ms": None,
            "total_latency_ms": round((time.perf_counter() - total_started) * 1000, 2),
            "pred_label": None,
            "pred_conf": None,
            "success_flag": False,
            "error_type": "validation_error",
            **get_gpu_metrics(runtime.device),
        }
        write_jsonl(REQUEST_LOG_PATH, error_record)
        raise HTTPException(status_code=400, detail="Message is empty")

    try:
        top_probs, top_ids, token_lengths, tokenize_ms, to_device_ms, model_ms, postprocess_ms = run_inference([text])
        pred_id = top_ids[0, 0].item()
        pred_conf = top_probs[0, 0].item()
        top3 = build_top3_scores(
            top_probs=top_probs,
            top_ids=top_ids,
            id2label=runtime.id2label,
            class_score_cls=ClassScore,
        )
        total_latency_ms = (time.perf_counter() - total_started) * 1000

        request_record = {
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
            "to_device_ms": round_or_none(to_device_ms, 2),
            "model_ms": round_or_none(model_ms, 2),
            "postprocess_ms": round_or_none(postprocess_ms, 2),
            "total_latency_ms": round_or_none(total_latency_ms, 2),
            "pred_label": runtime.id2label[str(pred_id)],
            "pred_conf": round_or_none(pred_conf, 4),
            "success_flag": True,
            "error_type": None,
            **get_gpu_metrics(runtime.device),
        }
        write_jsonl(REQUEST_LOG_PATH, request_record)

        return PredictResponse(
            topic=runtime.id2label[str(pred_id)],
            confidence=round(pred_conf, 4),
            latency_ms=round(total_latency_ms, 2),
            top3=top3,
        )
    except HTTPException:
        raise
    except Exception:
        error_record = {
            "event_type": "request",
            "request_id": request_id,
            "timestamp_start": timestamp_start,
            "timestamp_end": utc_now_iso(),
            "service_name": SERVICE_NAME,
            "service_type": SERVICE_TYPE,
            "endpoint": "/predict",
            "worker_id": os.getpid(),
            "text_len_chars": text_len_chars,
            "text_len_tokens": None,
            "tokenize_ms": None,
            "to_device_ms": None,
            "model_ms": None,
            "postprocess_ms": None,
            "total_latency_ms": round((time.perf_counter() - total_started) * 1000, 2),
            "pred_label": None,
            "pred_conf": None,
            "success_flag": False,
            "error_type": "InternalError",
            **get_gpu_metrics(runtime.device),
        }
        write_jsonl(REQUEST_LOG_PATH, error_record)
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
        top_probs, top_ids, token_lengths, tokenize_ms, to_device_ms, model_ms, postprocess_ms = run_inference(texts)
        predictions = build_batch_predictions(top_probs, top_ids)
        total_latency_ms = (time.perf_counter() - total_started) * 1000

        request_record = {
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
            "to_device_ms": round_or_none(to_device_ms, 2),
            "model_ms": round_or_none(model_ms, 2),
            "postprocess_ms": round_or_none(postprocess_ms, 2),
            "total_latency_ms": round_or_none(total_latency_ms, 2),
            "pred_label": [prediction.topic for prediction in predictions],
            "pred_conf": [prediction.confidence for prediction in predictions],
            "success_flag": True,
            "error_type": None,
            **get_gpu_metrics(runtime.device),
        }
        write_jsonl(REQUEST_LOG_PATH, request_record)

        return PredictBatchResponse(
            predictions=predictions,
            batch_size=batch_size,
            latency_ms=round(total_latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception:
        error_record = {
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
            "text_len_tokens": None,
            "tokenize_ms": None,
            "to_device_ms": None,
            "model_ms": None,
            "postprocess_ms": None,
            "total_latency_ms": round((time.perf_counter() - total_started) * 1000, 2),
            "pred_label": None,
            "pred_conf": None,
            "success_flag": False,
            "error_type": "InternalError",
            **get_gpu_metrics(runtime.device),
        }
        write_jsonl(REQUEST_LOG_PATH, error_record)
        logger.exception("Batch prediction failed for request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Internal prediction error")
