from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .backends.local_fastapi import get_runtime
from .config import REPO_ROOT
from .schemas import GenerateRequest, GenerateResponse


SERVICE_NAME = os.getenv("SERVICE_NAME", "small_llm_api")
SERVICE_TYPE = os.getenv("SERVICE_TYPE", "small_llm")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("APP_ENV", "local")
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")
SYSTEM_PROMPT = os.getenv(
    "SMALL_LLM_SYSTEM_PROMPT",
    "You are a concise, helpful assistant. Answer directly and avoid repetition.",
)
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "raw" / "small_llm" / "baseline_fastapi"
LOG_DIR = Path(os.getenv("SMALL_LLM_LOG_DIR", str(DEFAULT_LOG_DIR)))
REQUEST_LOG_PATH = Path(
    os.getenv(
        "SMALL_LLM_REQUEST_LOG_PATH",
        str(LOG_DIR / "requests_small_llm.jsonl"),
    )
)

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

runtime = get_runtime()
app = FastAPI(title="Small LLM Baseline API")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        "model_key": runtime.model_key,
        "model_name": runtime.model_name,
        "display_name": runtime.display_name,
        "device": runtime.device,
        "backend_key": runtime.backend_key,
        "loader_key": runtime.loader_key,
        "max_input_tokens": runtime.max_input_tokens,
        "max_new_tokens": runtime.max_new_tokens,
        "gpu_name": runtime.gpu_name,
        "gpu_total_memory_mb": runtime.gpu_total_memory_mb,
        "torch_version": runtime.torch_version,
        "python_version": runtime.python_version,
        "log_path": str(REQUEST_LOG_PATH.resolve()),
    }
    logger.info("Startup: %s", json.dumps(startup_record, ensure_ascii=False))
    write_jsonl(REQUEST_LOG_PATH, startup_record)


log_startup()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service_name": SERVICE_NAME,
        "service_type": SERVICE_TYPE,
        "model_key": runtime.model_key,
        "model_name": runtime.model_name,
        "device": runtime.device,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    request_id = str(uuid.uuid4())
    timestamp_start = utc_now_iso()
    started = time.perf_counter()
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        result = runtime.backend.generate_with_stats(
            messages=[
                {"role": "system", "content": runtime.system_prompt},
                {"role": "user", "content": message},
            ],
            max_new_tokens=runtime.max_new_tokens,
            temperature=runtime.temperature,
        )
    except Exception as exc:
        error_record = {
            "event_type": "request",
            "request_id": request_id,
            "timestamp_start": timestamp_start,
            "timestamp_end": utc_now_iso(),
            "endpoint": "/generate",
            "message_len_chars": len(message),
            "success_flag": False,
            "error_type": type(exc).__name__,
            "error_text": str(exc),
            "total_latency_ms": round((time.perf_counter() - started) * 1000, 4),
            "model_key": runtime.model_key,
            "model_name": runtime.model_name,
            "device": runtime.device,
        }
        write_jsonl(REQUEST_LOG_PATH, error_record)
        raise

    latency_ms = (time.perf_counter() - started) * 1000
    request_record = {
        "event_type": "request",
        "request_id": request_id,
        "timestamp_start": timestamp_start,
        "timestamp_end": utc_now_iso(),
        "endpoint": "/generate",
        "message_len_chars": len(message),
        "message_preview": message[:160],
        "success_flag": True,
        "error_type": None,
        "error_text": None,
        "model_key": runtime.model_key,
        "model_name": runtime.model_name,
        "device": runtime.device,
        "latency_ms": round(latency_ms, 4),
        **result.to_dict(),
    }
    write_jsonl(REQUEST_LOG_PATH, request_record)
    return GenerateResponse(
        text=result.text,
        latency_ms=round(latency_ms, 4),
        generation_ms=round(result.generation_ms, 4),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        tokens_per_sec=round(result.tokens_per_sec, 4)
        if result.tokens_per_sec is not None
        else None,
        ttft_ms=result.ttft_ms,
        peak_gpu_memory_mb=round(result.peak_gpu_memory_mb, 4)
        if result.peak_gpu_memory_mb is not None
        else None,
        model_key=runtime.model_key,
        model_name=runtime.model_name,
        device=runtime.device,
    )
