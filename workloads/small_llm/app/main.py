from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from .backends.local_fastapi import get_runtime
from .config import REPO_ROOT
from .logging_utils import append_jsonl_record, request_record, utc_now_iso
from .schemas import GenerateBatchRequest, GenerateRequest, GenerateResponse


SERVICE_NAME = os.getenv("SERVICE_NAME", "small_llm_api")
SERVICE_TYPE = os.getenv("SERVICE_TYPE", "small_llm")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("APP_ENV", "local")
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")
SYSTEM_PROMPT = os.getenv(
    "SMALL_LLM_SYSTEM_PROMPT",
    "You are a concise, helpful assistant. Answer directly and avoid repetition.",
)
DEFAULT_STAGE_A_LOG_DIR = REPO_ROOT / "workloads" / "small_llm" / "results" / "stage_a_baseline" / "requests"
LOG_DIR = Path(os.getenv("SMALL_LLM_LOG_DIR", str(DEFAULT_STAGE_A_LOG_DIR)))
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


def _optional_int_header(request: Request, header_name: str) -> int | None:
    value = request.headers.get(header_name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None

def log_startup() -> None:
    startup_record = {
        "benchmark_schema_version": "small_llm.v1",
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
    append_jsonl_record(REQUEST_LOG_PATH, startup_record)


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
def generate(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    request_id = str(uuid.uuid4())
    timestamp = utc_now_iso()
    started = time.perf_counter()
    message = request.message.strip()
    run_id = http_request.headers.get("x-run-id") or None
    prompt_name = http_request.headers.get("x-prompt-name") or None
    prompt_file = http_request.headers.get("x-prompt-file") or None
    request_index = _optional_int_header(http_request, "x-request-index")
    repeat_index = _optional_int_header(http_request, "x-repeat-index")
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        result = runtime.backend.generate_with_stats(
            messages=[
                {"role": "system", "content": runtime.system_prompt},
                {"role": "user", "content": message},
            ],
            max_new_tokens=request.max_new_tokens or runtime.max_new_tokens,
            temperature=runtime.temperature,
        )
    except Exception as exc:
        error_record = request_record(
            request_id=request_id,
            run_id=run_id,
            timestamp=timestamp,
            stage="baseline_fastapi",
            backend="baseline_fastapi",
            model_key=runtime.model_key,
            model_name=runtime.model_name,
            prompt_name=prompt_name,
            prompt_file=prompt_file,
            repeat_index=repeat_index,
            request_index=request_index,
            max_new_tokens=runtime.max_new_tokens,
            temperature=runtime.temperature,
            do_sample=False,
            wall_latency_ms=round((time.perf_counter() - started) * 1000, 4),
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            finish_reason=None,
            device=runtime.device,
            dtype=runtime.dtype,
            generated_text=None,
            event_type="request",
            timestamp_start=timestamp,
            timestamp_end=utc_now_iso(),
            endpoint="/generate",
            message_len_chars=len(message),
            success_flag=False,
            error_text=str(exc),
            total_latency_ms=round((time.perf_counter() - started) * 1000, 4),
        )
        append_jsonl_record(REQUEST_LOG_PATH, error_record)
        raise

    latency_ms = (time.perf_counter() - started) * 1000
    service_record = request_record(
        request_id=request_id,
        run_id=run_id,
        timestamp=timestamp,
        stage="baseline_fastapi",
        backend="baseline_fastapi",
        model_key=runtime.model_key,
        model_name=runtime.model_name,
        prompt_name=prompt_name,
        prompt_file=prompt_file,
        repeat_index=repeat_index,
        request_index=request_index,
        input_tokens=result.input_tokens,
        generated_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        max_new_tokens=runtime.max_new_tokens,
        temperature=runtime.temperature,
        do_sample=False,
        wall_latency_ms=round(latency_ms, 4),
        generation_ms=round(result.generation_ms, 4),
        ttft_ms=result.ttft_ms,
        tokens_per_sec=round(result.tokens_per_sec, 4)
        if result.tokens_per_sec is not None
        else None,
        success=True,
        error_type=None,
        error_message=None,
        finish_reason=None,
        device=runtime.device,
        dtype=runtime.dtype,
        gpu_memory_mb=None,
        peak_gpu_memory_mb=round(result.peak_gpu_memory_mb, 4)
        if result.peak_gpu_memory_mb is not None
        else None,
        generated_text=result.text,
        event_type="request",
        timestamp_start=timestamp,
        timestamp_end=utc_now_iso(),
        endpoint="/generate",
        message_len_chars=len(message),
        message_preview=message[:160],
        success_flag=True,
        error_text=None,
        latency_ms=round(latency_ms, 4),
    )
    append_jsonl_record(REQUEST_LOG_PATH, service_record)
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



@app.post("/generate_batch")
def generate_batch(request: GenerateBatchRequest, http_request: Request) -> dict:
    request_id = str(uuid.uuid4())
    timestamp = utc_now_iso()
    started = time.perf_counter()
    run_id = http_request.headers.get("x-run-id") or None
    prompt_file = http_request.headers.get("x-prompt-file") or None
    results = runtime.backend.generate_batch_with_stats(
        messages_batch=[
            [
                {"role": "system", "content": runtime.system_prompt},
                {"role": "user", "content": item.message.strip()},
            ]
            for item in request.items
        ],
        max_new_tokens=request.items[0].max_new_tokens or runtime.max_new_tokens,
        temperature=runtime.temperature,
    )
    responses = []
    for idx, (item, result) in enumerate(zip(request.items, results)):
        item_request_id = f"{request_id}:{idx}"
        latency_ms = (time.perf_counter() - started) * 1000
        service_record = request_record(
            request_id=item_request_id,
            run_id=run_id,
            timestamp=timestamp,
            stage="baseline_fastapi",
            backend="baseline_fastapi",
            model_key=runtime.model_key,
            model_name=runtime.model_name,
            prompt_name=item.prompt_name,
            prompt_file=item.prompt_file or prompt_file,
            repeat_index=item.repeat_index,
            request_index=item.request_index,
            input_tokens=result.input_tokens,
            generated_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            max_new_tokens=item.max_new_tokens or runtime.max_new_tokens,
            temperature=runtime.temperature,
            do_sample=False,
            wall_latency_ms=round(latency_ms, 4),
            generation_ms=round(result.generation_ms, 4),
            ttft_ms=result.ttft_ms,
            tokens_per_sec=round(result.tokens_per_sec, 4)
            if result.tokens_per_sec is not None
            else None,
            success=True,
            error_type=None,
            error_message=None,
            finish_reason=None,
            device=runtime.device,
            dtype=runtime.dtype,
            gpu_memory_mb=None,
            peak_gpu_memory_mb=round(result.peak_gpu_memory_mb, 4)
            if result.peak_gpu_memory_mb is not None
            else None,
            generated_text=result.text,
            event_type="request",
            timestamp_start=timestamp,
            timestamp_end=utc_now_iso(),
            endpoint="/generate_batch",
            message_len_chars=len(item.message.strip()),
            message_preview=item.message.strip()[:160],
            success_flag=True,
            error_text=None,
            latency_ms=round(latency_ms, 4),
        )
        append_jsonl_record(REQUEST_LOG_PATH, service_record)
        responses.append(
            GenerateResponse(
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
        )
    return {"items": responses}
