from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    message: str = Field(..., min_length=1)


class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    generation_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_sec: Optional[float]
    ttft_ms: Optional[float]
    peak_gpu_memory_mb: Optional[float]
    model_key: str
    model_name: str
    device: str

