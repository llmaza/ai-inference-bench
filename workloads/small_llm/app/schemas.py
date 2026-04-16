from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_new_tokens: Optional[int] = None


class GenerateBatchItem(BaseModel):
    message: str = Field(..., min_length=1)
    prompt_name: Optional[str] = None
    prompt_file: Optional[str] = None
    request_index: Optional[int] = None
    repeat_index: Optional[int] = None
    max_new_tokens: Optional[int] = None


class GenerateBatchRequest(BaseModel):
    items: List[GenerateBatchItem] = Field(..., min_length=1)


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
