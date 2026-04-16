from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch

from ..config import REPO_ROOT
from ..generation import get_local_chat_backend, resolve_device
from ..loaders import get_loader_config
from ..registry import get_model_config, get_serving_config


@dataclass
class Runtime:
    backend: object
    model_key: str
    model_name: str
    display_name: str
    device: str
    dtype: str
    backend_key: str
    loader_key: str
    max_input_tokens: int
    max_new_tokens: int
    temperature: float
    system_prompt: str
    torch_version: str
    python_version: str
    gpu_name: str | None
    gpu_total_memory_mb: float | None


def _gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _gpu_total_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return round(props.total_memory / (1024 * 1024), 2)


def get_runtime(model_key: str | None = None, serving_key: str | None = None) -> Runtime:
    model_cfg = get_model_config(model_key)
    serving_cfg = get_serving_config(serving_key)
    loader_cfg = get_loader_config(model_cfg.loader_key)
    device = resolve_device()
    max_input_tokens = int(
        os.getenv("SMALL_LLM_MAX_INPUT_TOKENS", str(model_cfg.default_max_input_tokens))
    )
    max_new_tokens = int(
        os.getenv("SMALL_LLM_MAX_NEW_TOKENS", str(serving_cfg.max_new_tokens or model_cfg.default_max_new_tokens))
    )
    system_prompt = os.getenv("SMALL_LLM_SYSTEM_PROMPT", serving_cfg.system_prompt)
    dtype = model_cfg.torch_dtype_cuda if device == "cuda" else "float32"
    backend = get_local_chat_backend(
        model_name=model_cfg.hf_model_name,
        device=device,
        max_input_tokens=max_input_tokens,
        torch_dtype_str=model_cfg.torch_dtype_cuda,
    )
    return Runtime(
        backend=backend,
        model_key=model_cfg.model_key,
        model_name=model_cfg.hf_model_name,
        display_name=model_cfg.display_name,
        device=device,
        dtype=dtype,
        backend_key=serving_cfg.backend_key,
        loader_key=loader_cfg["loader_key"],
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=float(os.getenv("SMALL_LLM_TEMPERATURE", str(serving_cfg.temperature))),
        system_prompt=system_prompt,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        gpu_name=_gpu_name(),
        gpu_total_memory_mb=_gpu_total_memory_mb(),
    )
