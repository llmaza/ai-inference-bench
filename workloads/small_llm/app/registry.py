from __future__ import annotations

import os
from pathlib import Path

from .config import (
    MODEL_CONFIG_DIR,
    SERVING_CONFIG_DIR,
    ModelConfig,
    ServingConfig,
    load_model_config,
    load_serving_config,
)


MODEL_REGISTRY: dict[str, Path] = {
    "qwen_1_5b_instruct": MODEL_CONFIG_DIR / "qwen_1_5b_instruct.yaml",
    "llama_3_2_1b_instruct": MODEL_CONFIG_DIR / "llama_3_2_1b_instruct.yaml",
}

SERVING_REGISTRY: dict[str, Path] = {
    "baseline_fastapi": SERVING_CONFIG_DIR / "baseline_fastapi.yaml",
    "trtllm_direct": SERVING_CONFIG_DIR / "trtllm.yaml",
    "triton": SERVING_CONFIG_DIR / "triton.yaml",
}

DEFAULT_MODEL_KEY = "qwen_1_5b_instruct"
DEFAULT_SERVING_KEY = "baseline_fastapi"


def resolve_model_key(explicit_model_key: str | None = None) -> str:
    if explicit_model_key:
        return explicit_model_key
    for env_name in ("MODEL_KEY", "SMALL_LLM_MODEL_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return DEFAULT_MODEL_KEY


def resolve_serving_key(explicit_serving_key: str | None = None) -> str:
    if explicit_serving_key:
        return explicit_serving_key
    return os.environ.get("SMALL_LLM_SERVING_KEY", DEFAULT_SERVING_KEY).strip() or DEFAULT_SERVING_KEY


def get_model_config(model_key: str | None = None) -> ModelConfig:
    resolved = resolve_model_key(model_key)
    path = MODEL_REGISTRY.get(resolved)
    if path is None:
        raise KeyError(f"Unknown MODEL_KEY: {resolved}")
    return load_model_config(path)


def get_serving_config(serving_key: str | None = None) -> ServingConfig:
    resolved = resolve_serving_key(serving_key)
    path = SERVING_REGISTRY.get(resolved)
    if path is None:
        raise KeyError(f"Unknown serving key: {resolved}")
    return load_serving_config(path)

