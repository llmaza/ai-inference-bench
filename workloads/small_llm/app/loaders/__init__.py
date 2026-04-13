from __future__ import annotations

from .llama_loader import get_llama_loader_config
from .qwen_loader import get_qwen_loader_config


def get_loader_config(loader_key: str) -> dict:
    if loader_key == "qwen":
        return get_qwen_loader_config()
    if loader_key == "llama":
        return get_llama_loader_config()
    raise KeyError(f"Unknown loader_key: {loader_key}")

