"""Shared small-LLM workload package."""

from .llm_inference import (
    LOCAL_LLM_DEFAULT,
    LOCAL_LLM_LLAMA,
    LOCAL_LLM_QWEN,
    GenerationStats,
    LocalChatBackend,
    get_local_chat_backend,
    resolve_device,
    resolve_model_name,
)

__all__ = [
    "GenerationStats",
    "LOCAL_LLM_DEFAULT",
    "LOCAL_LLM_LLAMA",
    "LOCAL_LLM_QWEN",
    "LocalChatBackend",
    "get_local_chat_backend",
    "resolve_device",
    "resolve_model_name",
]
