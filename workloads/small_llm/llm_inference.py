#!/usr/bin/env python3
from __future__ import annotations

from workloads.small_llm.app.generation import (
    GenerationStats,
    LocalChatBackend,
    get_local_chat_backend as _get_local_chat_backend,
    resolve_device,
)
from workloads.small_llm.app.registry import get_model_config, resolve_model_key


LOCAL_LLM_QWEN = get_model_config("qwen_1_5b_instruct").hf_model_name
LOCAL_LLM_LLAMA = get_model_config("llama_3_2_1b_instruct").hf_model_name
LOCAL_LLM_DEFAULT = LOCAL_LLM_QWEN


def resolve_model_name(explicit_model_name: str | None = None) -> str:
    if explicit_model_name:
        return explicit_model_name
    for env_name in ("LOCAL_LLM_MODEL", "HF_LLM_MODEL"):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return env_value
    return get_model_config(resolve_model_key()).hf_model_name


def get_local_chat_backend(
    model_name: str | None = None,
    device: str | None = None,
    max_input_tokens: int | None = None,
) -> LocalChatBackend:
    resolved_model_name = resolve_model_name(model_name)
    model_cfg = None
    for key in ("qwen_1_5b_instruct", "llama_3_2_1b_instruct"):
        cfg = get_model_config(key)
        if cfg.hf_model_name == resolved_model_name:
            model_cfg = cfg
            break
    if model_cfg is None:
        max_input = max_input_tokens if max_input_tokens is not None else 8192
        return _get_local_chat_backend(
            model_name=resolved_model_name,
            device=device or resolve_device(),
            max_input_tokens=max_input,
            torch_dtype_str="bfloat16",
        )
    return _get_local_chat_backend(
        model_name=model_cfg.hf_model_name,
        device=device or resolve_device(),
        max_input_tokens=max_input_tokens or model_cfg.default_max_input_tokens,
        torch_dtype_str=model_cfg.torch_dtype_cuda,
    )
