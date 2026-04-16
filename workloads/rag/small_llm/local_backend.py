#!/usr/bin/env python3
from __future__ import annotations

"""
Compatibility wrapper.

The shared small-LLM inference logic now lives in:
    workloads/small_llm/llm_inference.py
"""

from workloads.small_llm.llm_inference import (  # noqa: F401
    LOCAL_LLM_DEFAULT,
    LocalChatBackend,
    get_local_chat_backend,
    resolve_device,
)
