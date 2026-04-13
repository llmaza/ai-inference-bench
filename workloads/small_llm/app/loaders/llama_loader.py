from __future__ import annotations


def get_llama_loader_config() -> dict:
    return {
        "loader_key": "llama",
        "family": "llama",
        "supports_chat_template": True,
    }

