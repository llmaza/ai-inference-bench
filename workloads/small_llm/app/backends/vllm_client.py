from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

APP_DIR = Path(__file__).resolve().parents[1]


def _load_local_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_config = _load_local_module("small_llm_config_for_vllm", APP_DIR / "config.py")

REPO_ROOT = _config.REPO_ROOT
MODEL_CONFIG_DIR = _config.MODEL_CONFIG_DIR
SERVING_CONFIG_DIR = _config.SERVING_CONFIG_DIR
load_model_config = _config.load_model_config
load_serving_config = _config.load_serving_config

MODEL_REGISTRY: dict[str, Path] = {
    "qwen_1_5b_instruct": MODEL_CONFIG_DIR / "qwen_1_5b_instruct.yaml",
    "llama_3_2_1b_instruct": MODEL_CONFIG_DIR / "llama_3_2_1b_instruct.yaml",
}

SERVING_REGISTRY: dict[str, Path] = {
    "baseline_fastapi": SERVING_CONFIG_DIR / "baseline_fastapi.yaml",
    "trtllm_direct": SERVING_CONFIG_DIR / "trtllm.yaml",
    "triton": SERVING_CONFIG_DIR / "triton.yaml",
    "vllm": SERVING_CONFIG_DIR / "vllm.yaml",
}


def get_model_config(model_key: str | None = None):
    resolved = model_key or os.environ.get("MODEL_KEY", "").strip() or "qwen_1_5b_instruct"
    path = MODEL_REGISTRY.get(resolved)
    if path is None:
        raise KeyError(f"Unknown MODEL_KEY: {resolved}")
    return load_model_config(path)


def get_serving_config(serving_key: str | None = None):
    resolved = serving_key or os.environ.get("SMALL_LLM_SERVING_KEY", "").strip() or "baseline_fastapi"
    path = SERVING_REGISTRY.get(resolved)
    if path is None:
        raise KeyError(f"Unknown serving key: {resolved}")
    return load_serving_config(path)


DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_CHAT_PATH = "/chat/completions"
DEFAULT_VLLM_COMPLETIONS_PATH = "/completions"


@dataclass(frozen=True)
class VLLMOpenAIConfig:
    model_key: str
    model_name: str
    served_model_name: str
    base_url: str
    chat_path: str
    completions_path: str
    system_prompt: str
    max_input_len: int
    max_new_tokens: int
    temperature: float
    timeout_sec: float
    runtime_precision: str | None
    tokenizer_path: Path


def resolve_vllm_config(model_key: str | None = None) -> VLLMOpenAIConfig:
    model_cfg = get_model_config(model_key)
    serving_cfg = get_serving_config("vllm")
    base_url = os.getenv("SMALL_LLM_VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL).rstrip("/")
    runtime_precision = os.getenv("SMALL_LLM_VLLM_RUNTIME_PRECISION", "").strip() or None
    return VLLMOpenAIConfig(
        model_key=model_cfg.model_key,
        model_name=model_cfg.hf_model_name,
        served_model_name=os.getenv("SMALL_LLM_VLLM_MODEL_NAME", model_cfg.model_key),
        base_url=base_url,
        chat_path=os.getenv("SMALL_LLM_VLLM_CHAT_PATH", DEFAULT_VLLM_CHAT_PATH),
        completions_path=os.getenv("SMALL_LLM_VLLM_COMPLETIONS_PATH", DEFAULT_VLLM_COMPLETIONS_PATH),
        system_prompt=os.getenv("SMALL_LLM_SYSTEM_PROMPT", serving_cfg.system_prompt),
        max_input_len=int(os.getenv("SMALL_LLM_MAX_INPUT_TOKENS", str(model_cfg.default_max_input_tokens))),
        max_new_tokens=int(os.getenv("SMALL_LLM_MAX_NEW_TOKENS", str(serving_cfg.max_new_tokens))),
        temperature=float(os.getenv("SMALL_LLM_TEMPERATURE", str(serving_cfg.temperature))),
        timeout_sec=float(os.getenv("SMALL_LLM_VLLM_TIMEOUT_SEC", str(serving_cfg.timeout_sec))),
        runtime_precision=runtime_precision,
        tokenizer_path=Path(os.getenv("SMALL_LLM_VLLM_TOKENIZER_PATH", model_cfg.hf_model_name)),
    )


@dataclass
class VLLMGenerationStats:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    generation_ms: float
    tokens_per_sec: float | None
    ttft_ms: float | None
    peak_gpu_memory_mb: float | None
    finish_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "generation_ms": round(self.generation_ms, 4),
            "tokens_per_sec": round(self.tokens_per_sec, 4) if self.tokens_per_sec is not None else None,
            "ttft_ms": round(self.ttft_ms, 4) if self.ttft_ms is not None else None,
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 4) if self.peak_gpu_memory_mb is not None else None,
            "finish_reason": self.finish_reason,
        }


class VLLMOpenAIBackend:
    def __init__(self, config: VLLMOpenAIConfig):
        self.config = config
        tokenizer_source = self.config.tokenizer_path if self.config.tokenizer_path.exists() else self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, message: str) -> str:
        return self._build_prompt_from_messages(
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": message},
            ]
        )

    def _build_prompt_from_messages(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prompt_token_count(self, message: str) -> int:
        prompt = self._build_prompt(message)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_input_len)
        return int(encoded["input_ids"].shape[1])

    def _prompt_token_count_from_messages(self, messages: list[dict[str, str]]) -> int:
        prompt = self._build_prompt_from_messages(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_input_len)
        return int(encoded["input_ids"].shape[1])

    def _completion_token_count(self, text: str) -> int:
        if not text:
            return 0
        return int(len(self.tokenizer.encode(text, add_special_tokens=False)))

    def generate_with_stats(self, message: str, max_new_tokens: int | None = None) -> VLLMGenerationStats:
        prompt_tokens = self._prompt_token_count(message)
        payload = {
            "model": self.config.served_model_name,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": message},
            ],
            "max_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }
        started = time.perf_counter()
        response = requests.post(
            f"{self.config.base_url}{self.config.chat_path}",
            json=payload,
            timeout=self.config.timeout_sec,
        )
        generation_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        first_choice = choices[0] if choices else {}
        message_obj = first_choice.get("message") or {}
        generated_text = (message_obj.get("content") or first_choice.get("text") or "").strip()
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = self._completion_token_count(generated_text)
        output_tokens = int(output_tokens)
        total_tokens = int(input_tokens + output_tokens)
        tokens_per_sec = None
        if generation_ms > 0 and output_tokens > 0:
            tokens_per_sec = output_tokens / (generation_ms / 1000.0)
        return VLLMGenerationStats(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            generation_ms=generation_ms,
            tokens_per_sec=tokens_per_sec,
            ttft_ms=None,
            peak_gpu_memory_mb=None,
            finish_reason=first_choice.get("finish_reason"),
        )

    def generate_batch_with_stats(
        self,
        messages_batch: list[list[dict[str, str]]],
        max_new_tokens: int | None = None,
    ) -> list[VLLMGenerationStats]:
        prompts = [self._build_prompt_from_messages(messages) for messages in messages_batch]
        prompt_tokens = [self._prompt_token_count_from_messages(messages) for messages in messages_batch]
        payload = {
            "model": self.config.served_model_name,
            "prompt": prompts,
            "max_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }
        started = time.perf_counter()
        response = requests.post(
            f"{self.config.base_url}{self.config.completions_path}",
            json=payload,
            timeout=self.config.timeout_sec,
        )
        generation_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        choices_by_index = {choice.get("index", idx): choice for idx, choice in enumerate(choices)}
        stats: list[VLLMGenerationStats] = []
        for idx, messages in enumerate(messages_batch):
            choice = choices_by_index.get(idx) or {}
            text = (choice.get("text") or choice.get("message", {}).get("content") or "").strip()
            output_tokens = self._completion_token_count(text)
            tokens_per_sec = None
            if generation_ms > 0 and output_tokens > 0:
                tokens_per_sec = output_tokens / (generation_ms / 1000.0)
            stats.append(
                VLLMGenerationStats(
                    text=text,
                    input_tokens=int(prompt_tokens[idx]),
                    output_tokens=int(output_tokens),
                    total_tokens=int(prompt_tokens[idx] + output_tokens),
                    generation_ms=generation_ms,
                    tokens_per_sec=tokens_per_sec,
                    ttft_ms=None,
                    peak_gpu_memory_mb=None,
                    finish_reason=choice.get("finish_reason"),
                )
            )
        return stats


def get_backend(model_key: str | None = None) -> VLLMOpenAIBackend:
    return VLLMOpenAIBackend(resolve_vllm_config(model_key))
