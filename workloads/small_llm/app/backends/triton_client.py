from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from ..config import REPO_ROOT
from ..registry import get_model_config, get_serving_config


DEFAULT_TRITON_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TRITON_CHAT_PATH = "/chat/completions"
DEFAULT_TRITON_COMPLETIONS_PATH = "/completions"


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _metadata_value(metadata: dict[str, Any] | None, key: str, default: Any) -> Any:
    if not metadata:
        return default
    value = metadata.get(key)
    return default if value in (None, "") else value


@dataclass(frozen=True)
class TritonOpenAIConfig:
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
    engine_path: Path
    tokenizer_path: Path
    artifact_metadata_path: Path


def resolve_triton_config(model_key: str | None = None) -> TritonOpenAIConfig:
    model_cfg = get_model_config(model_key)
    serving_cfg = get_serving_config("triton")
    artifact_metadata_path = _resolve_repo_path(
        os.getenv(
            "SMALL_LLM_TRITON_ARTIFACT_METADATA_PATH",
            str(
                REPO_ROOT
                / "workloads"
                / "small_llm"
                / "results"
                / "stage_b_trtllm_direct"
                / "artifacts"
                / f"{model_cfg.model_key}_metadata.json"
            ),
        )
    )
    artifact_metadata = _load_json_if_exists(artifact_metadata_path)
    max_input_len = int(
        os.getenv(
            "SMALL_LLM_MAX_INPUT_TOKENS",
            str(_metadata_value(artifact_metadata, "max_input_len", model_cfg.default_max_input_tokens)),
        )
    )
    max_new_tokens = int(
        os.getenv(
            "SMALL_LLM_MAX_NEW_TOKENS",
            str(
                _metadata_value(
                    artifact_metadata,
                    "max_output_len",
                    serving_cfg.max_new_tokens or model_cfg.default_max_new_tokens,
                )
            ),
        )
    )
    runtime_precision = None
    if artifact_metadata:
        runtime_precision = artifact_metadata.get("runtime_precision")
    if runtime_precision is None:
        runtime_precision = os.getenv("SMALL_LLM_RUNTIME_PRECISION", "").strip() or None
    base_url = os.getenv("SMALL_LLM_TRITON_BASE_URL", DEFAULT_TRITON_BASE_URL).rstrip("/")
    served_model_name = os.getenv(
        "SMALL_LLM_TRITON_MODEL_NAME",
        model_cfg.model_key,
    )
    return TritonOpenAIConfig(
        model_key=model_cfg.model_key,
        model_name=model_cfg.hf_model_name,
        served_model_name=served_model_name,
        base_url=base_url,
        chat_path=os.getenv("SMALL_LLM_TRITON_CHAT_PATH", DEFAULT_TRITON_CHAT_PATH),
        completions_path=os.getenv("SMALL_LLM_TRITON_COMPLETIONS_PATH", DEFAULT_TRITON_COMPLETIONS_PATH),
        system_prompt=os.getenv("SMALL_LLM_SYSTEM_PROMPT", serving_cfg.system_prompt),
        max_input_len=max_input_len,
        max_new_tokens=max_new_tokens,
        temperature=float(os.getenv("SMALL_LLM_TEMPERATURE", str(serving_cfg.temperature))),
        timeout_sec=float(os.getenv("SMALL_LLM_TIMEOUT_SEC", str(serving_cfg.timeout_sec))),
        runtime_precision=runtime_precision,
        engine_path=_resolve_repo_path(
            os.getenv(
                "SMALL_LLM_TRITON_ENGINE_PATH",
                str(REPO_ROOT / "rank0.engine"),
            )
        ),
        tokenizer_path=_resolve_repo_path(
            os.getenv("SMALL_LLM_TRITON_TOKENIZER_PATH", str(REPO_ROOT))
        ),
        artifact_metadata_path=artifact_metadata_path,
    )


@dataclass
class TritonGenerationStats:
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
            "tokens_per_sec": round(self.tokens_per_sec, 4)
            if self.tokens_per_sec is not None
            else None,
            "ttft_ms": round(self.ttft_ms, 4) if self.ttft_ms is not None else None,
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 4)
            if self.peak_gpu_memory_mb is not None
            else None,
            "finish_reason": self.finish_reason,
        }


class TritonOpenAIBackend:
    def __init__(self, config: TritonOpenAIConfig):
        self.config = config
        tokenizer_source = (
            self.config.tokenizer_path
            if self.config.tokenizer_path.exists()
            else self.config.model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, message: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": message},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_prompt_from_messages(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prompt_token_count_from_messages(self, messages: list[dict[str, str]]) -> int:
        prompt = self._build_prompt_from_messages(messages)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_len,
        )
        return int(encoded["input_ids"].shape[1])

    def _prompt_token_count(self, message: str) -> int:
        prompt = self._build_prompt(message)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_len,
        )
        return int(encoded["input_ids"].shape[1])

    def _completion_token_count(self, text: str) -> int:
        if not text:
            return 0
        return int(len(self.tokenizer.encode(text, add_special_tokens=False)))

    def generate_with_stats(self, message: str, max_new_tokens: int | None = None) -> TritonGenerationStats:
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
        finish_reason = first_choice.get("finish_reason")
        return TritonGenerationStats(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            generation_ms=generation_ms,
            tokens_per_sec=tokens_per_sec,
            ttft_ms=None,
            peak_gpu_memory_mb=None,
            finish_reason=finish_reason,
        )


    def generate_batch_with_stats(
        self,
        messages_batch: list[list[dict[str, str]]],
        max_new_tokens: int | None = None,
    ) -> list[TritonGenerationStats]:
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
        stats: list[TritonGenerationStats] = []
        for idx, _messages in enumerate(messages_batch):
            choice = choices_by_index.get(idx) or {}
            text = (choice.get("text") or choice.get("message", {}).get("content") or "").strip()
            output_tokens = self._completion_token_count(text)
            tokens_per_sec = None
            if generation_ms > 0 and output_tokens > 0:
                tokens_per_sec = output_tokens / (generation_ms / 1000.0)
            stats.append(
                TritonGenerationStats(
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


@dataclass(frozen=True)
class TritonRuntime:
    config: TritonOpenAIConfig
    backend: TritonOpenAIBackend


def get_backend(model_key: str | None = None) -> TritonOpenAIBackend:
    return TritonOpenAIBackend(resolve_triton_config(model_key))
