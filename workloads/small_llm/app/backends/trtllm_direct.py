from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..config import REPO_ROOT
from ..generation import GenerationStats
from ..registry import get_model_config, get_serving_config


class TensorRTLLMUnavailable(RuntimeError):
    pass


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_trtllm_runtime():
    try:
        from tensorrt_llm.runtime import ModelRunnerCpp, SamplingConfig
    except Exception as exc:  # pragma: no cover - environment dependent
        raise TensorRTLLMUnavailable(
            "TensorRT-LLM runtime is unavailable in the current environment. "
            "Install/use the official TensorRT-LLM workflow first."
        ) from exc
    return ModelRunnerCpp, SamplingConfig


@dataclass
class TrtllmConfig:
    model_key: str
    model_name: str
    engine_dir: Path
    max_input_len: int
    max_new_tokens: int
    temperature: float
    system_prompt: str


def resolve_trtllm_config(model_key: str | None = None) -> TrtllmConfig:
    model_cfg = get_model_config(model_key)
    serving_cfg = get_serving_config("trtllm_direct")
    return TrtllmConfig(
        model_key=model_cfg.model_key,
        model_name=model_cfg.hf_model_name,
        engine_dir=_resolve_repo_path(str(os.getenv("TRTLLM_ENGINE_DIR", serving_cfg.engine_dir))),
        max_input_len=int(os.getenv("TRTLLM_MAX_INPUT_LEN", str(serving_cfg.max_input_len))),
        max_new_tokens=int(os.getenv("TRTLLM_MAX_NEW_TOKENS", str(serving_cfg.max_new_tokens))),
        temperature=float(os.getenv("TRTLLM_TEMPERATURE", str(serving_cfg.temperature))),
        system_prompt=os.getenv("SMALL_LLM_SYSTEM_PROMPT", serving_cfg.system_prompt),
    )


class TrtllmDirectBackend:
    def __init__(self, config: TrtllmConfig):
        self.config = config
        if not self.config.engine_dir.exists():
            raise FileNotFoundError(
                f"TensorRT-LLM engine dir not found: {self.config.engine_dir}. "
                "Run workloads/small_llm/scripts/build_trt_engine.sh first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_runner_cls, sampling_config_cls = _load_trtllm_runtime()
        self._sampling_config_cls = sampling_config_cls
        self._runner = model_runner_cls.from_dir(engine_dir=str(self.config.engine_dir))

    def _build_prompt(self, message: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": message},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_with_stats(self, message: str) -> GenerationStats:
        prompt = self._build_prompt(message)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_len,
        )
        input_ids = encoded["input_ids"][0].tolist()
        sampling_config = self._sampling_config_cls(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            temperature=self.config.temperature,
            top_k=1,
            top_p=1.0,
        )
        started = time.perf_counter()
        try:
            outputs = self._runner.generate(
                [input_ids],
                sampling_config=sampling_config,
                max_new_tokens=self.config.max_new_tokens,
            )
        except TypeError:
            outputs = self._runner.generate(
                batch_input_ids=[input_ids],
                sampling_config=sampling_config,
                max_new_tokens=self.config.max_new_tokens,
            )
        generation_ms = (time.perf_counter() - started) * 1000

        output_ids = None
        if isinstance(outputs, dict):
            output_ids = outputs.get("output_ids")
        elif hasattr(outputs, "output_ids"):
            output_ids = outputs.output_ids
        else:
            output_ids = outputs

        if isinstance(output_ids, torch.Tensor):
            token_ids = output_ids[0].tolist()
        elif isinstance(output_ids, list) and output_ids and isinstance(output_ids[0], list):
            token_ids = output_ids[0]
        else:
            token_ids = output_ids

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        output_tokens = len(token_ids)
        tokens_per_sec = None
        if generation_ms > 0 and output_tokens > 0:
            tokens_per_sec = output_tokens / (generation_ms / 1000.0)
        peak_gpu_memory_mb = None
        if torch.cuda.is_available():
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return GenerationStats(
            text=text,
            input_tokens=len(input_ids),
            output_tokens=output_tokens,
            total_tokens=len(input_ids) + output_tokens,
            generation_ms=generation_ms,
            tokens_per_sec=tokens_per_sec,
            ttft_ms=None,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
        )


def get_backend(model_key: str | None = None) -> TrtllmDirectBackend:
    return TrtllmDirectBackend(resolve_trtllm_config(model_key))
