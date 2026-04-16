#!/usr/bin/env python3
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device() -> str:
    requested = os.environ.get("LOCAL_LLM_DEVICE", "").strip().lower()
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(device: str, torch_dtype_str: str):
    if device != "cuda":
        return torch.float32
    if torch_dtype_str.lower() == "float16":
        return torch.float16
    if torch_dtype_str.lower() == "float32":
        return torch.float32
    return torch.bfloat16


@dataclass
class GenerationStats:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    generation_ms: float
    tokens_per_sec: float | None
    ttft_ms: float | None
    peak_gpu_memory_mb: float | None

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
        }


@dataclass
class LocalChatBackend:
    model_name: str
    device: str
    max_input_tokens: int
    torch_dtype_str: str

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=_resolve_torch_dtype(self.device, self.torch_dtype_str),
        )
        self.model.eval()
        self.model.to(self.device)
        gen_cfg = self.model.generation_config
        gen_cfg.do_sample = False
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.top_k = None
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # Serialize inference on the shared CUDA model to avoid concurrent kernel
        # launches from the FastAPI threadpool. This keeps the baseline stable under
        # sweep concurrency while we validate the service behavior.
        self._inference_lock = threading.Lock()

    def _prepare_inputs(self, messages: list[dict]) -> dict[str, torch.Tensor]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float = 0.15,
    ) -> str:
        return self.generate_with_stats(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ).text

    def generate_batch_with_stats(
        self,
        messages_batch: list[list[dict]],
        max_new_tokens: int,
        temperature: float = 0.15,
    ) -> list[GenerationStats]:
        with self._inference_lock:
            encoded_prompts = []
            for messages in messages_batch:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded_prompts.append(prompt)
            encoded = self.tokenizer(
                encoded_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
                padding=True,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            attention_mask = encoded.get("attention_mask")
            input_lengths = attention_mask.sum(dim=1).tolist() if attention_mask is not None else [encoded["input_ids"].shape[1]] * len(messages_batch)
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
            generation_ms = (time.perf_counter() - started) * 1000
            peak_gpu_memory_mb = None
            if self.device == "cuda":
                peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            results: list[GenerationStats] = []
            for row_index, input_len in enumerate(input_lengths):
                output_ids = generated[row_index][int(input_len):]
                text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                output_tokens = int(output_ids.shape[0])
                tokens_per_sec = None
                if generation_ms > 0 and output_tokens > 0:
                    tokens_per_sec = output_tokens / (generation_ms / 1000.0)
                results.append(
                    GenerationStats(
                        text=text,
                        input_tokens=int(input_len),
                        output_tokens=output_tokens,
                        total_tokens=int(input_len + output_tokens),
                        generation_ms=generation_ms,
                        tokens_per_sec=tokens_per_sec,
                        ttft_ms=None,
                        peak_gpu_memory_mb=peak_gpu_memory_mb,
                    )
                )
            return results

    def generate_with_stats(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float = 0.15,
    ) -> GenerationStats:
        del temperature
        with self._inference_lock:
            encoded = self._prepare_inputs(messages)
            input_len = encoded["input_ids"].shape[1]
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
            generation_ms = (time.perf_counter() - started) * 1000
            output_ids = generated[0][input_len:]
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            output_tokens = int(output_ids.shape[0])
            peak_gpu_memory_mb = None
            if self.device == "cuda":
                peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            tokens_per_sec = None
            if generation_ms > 0 and output_tokens > 0:
                tokens_per_sec = output_tokens / (generation_ms / 1000.0)
            return GenerationStats(
                text=text,
                input_tokens=int(input_len),
                output_tokens=output_tokens,
                total_tokens=int(input_len + output_tokens),
                generation_ms=generation_ms,
                tokens_per_sec=tokens_per_sec,
                ttft_ms=None,
                peak_gpu_memory_mb=peak_gpu_memory_mb,
            )


@lru_cache(maxsize=8)
def get_local_chat_backend(
    model_name: str,
    device: str | None = None,
    max_input_tokens: int = 8192,
    torch_dtype_str: str = "bfloat16",
) -> LocalChatBackend:
    return LocalChatBackend(
        model_name=model_name,
        device=device or resolve_device(),
        max_input_tokens=max_input_tokens,
        torch_dtype_str=torch_dtype_str,
    )

