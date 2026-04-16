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
        from tensorrt_llm.runtime import ModelRunnerCpp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise TensorRTLLMUnavailable(
            "TensorRT-LLM runtime is unavailable in the current environment. "
            "Install/use the official TensorRT-LLM workflow first."
        ) from exc
    return ModelRunnerCpp


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
        model_runner_cls = _load_trtllm_runtime()
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

    def _normalize_token_ids(self, output_ids) -> list[int]:
        if isinstance(output_ids, torch.Tensor):
            data = output_ids.tolist()
        else:
            data = output_ids

        # Common TRT-LLM shapes:
        # - [tokens]
        # - [[tokens]]
        # - [[[tokens]]]
        while isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
            data = data[0]

        if not isinstance(data, list):
            raise TypeError(f"Unexpected output token container type: {type(data).__name__}")

        normalized: list[int] = []
        for item in data:
            if isinstance(item, torch.Tensor):
                if item.numel() != 1:
                    raise TypeError("Unexpected tensor item shape in output token ids")
                normalized.append(int(item.item()))
            elif isinstance(item, (int, float)):
                normalized.append(int(item))
            else:
                raise TypeError(f"Unexpected token id type: {type(item).__name__}")
        return normalized

    def _normalize_sequence_length(self, sequence_lengths) -> int | None:
        if sequence_lengths is None:
            return None
        data = sequence_lengths
        if isinstance(data, torch.Tensor):
            data = data.tolist()
        while isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, torch.Tensor):
            return int(data.item())
        if isinstance(data, (int, float)):
            return int(data)
        return None

    def _extract_generation_ids(
        self,
        output_ids,
        input_ids: list[int],
        sequence_length: int | None,
    ) -> list[int]:
        full_ids = self._normalize_token_ids(output_ids)
        if sequence_length is not None and sequence_length > 0:
            full_ids = full_ids[:sequence_length]

        prefix_len = 0
        max_prefix = min(len(full_ids), len(input_ids))
        while prefix_len < max_prefix and full_ids[prefix_len] == input_ids[prefix_len]:
            prefix_len += 1

        generated_ids = full_ids[prefix_len:]
        trimmed: list[int] = []
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        for token_id in generated_ids:
            if token_id == eos_id or token_id == pad_id:
                break
            trimmed.append(token_id)
        return trimmed

    def _strip_echo_from_text(self, prompt_text: str, generated_text: str) -> str:
        text = generated_text.strip()
        if prompt_text:
            normalized_prompt = prompt_text.strip()
            if text.startswith(normalized_prompt):
                text = text[len(normalized_prompt) :].lstrip()
        for marker in ("assistant\n", "assistant:", "assistant"):
            if text.lower().startswith(marker):
                text = text[len(marker) :].lstrip()
                break
        return text.strip()

    def generate_batch_with_stats(
        self,
        messages: list[str],
        max_new_tokens: int | None = None,
    ) -> list[GenerationStats]:
        prompts = [self._build_prompt(message) for message in messages]
        encoded_inputs = []
        input_token_ids_batch: list[list[int]] = []
        for prompt in prompts:
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_input_len,
            )
            input_ids = encoded["input_ids"][0]
            encoded_inputs.append(input_ids)
            input_token_ids_batch.append([int(token_id) for token_id in input_ids.tolist()])
        started = time.perf_counter()
        try:
            outputs = self._runner.generate(
                batch_input_ids=encoded_inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                temperature=self.config.temperature,
                top_k=1,
                top_p=1.0,
                num_beams=1,
                return_dict=True,
            )
        except TypeError:
            outputs = self._runner.generate(
                batch_input_ids=encoded_inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                temperature=self.config.temperature,
                top_k=1,
                top_p=1.0,
                beam_width=1,
                return_dict=True,
            )
        generation_ms = (time.perf_counter() - started) * 1000

        output_ids = None
        sequence_lengths = None
        if isinstance(outputs, dict):
            output_ids = outputs.get("output_ids")
            sequence_lengths = outputs.get("sequence_lengths") or outputs.get("sequence_length")
        elif hasattr(outputs, "output_ids"):
            output_ids = outputs.output_ids
            sequence_lengths = getattr(outputs, "sequence_lengths", None) or getattr(outputs, "sequence_length", None)
        else:
            output_ids = outputs

        if isinstance(sequence_lengths, torch.Tensor):
            sequence_lengths = sequence_lengths.tolist()
        results: list[GenerationStats] = []
        for index, input_token_ids in enumerate(input_token_ids_batch):
            row_output_ids = output_ids[index] if isinstance(output_ids, (list, tuple, torch.Tensor)) else output_ids
            row_sequence_length = None
            if isinstance(sequence_lengths, list) and index < len(sequence_lengths):
                row_sequence_length = sequence_lengths[index]
            elif isinstance(sequence_lengths, (int, float)):
                row_sequence_length = int(sequence_lengths)
            generated_ids = self._extract_generation_ids(row_output_ids, input_token_ids, row_sequence_length)
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if not generated_text:
                full_text = self.tokenizer.decode(
                    self._normalize_token_ids(row_output_ids),
                    skip_special_tokens=True,
                ).strip()
                prompt_text = self.tokenizer.decode(input_token_ids, skip_special_tokens=True).strip()
                generated_text = self._strip_echo_from_text(prompt_text, full_text)
            output_tokens = len(generated_ids)
            tokens_per_sec = None
            if generation_ms > 0 and output_tokens > 0:
                tokens_per_sec = output_tokens / (generation_ms / 1000.0)
            results.append(
                GenerationStats(
                    text=generated_text,
                    input_tokens=int(len(input_token_ids)),
                    output_tokens=output_tokens,
                    total_tokens=int(len(input_token_ids)) + output_tokens,
                    generation_ms=generation_ms,
                    tokens_per_sec=tokens_per_sec,
                    ttft_ms=None,
                    peak_gpu_memory_mb=None,
                )
            )
        return results

    def generate_with_stats(self, message: str, max_new_tokens: int | None = None) -> GenerationStats:
        prompt = self._build_prompt(message)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_len,
        )
        input_ids = encoded["input_ids"][0]
        input_token_ids = [int(token_id) for token_id in input_ids.tolist()]
        started = time.perf_counter()
        try:
            outputs = self._runner.generate(
                [input_ids],
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                temperature=self.config.temperature,
                top_k=1,
                top_p=1.0,
                num_beams=1,
                return_dict=True,
            )
        except TypeError:
            try:
                outputs = self._runner.generate(
                    batch_input_ids=[input_ids],
                    max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                    end_id=self.tokenizer.eos_token_id,
                    pad_id=self.tokenizer.pad_token_id,
                    temperature=self.config.temperature,
                    top_k=1,
                    top_p=1.0,
                    num_beams=1,
                    return_dict=True,
                )
            except (TypeError, UnboundLocalError, AttributeError):
                outputs = self._runner.generate(
                    batch_input_ids=[input_ids],
                    max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                    end_id=self.tokenizer.eos_token_id,
                    pad_id=self.tokenizer.pad_token_id,
                    temperature=self.config.temperature,
                    top_k=1,
                    top_p=1.0,
                    beam_width=1,
                    return_dict=True,
                )
        except (UnboundLocalError, AttributeError):
            outputs = self._runner.generate(
                batch_input_ids=[input_ids],
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                temperature=self.config.temperature,
                top_k=1,
                top_p=1.0,
                beam_width=1,
                return_dict=True,
            )
        generation_ms = (time.perf_counter() - started) * 1000

        output_ids = None
        sequence_length = None
        if isinstance(outputs, dict):
            output_ids = outputs.get("output_ids")
            sequence_length = outputs.get("sequence_lengths") or outputs.get("sequence_length")
        elif hasattr(outputs, "output_ids"):
            output_ids = outputs.output_ids
            sequence_length = getattr(outputs, "sequence_lengths", None) or getattr(
                outputs, "sequence_length", None
            )
        else:
            output_ids = outputs

        seq_len = self._normalize_sequence_length(sequence_length)
        generated_ids = self._extract_generation_ids(output_ids, input_token_ids, seq_len)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not generated_text:
            full_text = self.tokenizer.decode(
                self._normalize_token_ids(output_ids),
                skip_special_tokens=True,
            ).strip()
            prompt_text = self.tokenizer.decode(input_token_ids, skip_special_tokens=True).strip()
            generated_text = self._strip_echo_from_text(prompt_text, full_text)
        output_tokens = len(generated_ids)
        tokens_per_sec = None
        if generation_ms > 0 and output_tokens > 0:
            tokens_per_sec = output_tokens / (generation_ms / 1000.0)
        return GenerationStats(
            text=generated_text,
            input_tokens=int(input_ids.numel()),
            output_tokens=output_tokens,
            total_tokens=int(input_ids.numel()) + output_tokens,
            generation_ms=generation_ms,
            tokens_per_sec=tokens_per_sec,
            ttft_ms=None,
            peak_gpu_memory_mb=None,
        )


def get_backend(model_key: str | None = None) -> TrtllmDirectBackend:
    return TrtllmDirectBackend(resolve_trtllm_config(model_key))
