from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from workloads.small_llm.app.registry import get_model_config, get_serving_config


def _load_tokenizer(model_key: str):
    model_cfg = get_model_config(model_key)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model_cfg


def _prompt_token_count(tokenizer, system_prompt: str, message: str, max_input_len: int = 8192) -> int:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len)
    return int(encoded["input_ids"].shape[1])


def build_message_for_target_prompt_len(
    *,
    model_key: str,
    system_prompt: str,
    target_prompt_len: int,
    seed_text: str,
    max_input_len: int = 8192,
) -> tuple[str, int]:
    tokenizer, _ = _load_tokenizer(model_key)
    target_prompt_len = max(1, int(target_prompt_len))

    def make_message(repetitions: int) -> str:
        filler = "контекст"
        body = " ".join([filler] * max(0, repetitions)).strip()
        if body:
            return f"{seed_text} {body}".strip()
        return seed_text.strip()

    baseline = _prompt_token_count(tokenizer, system_prompt, seed_text, max_input_len=max_input_len)
    if baseline >= target_prompt_len:
        return seed_text.strip(), baseline

    low = 0
    high = 1
    while high < 10000:
        actual = _prompt_token_count(tokenizer, system_prompt, make_message(high), max_input_len=max_input_len)
        if actual >= target_prompt_len:
            break
        low = high
        high *= 2

    best_message = make_message(high)
    best_actual = _prompt_token_count(tokenizer, system_prompt, best_message, max_input_len=max_input_len)
    best_delta = abs(best_actual - target_prompt_len)

    while low <= high:
        mid = (low + high) // 2
        actual = _prompt_token_count(tokenizer, system_prompt, make_message(mid), max_input_len=max_input_len)
        delta = abs(actual - target_prompt_len)
        if delta < best_delta:
            best_delta = delta
            best_message = make_message(mid)
            best_actual = actual
        if actual < target_prompt_len:
            low = mid + 1
        else:
            high = mid - 1

    return best_message, best_actual


def materialize_sweep_prompts(
    *,
    model_key: str,
    target_prompt_len: int,
    prompts_source: Path,
    output_dir: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    model_cfg = get_model_config(model_key)
    serving_cfg = get_serving_config("baseline_fastapi")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(prompts_source.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_id = row.get("prompt_id") or row.get("prompt_name") or f"sweep_{index:03d}"
        seed_text = row.get("message") or row.get("text") or f"Сформируй краткий ответ для {prompt_id}."
        message, actual_prompt_len = build_message_for_target_prompt_len(
            model_key=model_key,
            system_prompt=serving_cfg.system_prompt,
            target_prompt_len=target_prompt_len,
            seed_text=seed_text,
            max_input_len=model_cfg.default_max_input_tokens,
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "message": message,
                "target_prompt_len": int(target_prompt_len),
                "actual_prompt_len": int(actual_prompt_len),
            }
        )

    if not rows:
        raise ValueError(f"No prompts found in {prompts_source}")

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="small_llm_sweep_prompts_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_key}_prompt_len_{target_prompt_len}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"prompt_id": row["prompt_id"], "message": row["message"]}, ensure_ascii=False) + "\n")
    return output_path, rows


def sleep_for_offered_load(run_started_at: float, request_index: int, offered_load_rps: float | int | None) -> None:
    if not offered_load_rps or offered_load_rps <= 0:
        return
    scheduled_start = run_started_at + (request_index / float(offered_load_rps))
    delay = scheduled_start - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
