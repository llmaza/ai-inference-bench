import time
from typing import List


def validate_message(message: str) -> str:
    return message.strip()


def validate_messages(messages: List[str]) -> List[str]:
    return [message.strip() for message in messages]


def prepare_inputs(texts, tokenizer, max_length: int):
    t0 = time.perf_counter()
    inputs = tokenizer(
        texts,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    tokenize_ms = (time.perf_counter() - t0) * 1000
    token_lengths = [int(mask.sum()) for mask in inputs["attention_mask"]]
    return inputs, token_lengths, tokenize_ms


def prepare_single_input(text: str, tokenizer, max_length: int):
    inputs, token_lengths, tokenize_ms = prepare_inputs(
        texts=[text],
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return inputs, token_lengths[0], tokenize_ms
