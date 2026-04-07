import time
from typing import Dict, List


def validate_message(message: str) -> str:
    return message.strip()


def validate_messages(messages: List[str]) -> List[str]:
    return [message.strip() for message in messages]


def prepare_inputs(
    texts,
    tokenizer,
    device,
    max_length: int,
    torch_module,
):
    t0 = time.perf_counter()
    inputs_cpu = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    tokenize_ms = (time.perf_counter() - t0) * 1000

    token_lengths = [
        int(mask.sum().item()) for mask in inputs_cpu["attention_mask"]
    ]

    t1 = time.perf_counter()
    inputs = {k: v.to(device) for k, v in inputs_cpu.items()}
    if device.type == "cuda":
        torch_module.cuda.synchronize()
    to_device_ms = (time.perf_counter() - t1) * 1000

    return inputs, token_lengths, tokenize_ms, to_device_ms


def prepare_single_input(text: str, tokenizer, device, max_length: int, torch_module):
    inputs, token_lengths, tokenize_ms, to_device_ms = prepare_inputs(
        texts=[text],
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        torch_module=torch_module,
    )
    return inputs, token_lengths[0], tokenize_ms, to_device_ms


def build_top3_scores(top_probs, top_ids, id2label: Dict[str, str], class_score_cls):
    return [
        class_score_cls(
            topic=id2label[str(cls_id.item())],
            confidence=round(cls_prob.item(), 4),
        )
        for cls_prob, cls_id in zip(top_probs[0], top_ids[0])
    ]


def build_top3_scores_for_index(
    top_probs,
    top_ids,
    row_index: int,
    id2label: Dict[str, str],
    class_score_cls,
):
    return [
        class_score_cls(
            topic=id2label[str(cls_id.item())],
            confidence=round(cls_prob.item(), 4),
        )
        for cls_prob, cls_id in zip(top_probs[row_index], top_ids[row_index])
    ]
