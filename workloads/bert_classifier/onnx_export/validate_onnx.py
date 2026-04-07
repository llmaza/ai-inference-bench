import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "workloads" / "bert_classifier" / "artifacts" / "model"
DEFAULT_ONNX_PATH = SCRIPT_DIR / "bert_classifier.onnx"
DEFAULT_SCENARIO_PATH = REPO_ROOT / "benchmark" / "scenarios" / "bert_inputs.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PyTorch vs ONNX parity for BERT classifier")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--onnx-path", type=str, default=str(DEFAULT_ONNX_PATH))
    parser.add_argument("--scenario-path", type=str, default=str(DEFAULT_SCENARIO_PATH))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Optional direct input text. Can be passed multiple times.",
    )
    return parser


def load_sample_texts(scenario_path: Path, direct_texts: List[str], num_samples: int) -> List[str]:
    if direct_texts:
        return direct_texts[:num_samples]

    samples = []
    with scenario_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            message = row.get("message", "").strip()
            if message:
                samples.append(message)
            if len(samples) >= num_samples:
                break
    if not samples:
        raise ValueError("No validation texts found")
    return samples


def build_ort_inputs(encoded_inputs, session_inputs) -> dict:
    ort_inputs = {}
    for input_meta in session_inputs:
        name = input_meta.name
        if name not in encoded_inputs:
            raise KeyError("ONNX session expects missing input: %s" % name)
        ort_inputs[name] = encoded_inputs[name].cpu().numpy()
    return ort_inputs


def compare_models(
    texts: List[str],
    tokenizer,
    pt_model,
    ort_session,
    max_length: int,
) -> Tuple[int, float]:
    matched = 0
    max_abs_diff = 0.0

    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        with torch.no_grad():
            pt_logits = pt_model(**encoded).logits.cpu().numpy()

        ort_inputs = build_ort_inputs(encoded, ort_session.get_inputs())
        ort_logits = ort_session.run(["logits"], ort_inputs)[0]

        pt_pred = int(np.argmax(pt_logits, axis=1)[0])
        ort_pred = int(np.argmax(ort_logits, axis=1)[0])
        if pt_pred == ort_pred:
            matched += 1

        sample_diff = float(np.max(np.abs(pt_logits - ort_logits)))
        if sample_diff > max_abs_diff:
            max_abs_diff = sample_diff

    return matched, max_abs_diff


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required for validation. Install it in the active environment with: "
            "pip install onnxruntime"
        ) from exc

    model_dir = Path(args.model_dir).expanduser().resolve()
    onnx_path = Path(args.onnx_path).expanduser().resolve()
    scenario_path = Path(args.scenario_path).expanduser().resolve()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    pt_model.eval()
    pt_model.to("cpu")

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    texts = load_sample_texts(scenario_path, args.text, args.num_samples)
    matched, max_abs_diff = compare_models(
        texts=texts,
        tokenizer=tokenizer,
        pt_model=pt_model,
        ort_session=ort_session,
        max_length=args.max_length,
    )

    total = len(texts)
    print("ONNX validation complete")
    print("model_dir:", model_dir)
    print("onnx_path:", onnx_path)
    print("samples:", total)
    print("class_parity: %d/%d" % (matched, total))
    print("class_parity_rate:", round(matched / total, 4) if total else None)
    print("max_absolute_difference:", max_abs_diff)


if __name__ == "__main__":
    main()
