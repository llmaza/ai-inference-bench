import argparse
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SOURCE_MODEL_DIR = REPO_ROOT / "workloads" / "bert_classifier" / "artifacts" / "model"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "bert_classifier.onnx"
DEFAULT_SAMPLE_TEXT = "Просим направить документы по валютному договору"


class LogitsOnlyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the BERT classifier to ONNX")
    parser.add_argument("--model-dir", type=str, default=str(SOURCE_MODEL_DIR))
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--sample-text", type=str, default=DEFAULT_SAMPLE_TEXT)
    parser.add_argument(
        "--fixed-shape",
        action="store_true",
        help="Export with fixed batch/sequence shape instead of dynamic axes.",
    )
    parser.add_argument(
        "--use-dynamo-exporter",
        action="store_true",
        help="Use the newer dynamo-based exporter. May require onnxscript.",
    )
    return parser


def load_export_inputs(tokenizer, sample_text: str, max_length: int) -> Dict[str, torch.Tensor]:
    return tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def get_input_names(inputs: Dict[str, torch.Tensor]) -> List[str]:
    ordered_names = ["input_ids", "attention_mask", "token_type_ids"]
    return [name for name in ordered_names if name in inputs]


def get_dynamic_axes(input_names: List[str]) -> Dict[str, Dict[int, str]]:
    dynamic_axes = {"logits": {0: "batch_size"}}
    for name in input_names:
        dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
    return dynamic_axes


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "onnx is required for export. Install it in the active environment with: "
            "pip install onnx"
        ) from exc

    model_dir = Path(args.model_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to("cpu")

    inputs = load_export_inputs(
        tokenizer=tokenizer,
        sample_text=args.sample_text,
        max_length=args.max_length,
    )
    input_names = get_input_names(inputs)
    input_tensors = tuple(inputs[name] for name in input_names)

    dynamic_axes = None if args.fixed_shape else get_dynamic_axes(input_names)

    wrapper = LogitsOnlyWrapper(model)
    with torch.no_grad():
        export_kwargs = {
            "input_names": input_names,
            "output_names": ["logits"],
            "dynamic_axes": dynamic_axes,
            "opset_version": args.opset,
            "do_constant_folding": True,
        }

        if args.use_dynamo_exporter:
            export_kwargs["dynamo"] = True
        else:
            # Prefer the legacy exporter for compatibility with environments
            # that do not have the newer onnxscript dependency installed.
            export_kwargs["dynamo"] = False

        torch.onnx.export(
            wrapper,
            input_tensors,
            str(output_path),
            **export_kwargs,
        )

    print("ONNX export complete")
    print("model_dir:", model_dir)
    print("output_path:", output_path)
    print("tokenizer:", getattr(tokenizer, "name_or_path", str(model_dir)))
    print("max_length:", args.max_length)
    print("opset:", args.opset)
    print("dynamic_axes:", not args.fixed_shape)
    print("use_dynamo_exporter:", args.use_dynamo_exporter)
    print("input_names:", ",".join(input_names))
    print("output_names: logits")


if __name__ == "__main__":
    main()
