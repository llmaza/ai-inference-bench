import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import onnxruntime as ort
from transformers import AutoTokenizer


REPO_ROOT = Path(os.getenv("AI_BENCH_REPO_ROOT", "/app"))
DEFAULT_MODEL_DIR = REPO_ROOT / "workloads" / "bert_classifier" / "artifacts" / "model"
DEFAULT_ONNX_PATH = (
    REPO_ROOT / "workloads" / "bert_classifier" / "onnx_export" / "bert_classifier.onnx"
)


@dataclass
class OnnxRuntimeState:
    model_dir: Path
    onnx_path: Path
    max_length: int
    tokenizer: object
    id2label: Dict[str, str]
    session: ort.InferenceSession
    providers: List[str]
    active_provider: str


def resolve_model_dir() -> Path:
    configured_path = os.getenv("BERT_MODEL_DIR")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_MODEL_DIR.resolve()


def resolve_onnx_path() -> Path:
    configured_path = os.getenv("BERT_ONNX_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_ONNX_PATH.resolve()


def load_id2label(model_dir: Path) -> Dict[str, str]:
    with open(model_dir / "id2label.json", "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_providers() -> List[str]:
    requested_provider = os.getenv("BERT_ONNX_PROVIDER")
    available = ort.get_available_providers()
    if requested_provider:
        if requested_provider not in available:
            raise RuntimeError(
                "Requested ONNX provider %s not available. Available providers: %s"
                % (requested_provider, ", ".join(available))
            )
        return [requested_provider]

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_runtime() -> OnnxRuntimeState:
    model_dir = resolve_model_dir()
    onnx_path = resolve_onnx_path()
    max_length = int(os.getenv("BERT_MAX_LENGTH", "256"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    providers = resolve_providers()
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_provider = session.get_providers()[0] if session.get_providers() else "unknown"

    return OnnxRuntimeState(
        model_dir=model_dir,
        onnx_path=onnx_path,
        max_length=max_length,
        tokenizer=tokenizer,
        id2label=load_id2label(model_dir),
        session=session,
        providers=session.get_providers(),
        active_provider=active_provider,
    )


def get_provider_metrics(runtime: OnnxRuntimeState) -> Dict[str, Optional[str]]:
    return {
        "provider": runtime.active_provider,
        "providers": ",".join(runtime.providers),
    }
