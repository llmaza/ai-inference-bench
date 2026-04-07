import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = REPO_ROOT / "workloads" / "bert_classifier" / "artifacts" / "model"


@dataclass
class ModelRuntime:
    model_dir: Path
    max_length: int
    device: torch.device
    tokenizer: object
    model: object
    id2label: Dict[str, str]
    torch: object
    cuda_available: bool
    cuda_version: Optional[str]
    torch_version: str
    gpu_name: Optional[str]
    gpu_count: int
    gpu_total_memory_mb: Optional[float]


def resolve_model_dir() -> Path:
    configured_path = os.getenv("BERT_MODEL_DIR")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_MODEL_DIR.resolve()


def load_id2label(model_dir: Path) -> Dict[str, str]:
    with open(model_dir / "id2label.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_gpu_metrics(device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "device_used": str(device),
            "gpu_id": None,
            "gpu_name": None,
            "gpu_mem_alloc_mb": None,
            "gpu_mem_reserved_mb": None,
            "gpu_mem_peak_mb": None,
        }

    gpu_id = torch.cuda.current_device()
    return {
        "device_used": str(device),
        "gpu_id": gpu_id,
        "gpu_name": torch.cuda.get_device_name(gpu_id),
        "gpu_mem_alloc_mb": round(torch.cuda.memory_allocated(gpu_id) / 1024**2, 2),
        "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved(gpu_id) / 1024**2, 2),
        "gpu_mem_peak_mb": round(torch.cuda.max_memory_allocated(gpu_id) / 1024**2, 2),
    }


def get_runtime() -> ModelRuntime:
    model_dir = resolve_model_dir()
    max_length = int(os.getenv("BERT_MAX_LENGTH", "256"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_total_memory_mb = (
        round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2)
        if device.type == "cuda"
        else None
    )

    return ModelRuntime(
        model_dir=model_dir,
        max_length=max_length,
        device=device,
        tokenizer=tokenizer,
        model=model,
        id2label=load_id2label(model_dir),
        torch=torch,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_total_memory_mb=gpu_total_memory_mb,
    )
