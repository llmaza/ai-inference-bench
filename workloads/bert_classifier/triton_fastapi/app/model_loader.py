import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.bert_classifier.triton.client import TritonBertClient

DEFAULT_MODEL_DIR = REPO_ROOT / "workloads" / "bert_classifier" / "artifacts" / "model"
DEFAULT_TRITON_URL = os.getenv(
    "BERT_TRITON_URL",
    "http://127.0.0.1:8000/v2/models/bert_classifier/infer",
)


@dataclass
class TritonFastAPIRuntime:
    model_dir: Path
    triton_url: str
    max_length: int
    client: TritonBertClient
    id2label: Dict[str, str]


def resolve_model_dir() -> Path:
    configured_path = os.getenv("BERT_MODEL_DIR")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_MODEL_DIR.resolve()


def get_runtime() -> TritonFastAPIRuntime:
    model_dir = resolve_model_dir()
    max_length = int(os.getenv("BERT_MAX_LENGTH", "256"))
    triton_url = os.getenv("BERT_TRITON_URL", DEFAULT_TRITON_URL)
    client = TritonBertClient(
        triton_url=triton_url,
        model_dir=model_dir,
        max_length=max_length,
        timeout_sec=float(os.getenv("BERT_TRITON_TIMEOUT_SEC", "30.0")),
    )
    return TritonFastAPIRuntime(
        model_dir=model_dir,
        triton_url=triton_url,
        max_length=max_length,
        client=client,
        id2label=client.id2label,
    )
