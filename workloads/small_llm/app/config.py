from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SMALL_LLM_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = SMALL_LLM_ROOT / "configs"
MODEL_CONFIG_DIR = CONFIG_ROOT / "models"
SERVING_CONFIG_DIR = CONFIG_ROOT / "serving"


def _coerce_scalar(value: str):
    text = value.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def load_flat_yaml(path: Path) -> dict:
    data = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line in {path}: {raw_line}")
        key, value = raw_line.split(":", 1)
        data[key.strip()] = _coerce_scalar(value)
    return data


@dataclass(frozen=True)
class ModelConfig:
    model_key: str
    display_name: str
    hf_model_name: str
    loader_key: str
    active_default: bool
    default_max_input_tokens: int
    default_max_new_tokens: int
    torch_dtype_cuda: str


@dataclass(frozen=True)
class ServingConfig:
    backend_key: str
    host: str
    port: int
    system_prompt: str
    max_new_tokens: int
    temperature: float
    concurrency_default: int
    timeout_sec: float
    max_batch_size: int
    engine_dir: str
    converted_checkpoint_dir: str
    model_source_dir: str
    max_input_len: int
    max_seq_len: int
    gemm_plugin: str


def load_model_config(path: Path) -> ModelConfig:
    data = load_flat_yaml(path)
    return ModelConfig(
        model_key=str(data["model_key"]),
        display_name=str(data["display_name"]),
        hf_model_name=str(data["hf_model_name"]),
        loader_key=str(data["loader_key"]),
        active_default=bool(data.get("active_default", False)),
        default_max_input_tokens=int(data.get("default_max_input_tokens", 8192)),
        default_max_new_tokens=int(data.get("default_max_new_tokens", 512)),
        torch_dtype_cuda=str(data.get("torch_dtype_cuda", "bfloat16")),
    )


def load_serving_config(path: Path) -> ServingConfig:
    data = load_flat_yaml(path)
    return ServingConfig(
        backend_key=str(data["backend_key"]),
        host=str(data.get("host", "0.0.0.0")),
        port=int(data.get("port", 8010)),
        system_prompt=str(data["system_prompt"]),
        max_new_tokens=int(data.get("max_new_tokens", 512)),
        temperature=float(data.get("temperature", 0.15)),
        concurrency_default=int(data.get("concurrency_default", 1)),
        timeout_sec=float(data.get("timeout_sec", 180.0)),
        max_batch_size=int(data.get("max_batch_size", 1)),
        engine_dir=str(data.get("engine_dir", "")),
        converted_checkpoint_dir=str(data.get("converted_checkpoint_dir", "")),
        model_source_dir=str(data.get("model_source_dir", "")),
        max_input_len=int(data.get("max_input_len", 8192)),
        max_seq_len=int(data.get("max_seq_len", 8704)),
        gemm_plugin=str(data.get("gemm_plugin", "auto")),
    )
