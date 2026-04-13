#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${MODEL_KEY:-qwen_1_5b_instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export REPO_ROOT
PYTHON_BIN="${PYTHON_BIN:-}"

cd "$REPO_ROOT"
if ! command -v trtllm-build >/dev/null 2>&1; then
  if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO_ROOT/aienv/bin/activate" ]]; then
    source "$REPO_ROOT/aienv/bin/activate"
  fi
fi
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No python interpreter found on PATH." >&2
    exit 1
  fi
fi

TRTLLM_CONVERT_SCRIPT="${TRTLLM_CONVERT_SCRIPT:-}"
if [[ -z "${TRTLLM_CONVERT_SCRIPT}" ]]; then
  echo "Set TRTLLM_CONVERT_SCRIPT to the official TensorRT-LLM Qwen convert_checkpoint.py path." >&2
  echo "Example: export TRTLLM_CONVERT_SCRIPT=/path/to/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py" >&2
  exit 1
fi

if ! command -v trtllm-build >/dev/null 2>&1; then
  echo "trtllm-build is not available on PATH. Install/use the official TensorRT-LLM environment first." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m pip show huggingface_hub >/dev/null 2>&1; then
  echo "huggingface_hub is required in the active Python environment for model snapshot download." >&2
  echo "Checked interpreter: $PYTHON_BIN" >&2
  exit 1
fi

readarray -t CFG < <("$PYTHON_BIN" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
from workloads.small_llm.app.registry import get_model_config, get_serving_config
import os

model_key = os.environ.get("MODEL_KEY", "qwen_1_5b_instruct")
model_cfg = get_model_config(model_key)
serving_cfg = get_serving_config("trtllm_direct")
repo_root = Path(os.environ["REPO_ROOT"])

source_dir = Path(str(serving_cfg.model_source_dir))
if not source_dir.is_absolute():
    source_dir = repo_root / source_dir
source_dir.parent.mkdir(parents=True, exist_ok=True)
local_dir = snapshot_download(
    repo_id=model_cfg.hf_model_name,
    local_dir=str(source_dir),
    local_dir_use_symlinks=False,
)

converted_dir = Path(str(serving_cfg.converted_checkpoint_dir))
engine_dir = Path(str(serving_cfg.engine_dir))
if not converted_dir.is_absolute():
    converted_dir = repo_root / converted_dir
if not engine_dir.is_absolute():
    engine_dir = repo_root / engine_dir

print(local_dir)
print(converted_dir)
print(engine_dir)
print(serving_cfg.max_input_len)
print(serving_cfg.max_seq_len)
print(serving_cfg.max_batch_size)
print(serving_cfg.gemm_plugin)
PY
)

MODEL_SOURCE_DIR="${CFG[0]}"
CONVERTED_DIR="${CFG[1]}"
ENGINE_DIR="${CFG[2]}"
MAX_INPUT_LEN="${CFG[3]}"
MAX_SEQ_LEN="${CFG[4]}"
MAX_BATCH_SIZE="${CFG[5]}"
GEMM_PLUGIN="${CFG[6]}"

mkdir -p "$CONVERTED_DIR" "$ENGINE_DIR"

"$PYTHON_BIN" "$TRTLLM_CONVERT_SCRIPT" \
  --model_dir "$MODEL_SOURCE_DIR" \
  --output_dir "$CONVERTED_DIR" \
  --dtype bfloat16

trtllm-build \
  --checkpoint_dir "$CONVERTED_DIR" \
  --output_dir "$ENGINE_DIR" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --max_input_len "$MAX_INPUT_LEN" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --gemm_plugin "$GEMM_PLUGIN"

echo "TensorRT-LLM engine built at: $ENGINE_DIR"
