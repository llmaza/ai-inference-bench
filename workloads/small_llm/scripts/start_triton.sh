#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

TRITON_MODEL_PATH="${TRITON_MODEL_PATH:-$REPO_ROOT}"
TRITON_TOKENIZER_PATH="${TRITON_TOKENIZER_PATH:-$REPO_ROOT}"
TRITON_HOST="${TRITON_HOST:-0.0.0.0}"
TRITON_PORT="${TRITON_PORT:-8000}"
TRITON_MAX_BATCH_SIZE="${TRITON_MAX_BATCH_SIZE:-1}"
TRITON_MAX_NUM_TOKENS="${TRITON_MAX_NUM_TOKENS:-8192}"
TRITON_MAX_SEQ_LEN="${TRITON_MAX_SEQ_LEN:-8704}"

if ! command -v trtllm-serve >/dev/null 2>&1; then
  echo "trtllm-serve is not available. Run this inside the official TRT-LLM / Triton container." >&2
  exit 1
fi

if [[ ! -f "$TRITON_MODEL_PATH/rank0.engine" && ! -f "$TRITON_MODEL_PATH/engine.plan" ]]; then
  echo "TensorRT engine not found under: $TRITON_MODEL_PATH" >&2
  exit 1
fi

if [[ ! -e "$TRITON_TOKENIZER_PATH/tokenizer.json" && ! -e "$TRITON_TOKENIZER_PATH/tokenizer_config.json" ]]; then
  echo "Tokenizer files not found under: $TRITON_TOKENIZER_PATH" >&2
  exit 1
fi

exec trtllm-serve serve "$TRITON_MODEL_PATH" \
  --tokenizer "$TRITON_TOKENIZER_PATH" \
  --backend trt \
  --host "$TRITON_HOST" \
  --port "$TRITON_PORT" \
  --max_batch_size "$TRITON_MAX_BATCH_SIZE" \
  --max_num_tokens "$TRITON_MAX_NUM_TOKENS" \
  --max_seq_len "$TRITON_MAX_SEQ_LEN" \
  --log_level info
