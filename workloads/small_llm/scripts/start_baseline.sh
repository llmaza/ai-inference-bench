#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO_ROOT/aienv/bin/activate" ]]; then
  source "$REPO_ROOT/aienv/bin/activate"
fi
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export MODEL_KEY="${MODEL_KEY:-qwen_1_5b_instruct}"
export SMALL_LLM_SERVING_KEY="${SMALL_LLM_SERVING_KEY:-baseline_fastapi}"

python -m uvicorn workloads.small_llm.app.main:app --host 0.0.0.0 --port 8010
