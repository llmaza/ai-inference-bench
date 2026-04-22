#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO_ROOT/aienv/bin/activate" ]]; then
  source "$REPO_ROOT/aienv/bin/activate"
fi
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODE="${1:-debug}"
BACKEND="${2:-baseline_fastapi}"
MODEL_KEY="${3:-qwen_1_5b_instruct}"
NOTES="${4:-}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
REPEATS="${REPEATS:-3}"

if [[ "$MODE" != "debug" && "$MODE" != "main" ]]; then
  echo "error: mode must be 'debug' or 'main'"
  exit 1
fi
if [[ "$BACKEND" != "baseline_fastapi" && "$BACKEND" != "vllm" ]]; then
  echo "error: backend must be 'baseline_fastapi' or 'vllm'"
  exit 1
fi

for SWEEP in A B C D E F; do
  echo "[small_llm] mode=$MODE backend=$BACKEND sweep=$SWEEP"
  python workloads/small_llm/benchmarks/run_sweeps.py "$SWEEP"     --backend "$BACKEND"     --model-key "$MODEL_KEY"     --results-mode "$MODE"     --warmup-requests "$WARMUP_REQUESTS"     --repeats "$REPEATS"     --notes "$NOTES"
done
