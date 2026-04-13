#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-}"
if [[ -z "$PROMPT" ]]; then
  echo "Usage: $0 \"your prompt here\"" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO_ROOT/aienv/bin/activate" ]]; then
  source "$REPO_ROOT/aienv/bin/activate"
fi
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
python workloads/small_llm/benchmarks/run_trtllm.py --single-prompt "$PROMPT"
