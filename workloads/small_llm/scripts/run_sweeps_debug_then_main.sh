#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND="${1:-baseline_fastapi}"
MODEL_KEY="${2:-qwen_1_5b_instruct}"
NOTES_BASE="${3:-iter}"

"$SCRIPT_DIR/run_sweeps_iteration.sh" debug "$BACKEND" "$MODEL_KEY" "${NOTES_BASE}_debug"

echo
echo "Debug sweep run completed for backend=$BACKEND."
echo "If debug looks good, run main with:"
echo "  $SCRIPT_DIR/run_sweeps_iteration.sh main $BACKEND $MODEL_KEY ${NOTES_BASE}_main"
