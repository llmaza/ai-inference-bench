#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${MODEL_KEY:-qwen_1_5b_instruct}"

cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
python - <<'PY'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from workloads.small_llm.app.registry import get_model_config, resolve_model_key

model_key = resolve_model_key(os.environ.get("MODEL_KEY"))
cfg = get_model_config(model_key)
print(f"Downloading {cfg.hf_model_name} for MODEL_KEY={cfg.model_key}")
AutoTokenizer.from_pretrained(cfg.hf_model_name)
AutoModelForCausalLM.from_pretrained(cfg.hf_model_name)
print("Done")
PY

