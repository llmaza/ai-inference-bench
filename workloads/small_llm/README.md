# Shared Small LLM

This workload contains one shared small-LLM service that can be reused by:

- `workloads/rag`
- future `workloads/ocr`

## Architecture

- `app/registry.py`
  - model registry and serving registry
- `app/config.py`
  - flat YAML config loading
- `app/generation.py`
  - shared local generation backend
- `app/backends/local_fastapi.py`
  - Stage A runtime wiring
- `app/loaders/`
  - model-family-specific loader metadata
- `llm_inference.py`
  - compatibility wrapper for existing RAG imports

Current active default:

- `MODEL_KEY=qwen_1_5b_instruct`

Inactive placeholder for later switch:

- `MODEL_KEY=llama_3_2_1b_instruct`

## Benchmark Stages

- Stage A: local FastAPI baseline
- Stage B: TensorRT-LLM direct
- Stage C: Triton + TensorRT-LLM

Stage B now has setup/build/benchmark files in place, but it still depends on the official TensorRT-LLM environment being present. Stage C remains a placeholder.

## Running Stage A

Qwen default:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
MODEL_KEY=qwen_1_5b_instruct SMALL_LLM_SERVING_KEY=baseline_fastapi \
python -m uvicorn workloads.small_llm.app.main:app --host 0.0.0.0 --port 8010
```

Switch to Llama later:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
MODEL_KEY=llama_3_2_1b_instruct SMALL_LLM_SERVING_KEY=baseline_fastapi \
python -m uvicorn workloads.small_llm.app.main:app --host 0.0.0.0 --port 8010
```

## Benchmarking Stage A

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
python workloads/small_llm/benchmarks/run_baseline.py \
  --base-url http://127.0.0.1:8010 \
  --prompts workloads/small_llm/benchmarks/prompts.labor_sample.jsonl \
  --repeats 3 \
  --concurrency 1 \
  --timeout-sec 180
```

## Stage B: TensorRT-LLM Direct

Stage B uses the same prompt set and generation settings as Stage A, but runs direct TensorRT-LLM inference for the same Qwen model.

Main files:

- `workloads/small_llm/app/backends/trtllm_direct.py`
- `workloads/small_llm/benchmarks/run_trtllm.py`
- `workloads/small_llm/benchmarks/compare_baseline_vs_trtllm.py`
- `workloads/small_llm/scripts/build_trt_engine.sh`
- `workloads/small_llm/scripts/run_trtllm_one.sh`

Important:

- this uses the official TensorRT-LLM convert + `trtllm-build` workflow
- it requires the TensorRT-LLM runtime/tooling to actually be installed
- if `trtllm-build` or `tensorrt_llm` is missing, the scripts fail explicitly instead of pretending to work

Build engine:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
export MODEL_KEY=qwen_1_5b_instruct
export TRTLLM_CONVERT_SCRIPT=/path/to/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py
bash workloads/small_llm/scripts/build_trt_engine.sh
```

Run one prompt:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
python workloads/small_llm/benchmarks/run_trtllm.py \
  --model-key qwen_1_5b_instruct \
  --single-prompt "Summarize the main rules around probation periods in labor law in 3 short bullet points."
```

Run the TRT-LLM benchmark:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
python workloads/small_llm/benchmarks/run_trtllm.py \
  --model-key qwen_1_5b_instruct \
  --prompts workloads/small_llm/benchmarks/prompts.labor_sample.jsonl \
  --repeats 3
```

Compare against Stage A:

```bash
python workloads/small_llm/benchmarks/compare_baseline_vs_trtllm.py \
  --baseline-summary /path/to/baseline_summary.json \
  --trtllm-summary /path/to/trtllm_summary.json
```

## Hugging Face Access

The local FastAPI baseline may still need Hugging Face access once to download model weights into the local cache. That does not change the architecture: Stage A remains the local FastAPI baseline.

If a model requires access, export `HF_TOKEN` before first download:

```bash
export HF_TOKEN=your_hugging_face_token
```

Llama verification example:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
python - <<'PY'
from transformers import AutoTokenizer
name = "meta-llama/Llama-3.2-1B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
print("ok", tok.name_or_path)
PY
```
