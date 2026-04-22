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
- Stage C: Triton OpenAI-compatible TRT-LLM serving

Stage B uses the local TensorRT-LLM runtime directly.
Stage C uses the `trtllm-serve` OpenAI-compatible endpoint from the official TRT-LLM container/tooling.

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
  --prompts workloads/small_llm/benchmarks/prompts/prompts_rag_medium_tk_rk_baseline.jsonl \
  --repeats 3 \
  --concurrency 1 \
  --timeout-sec 180
```

Canonical Stage A outputs:

- request logs:
  - `workloads/small_llm/results/stage_a_baseline/requests/requests_small_llm.jsonl`
- benchmark per-request runs:
  - `workloads/small_llm/results/stage_a_baseline/runs/*.jsonl`
- benchmark run summaries:
  - `workloads/small_llm/results/stage_a_baseline/runs/*_summary.json`

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
  --prompts workloads/small_llm/benchmarks/prompts/prompts_rag_medium_tk_rk_baseline.jsonl \
  --repeats 3
```

Canonical Stage B outputs:

- request logs:
  - `workloads/small_llm/results/stage_b_trtllm_direct/requests/requests_trtllm.jsonl`
- benchmark per-request runs:
  - `workloads/small_llm/results/stage_b_trtllm_direct/runs/*.jsonl`
- benchmark run summaries:
  - `workloads/small_llm/results/stage_b_trtllm_direct/runs/*_summary.json`
- artifact metadata:
  - `workloads/small_llm/results/stage_b_trtllm_direct/artifacts/<model_key>_metadata.json`

## Stage C: Triton OpenAI-Compatible TRT-LLM

Stage C benchmarks the same prompt set against a Triton-compatible OpenAI API backed by TRT-LLM.

Start the server inside the official TRT-LLM / Triton container:

```bash
cd /home/user/projects/ai-inference-bench
export TRITON_MODEL_PATH=/home/user/projects/ai-inference-bench
export TRITON_TOKENIZER_PATH=/home/user/projects/ai-inference-bench
bash workloads/small_llm/scripts/start_triton.sh
```

Run one prompt:

```bash
cd /home/user/projects/ai-inference-bench
python workloads/small_llm/benchmarks/run_triton.py \
  --model-key qwen_1_5b_instruct \
  --single-prompt "Summarize the main rules around probation periods in labor law in 3 short bullet points."
```

Run the Triton benchmark:

```bash
cd /home/user/projects/ai-inference-bench
python workloads/small_llm/benchmarks/run_triton.py \
  --model-key qwen_1_5b_instruct \
  --prompts workloads/small_llm/benchmarks/prompts/prompts_rag_medium_tk_rk_baseline.jsonl \
  --repeats 3
```

Canonical Stage C outputs:

- request logs:
  - `workloads/small_llm/results/stage_c_triton_trtllm/requests/requests_triton_trtllm.jsonl`
- benchmark per-request runs:
  - `workloads/small_llm/results/stage_c_triton_trtllm/runs/*.jsonl`
- benchmark run summaries:
  - `workloads/small_llm/results/stage_c_triton_trtllm/runs/*_summary.json`
- Triton/server metadata:
  - `workloads/small_llm/results/stage_c_triton_trtllm/artifacts/<model_key>_triton_server_metadata.json`

Older benchmark files under `workloads/small_llm/results/benchmarks/` are still valid historical outputs, but the stage-specific paths above are now the canonical locations.

## Sweep Runs

Sweeps are an extra layer on top of the existing benchmark scripts. The current Stage A, Stage B, and Stage C CLIs stay unchanged.

Sweep aliases:

- `A` -> `prefill`
- `B` -> `decode`
- `C` -> `concurrency`
- `D` -> `long_context_concurrency`
- `E` -> `batching`
- `F` -> `overload`

Supported backends:

- `baseline_fastapi`
- `trtllm_direct`
- `triton`
- `vllm`

Example commands:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
python workloads/small_llm/benchmarks/run_sweeps.py A --backend baseline_fastapi
python workloads/small_llm/benchmarks/run_sweeps.py B --backend trtllm_direct
python workloads/small_llm/benchmarks/run_sweeps.py C --backend triton
python workloads/small_llm/benchmarks/run_sweeps.py A --backend vllm
```

Sweep outputs:

- global CSV ledger:
  - `workloads/small_llm/results/tables/sweep_runs.csv`
- per-sweep CSVs:
  - `workloads/small_llm/results/sweeps/*.csv`

The vLLM path expects a running OpenAI-compatible vLLM server.

The sweep runner appends one row per sweep point to the global ledger and the matching per-sweep CSV.

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
