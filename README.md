# ai-inference-bench

`ai-inference-bench` is an AI inference benchmark suite for comparing serving and runtime approaches across multiple workload types.

Only Phase A1 is currently implemented. The active scope is a BERT classifier baseline served with FastAPI, a standardized BERT benchmark harness, and an ONNX export plus runtime-comparison path. ONNX/Triton/hybrid/embeddings/LLM areas outside the current BERT work remain placeholders.

## Current Status

Implemented now:
- BERT baseline FastAPI service
- BERT benchmark runner and standardized result outputs
- BERT ONNX export and parity validation
- Direct runtime comparison for PyTorch vs ONNX Runtime on CPU and CUDA
- Local BERT model artifacts stored inside this repo

Present but still placeholders:
- Triton serving path
- Hybrid BERT + regex workload
- Embeddings workload
- Small LLM workload

## Repository Overview

```text
ai-inference-bench/
├── benchmark/                   # Active benchmark harness for Phase A1
├── docs/                        # Methodology and case-study notes
├── results/
│   ├── raw/                     # Per-request outputs and runtime logs
│   ├── summaries/               # Reserved for future summary artifacts
│   ├── plots/                   # Reserved for future plots
│   └── tables/                  # Aggregate benchmark tables
├── tests/                       # Placeholder test modules
└── workloads/
    ├── bert_classifier/
    │   ├── artifacts/model/     # Local BERT model bundle used in Phase A1
    │   ├── baseline_fastapi/    # Implemented FastAPI baseline
    │   ├── onnx_export/         # Implemented ONNX export and validation
    │   ├── triton/              # Placeholder
    │   └── configs/
    ├── hybrid_bert_regex/       # Placeholder
    ├── embeddings/              # Placeholder
    └── small_llm/               # Placeholder
```

## Quickstart: Local Environment

Create a dedicated environment for this repo:

```bash
cd /home/user/projects/ai-inference-bench
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `onnxruntime-gpu` is included for CUDA-enabled ONNX Runtime benchmarking.
- If you only need CPU ONNX tests, `onnxruntime` is sufficient.
- The repo now defaults to its local model bundle under `workloads/bert_classifier/artifacts/model`.

## Quickstart: BERT Baseline

Start the BERT baseline FastAPI service:

```bash
cd /home/user/projects/ai-inference-bench/workloads/bert_classifier/baseline_fastapi
source /home/user/projects/ai-inference-bench/.venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Notes:
- The service uses CUDA if available.
- Service request logs are written under `results/raw/bert_classifier/baseline_fastapi/`.

## Run the FastAPI Benchmark

In a second terminal, with the same environment activated:

```bash
source /home/user/projects/ai-inference-bench/.venv/bin/activate
cd /home/user/projects/ai-inference-bench
python -m benchmark.runner \
  --base-url http://127.0.0.1:8001 \
  --warmup-requests 100 \
  --hardware "NVIDIA GeForce RTX 3080"
```

## Run the Runtime Comparison

Backend-only comparison, without FastAPI:

```bash
source /home/user/projects/ai-inference-bench/.venv/bin/activate
cd /home/user/projects/ai-inference-bench
python -m benchmark.compare_backends \
  --onnx-path /home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/bert_classifier.onnx \
  --dataset-path /home/user/projects/ai-inference-bench/benchmark/scenarios/bert_inputs.jsonl \
  --num-texts 100 \
  --batch-sizes 1 4 8 16 \
  --warmup-batches 4 \
  --backends pytorch_cpu pytorch_cuda onnx_cpu onnx_cuda
```

## Expected Outputs

After running benchmarks, you should see:

- Per-request FastAPI benchmark outputs in:
  - `results/raw/<run_id>_requests.jsonl`
- Aggregate FastAPI benchmark table in:
  - `results/tables/benchmark_runs.csv`
- Service-side request logs in:
  - `results/raw/bert_classifier/baseline_fastapi/requests_bert.jsonl`
- Backend comparison raw outputs in:
  - `results/raw/backend_comparison_<run_id>.jsonl`
- Backend comparison table in:
  - `results/tables/backend_comparison.csv`

## Scope Boundary

This README describes the current Phase A1 state only. Future Triton, hybrid, embeddings, and LLM sections are intentionally not documented as working features yet because they are still placeholders.
