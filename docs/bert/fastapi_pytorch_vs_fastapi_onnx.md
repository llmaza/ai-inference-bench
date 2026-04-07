# FastAPI PyTorch vs FastAPI ONNX Runtime

## Summary

This report compares two end-to-end FastAPI services for the same RuBERT topic classifier:

- `FastAPI + PyTorch`
- `FastAPI + ONNX Runtime`

Both services used the same:

- local BERT model artifacts
- scenario input file
- benchmark harness
- request concurrency
- warmup count
- GPU hardware

The goal was to measure end-to-end API latency and throughput, not just backend-only model execution.

## Measured Results

| Setup | p50 ms | p95 ms | mean ms | throughput RPS | api p50 ms | api p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| FastAPI + PyTorch | 22.22 | 31.16 | 22.82 | 171.86 | 18.72 | 26.53 |
| FastAPI + ONNX Runtime | 8.42 | 14.11 | 8.66 | 442.82 | 4.81 | 10.24 |

## Interpretation

`FastAPI + ONNX Runtime` outperformed `FastAPI + PyTorch` across all reported service metrics.

- Throughput improved from `171.86 RPS` to `442.82 RPS`.
- End-to-end p50 latency improved from `22.22 ms` to `8.42 ms`.
- End-to-end p95 latency improved from `31.16 ms` to `14.11 ms`.
- Service-reported model latency also dropped substantially:
  - `api p50`: `18.72 ms` to `4.81 ms`
  - `api p95`: `26.53 ms` to `10.24 ms`

This indicates the ONNX Runtime gain is not limited to backend-only synthetic tests. The improvement carries through into a realistic API-serving setup built on FastAPI.

## Why ONNX Was Faster Here

The primary difference between the two services is the inference backend:

- PyTorch service executes the classifier through PyTorch.
- ONNX service executes the exported model through ONNX Runtime with CUDA.

In this setup, ONNX Runtime reduced inference overhead and executed the model graph more efficiently on GPU. Since both services keep the same FastAPI layer and request shape, the large improvement is best explained by the model execution backend rather than by API code differences.

## Methodology

### Scope

This comparison measures end-to-end service behavior for single-message requests.

- API request shape: one input message per request
- concurrency: `4`
- total requests: `100`
- warmup requests: `100`
- hardware: `NVIDIA GeForce RTX 3080`

This is not yet a batched API comparison. At the time of measurement, the benchmark was run against `/predict`, not `/predict_batch`.

### Services Compared

PyTorch service:

- path: `/home/user/projects/ai-inference-bench/workloads/bert_classifier/baseline_fastapi`
- backend: PyTorch

ONNX service:

- path: `/home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_fastapi`
- backend: ONNX Runtime

### Input Scenario

The benchmark replayed messages from:

- `/home/user/projects/ai-inference-bench/benchmark/scenarios/bert_inputs.jsonl`

### Output Files

Benchmark outputs were written to:

- raw request results:
  - `/home/user/projects/ai-inference-bench/results/raw/02eb2b3d-ddff-435d-8951-5e861a8869ab_requests.jsonl`
  - `/home/user/projects/ai-inference-bench/results/raw/82cdde0f-10f2-473f-8ff9-fa4ca827ea3a_requests.jsonl`
- summary table:
  - `/home/user/projects/ai-inference-bench/results/tables/benchmark_runs.csv`

## Commands

### Environment

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench
```

### Start FastAPI + PyTorch

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench/workloads/bert_classifier/baseline_fastapi
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Benchmark FastAPI + PyTorch

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench
python -m benchmark.runner \
  --base-url http://127.0.0.1:8001 \
  --serving-mode fastapi_pytorch \
  --model-name rubert_topic_classifier_pytorch \
  --hardware "NVIDIA GeForce RTX 3080" \
  --warmup-requests 100 \
  --notes "FastAPI + PyTorch"
```

### Start FastAPI + ONNX Runtime

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_fastapi
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
```

### Benchmark FastAPI + ONNX Runtime

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench
python -m benchmark.runner \
  --base-url http://127.0.0.1:8002 \
  --serving-mode fastapi_onnx \
  --model-name rubert_topic_classifier_onnx \
  --hardware "NVIDIA GeForce RTX 3080" \
  --warmup-requests 100 \
  --notes "FastAPI + ONNX Runtime"
```

## Next Step

The next useful comparison is batched API serving:

- `FastAPI + PyTorch` via `/predict_batch`
- `FastAPI + ONNX Runtime` via `/predict_batch`

That will let us compare API-layer batch sizes such as `2`, `4`, `8`, and `16`, before moving on to Triton dynamic batching.
