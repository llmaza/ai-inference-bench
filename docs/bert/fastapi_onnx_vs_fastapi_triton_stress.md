# FastAPI + ONNX vs FastAPI + Triton(ONNX) Under Stress

## Summary

This report compares two text-serving API paths for the same RuBERT classifier under higher concurrency:

- `FastAPI + ONNX Runtime`
- `FastAPI + Triton(ONNX)`

At this stage, Triton dynamic batching was enabled in the Triton model config. The goal was to test whether Triton’s scheduler could outperform the in-process ONNX Runtime FastAPI service under heavier load.

## Key Result

For this workload and this machine, `FastAPI + ONNX Runtime` remained faster than `FastAPI + Triton(ONNX)` across both:

- single-item `/predict`
- batched `/predict_batch` with `batch_size=4`

Even under higher concurrency, Triton did not overtake the in-process ONNX Runtime FastAPI baseline.

## Single-Item Stress Results

| Concurrency | ONNX p50 ms | ONNX p95 ms | ONNX throughput RPS | Triton p50 ms | Triton p95 ms | Triton throughput RPS |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 10.61 | 18.14 | 678.42 | 20.22 | 28.29 | 373.34 |
| 16 | 19.27 | 28.57 | 751.76 | 41.00 | 54.92 | 372.58 |
| 32 | 35.29 | 58.84 | 753.49 | 77.42 | 98.11 | 375.03 |

### Observed

- `FastAPI + ONNX Runtime` was substantially faster at every tested concurrency.
- Triton throughput stayed nearly flat around `~373-375 RPS`.
- ONNX Runtime scaled to around `~678-753 RPS`.
- Triton latency degraded much more sharply as concurrency increased.

## Batch-4 Stress Results

| Concurrency | ONNX p50 ms | ONNX p95 ms | ONNX throughput items/s | Triton p50 ms | Triton p95 ms | Triton throughput items/s |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 53.11 | 69.46 | 671.50 | 54.32 | 82.21 | 514.95 |
| 16 | 98.67 | 111.81 | 653.36 | 109.95 | 139.92 | 515.31 |
| 32 | 128.57 | 283.74 | 697.18 | 224.73 | 256.05 | 516.42 |

### Observed

- At batch size `4`, Triton narrowed the gap slightly at lower concurrency, but still did not win.
- Triton throughput again plateaued around `~515 items/s`.
- ONNX Runtime remained higher at `~653-697 items/s`.
- At higher concurrency, Triton latency increased substantially.

## Interpretation

These results indicate that in this repo’s current architecture, Triton dynamic batching did not recover enough performance to beat the simpler in-process ONNX Runtime FastAPI path.

The likely reasons are:

- `FastAPI + ONNX Runtime` is already highly efficient
- `FastAPI + Triton(ONNX)` adds:
  - an additional hop between FastAPI and Triton
  - request serialization/deserialization overhead
  - Triton serving overhead
- the current workload characteristics are not strong enough to let Triton’s batching features outweigh those added costs

## What This Does and Does Not Mean

These findings do **not** mean Triton is useless. They mean that for this specific:

- single-model workload
- single-machine setup
- current request pattern
- current batching configuration

the simplest high-performance API path is still:

- `FastAPI + ONNX Runtime`

Triton may still be valuable when you need:

- production inference-server features
- model lifecycle management
- multi-model serving
- stronger batching/scheduling workloads
- larger-scale deployment concerns

## Current Best Serving Choice

Based on the experiments so far, the best-performing text API path in this repo is:

- `FastAPI + ONNX Runtime`

## Methodology

### Services Compared

FastAPI + ONNX Runtime:

- base URL: `http://127.0.0.1:8004`

FastAPI + Triton(ONNX):

- base URL: `http://127.0.0.1:8003`

Triton backend:

- HTTP inference API on `http://127.0.0.1:8000`
- dynamic batching enabled in:
  - [config.pbtxt](/home/user/projects/ai-inference-bench/workloads/bert_classifier/triton/model_repository/bert_classifier/config.pbtxt)

### Traffic Patterns

Single-item:

- endpoint: `/predict`
- batch size: `1`
- total requests: `320`
- warmup requests: `160`
- concurrencies: `8`, `16`, `32`

Batch-4:

- endpoint: `/predict_batch`
- batch size: `4`
- total requests: `320`
- warmup requests: `160`
- concurrencies: `8`, `16`, `32`

### Scenario Input

- `/home/user/projects/ai-inference-bench/benchmark/scenarios/bert_inputs.jsonl`

## Commands Used

### FastAPI + ONNX Stress

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench

for C in 8 16 32; do
  python -m benchmark.runner \
    --base-url http://127.0.0.1:8004 \
    --endpoint /predict \
    --batch-size 1 \
    --num-requests 320 \
    --concurrency "$C" \
    --warmup-requests 160 \
    --serving-mode fastapi_onnx_stress \
    --model-name rubert_topic_classifier_onnx \
    --hardware "NVIDIA GeForce RTX 3080" \
    --notes "FastAPI + ONNX stress concurrency=$C"
done
```

### FastAPI + Triton Stress

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench

for C in 8 16 32; do
  python -m benchmark.runner \
    --base-url http://127.0.0.1:8003 \
    --endpoint /predict \
    --batch-size 1 \
    --num-requests 320 \
    --concurrency "$C" \
    --warmup-requests 160 \
    --serving-mode fastapi_triton_stress \
    --model-name rubert_topic_classifier_triton \
    --hardware "NVIDIA GeForce RTX 3080" \
    --notes "FastAPI + Triton stress concurrency=$C"
done
```

### FastAPI + ONNX Batch-4 Stress

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench

for C in 8 16 32; do
  python -m benchmark.runner \
    --base-url http://127.0.0.1:8004 \
    --endpoint /predict_batch \
    --batch-size 4 \
    --num-requests 320 \
    --concurrency "$C" \
    --warmup-requests 160 \
    --serving-mode fastapi_onnx_batch_stress \
    --model-name rubert_topic_classifier_onnx \
    --hardware "NVIDIA GeForce RTX 3080" \
    --notes "FastAPI + ONNX batch=4 stress concurrency=$C"
done
```

### FastAPI + Triton Batch-4 Stress

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench

for C in 8 16 32; do
  python -m benchmark.runner \
    --base-url http://127.0.0.1:8003 \
    --endpoint /predict_batch \
    --batch-size 4 \
    --num-requests 320 \
    --concurrency "$C" \
    --warmup-requests 160 \
    --serving-mode fastapi_triton_batch_stress \
    --model-name rubert_topic_classifier_triton \
    --hardware "NVIDIA GeForce RTX 3080" \
    --notes "FastAPI + Triton batch=4 stress concurrency=$C"
done
```
