# FastAPI Batch Sweep: PyTorch vs ONNX Runtime

## Summary

This report compares batched FastAPI serving for the same RuBERT topic classifier using two execution backends:

- `FastAPI + PyTorch`
- `FastAPI + ONNX Runtime`

Both services exposed the same batched endpoint:

- `POST /predict_batch`

The benchmark swept API batch sizes:

- `1`
- `2`
- `4`
- `8`
- `16`

The goal was to measure how end-to-end API latency and logical item throughput change as batch size increases, and to compare how PyTorch and ONNX behave under the same serving conditions.

## PyTorch FastAPI Batch Sweep

| Batch | p50 ms | p95 ms | Throughput items/s | HTTP calls |
|---|---:|---:|---:|---:|
| 1 | 22.52 | 30.08 | 169.45 | 160 |
| 2 | 31.11 | 39.75 | 253.57 | 80 |
| 4 | 48.50 | 65.91 | 323.85 | 40 |
| 8 | 91.61 | 122.53 | 333.26 | 20 |
| 16 | 189.02 | 218.59 | 315.62 | 10 |

### Observed

- Throughput improves significantly from batch `1` to `4`.
- PyTorch reaches its best throughput around batch `8` in this setup.
- At batch `16`, throughput drops slightly while latency continues to rise.
- End-to-end latency increases steadily with batch size, which is expected because each HTTP call now waits for more items to be processed together.

### Interpretation

For `FastAPI + PyTorch`, batching clearly helps throughput, but only up to a point. In this workload, the best throughput zone is around batch `4-8`. Larger batches continue to reduce HTTP call count, but they also increase service time enough that the overall gain begins to flatten or reverse.

## ONNX FastAPI Batch Sweep

| Batch | p50 ms | p95 ms | Throughput items/s | HTTP calls |
|---|---:|---:|---:|---:|
| 1 | 8.50 | 13.47 | 433.75 | 160 |
| 2 | 11.84 | 23.36 | 576.68 | 80 |
| 4 | 29.80 | 42.70 | 563.76 | 40 |
| 8 | 53.68 | 77.92 | 530.34 | 20 |
| 16 | 138.93 | 160.97 | 440.82 | 10 |

### Observed

- ONNX is substantially faster than PyTorch at every tested batch size.
- ONNX reaches peak throughput at batch `2`, with batch `4` very close behind.
- After batch `4`, throughput declines while latency rises sharply.

### Interpretation

For `FastAPI + ONNX Runtime`, smaller API batches already extract most of the available benefit. In this setup, the best practical range is batch `2-4`. Larger batches still reduce HTTP overhead, but the added service time outweighs that benefit.

## Direct Comparison

| Batch | PyTorch throughput | ONNX throughput | ONNX gain |
|---|---:|---:|---:|
| 1 | 169.45 | 433.75 | ~2.56x |
| 2 | 253.57 | 576.68 | ~2.27x |
| 4 | 323.85 | 563.76 | ~1.74x |
| 8 | 333.26 | 530.34 | ~1.59x |
| 16 | 315.62 | 440.82 | ~1.40x |

## Key Findings

- ONNX Runtime outperformed PyTorch across the full FastAPI batch sweep.
- The strongest relative ONNX advantage appeared at small batch sizes, especially `1-2`.
- As batch size increased, both backends benefited from amortized HTTP and inference overhead, so the relative ONNX advantage narrowed.
- Best observed operating ranges in this experiment:
  - `FastAPI + PyTorch`: batch `4-8`
  - `FastAPI + ONNX Runtime`: batch `2-4`

## Why This Pattern Makes Sense

At small batch sizes, ONNX Runtime benefits from lower inference overhead and more efficient graph execution, so the advantage over PyTorch is largest. As batch size grows, both backends amortize request overhead and make better use of the GPU, which reduces the relative gap. Eventually, larger API batches increase end-to-end service time enough that throughput stops improving and latency becomes much worse.

This is why the best throughput point is not always the largest batch size. Real serving systems often have an optimal middle range where batching is large enough to improve utilization, but not so large that response time balloons.

## Methodology

### Services

PyTorch batch service:

- `/home/user/projects/ai-inference-bench/workloads/bert_classifier/baseline_fastapi`

ONNX batch service:

- `/home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_fastapi`

### Endpoint

Both experiments used:

- `POST /predict_batch`

Request shape:

```json
{
  "messages": ["text 1", "text 2", "text 3"]
}
```

### Benchmark Settings

- total logical requests: `160`
- warmup logical requests: `160`
- concurrency: `4`
- batch sizes: `1, 2, 4, 8, 16`
- hardware: `NVIDIA GeForce RTX 3080`

### Important Metric Notes

- `total_requests` is the number of logical items evaluated.
- `total_http_requests` is the number of actual batch calls sent to the API.
- `throughput_rps` in these summaries is effectively logical items per second.
- As batch size grows, `total_http_requests` falls, but latency per batch call rises.

### Scenario Input File

- `/home/user/projects/ai-inference-bench/benchmark/scenarios/bert_inputs.jsonl`

### Result Files

Raw per-run outputs:

- `/home/user/projects/ai-inference-bench/results/raw/99c4ba86-c4c3-44d3-b9c7-f21a80b00e1f_requests.jsonl`
- `/home/user/projects/ai-inference-bench/results/raw/47d7ade1-d92e-4c46-9370-f74b97c077b8_requests.jsonl`
- `/home/user/projects/ai-inference-bench/results/raw/b91af80f-ad49-4d6b-a1d6-5498c853a83b_requests.jsonl`
- `/home/user/projects/ai-inference-bench/results/raw/fe87b04f-7436-43ff-9bd3-b40527918007_requests.jsonl`
- `/home/user/projects/ai-inference-bench/results/raw/9add4529-7e9b-4bd6-b205-e80dec1f2614_requests.jsonl`
- `/home/user/projects/ai-inference-bench/results/raw/bca55055-3e63-4fc3-a8be-fffec818d0b2_requests.jsonl`
- `/home/user/projects/ai-inference-bench/results/raw/fbc506b2-f5b3-4977-a6da-03248998e22b_requests.jsonl`
- `/home/user/projects/ai-inference-bench/results/raw/6f72fdf4-8b18-4f51-8bcf-f26dc322993e_requests.jsonl`
- `/home/user/projects/ai-inference-bench/results/raw/f499ff9f-9e41-4762-9659-c230b3e30f08_requests.jsonl`
- `/home/user/projects/ai-inference-bench/results/raw/ef467afc-ae84-42fc-bb12-3bb31898c6d6_requests.jsonl`

Summary table:

- `/home/user/projects/ai-inference-bench/results/tables/benchmark_runs.csv`

## Commands

### Environment

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench
```

### Start PyTorch Batch API

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench/workloads/bert_classifier/baseline_fastapi
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Start ONNX Batch API

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_fastapi
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
```

### Run PyTorch Batch Sweep

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench

for B in 1 2 4 8 16; do
  python -m benchmark.runner \
    --base-url http://127.0.0.1:8001 \
    --endpoint /predict_batch \
    --batch-size "$B" \
    --num-requests 160 \
    --concurrency 4 \
    --warmup-requests 160 \
    --serving-mode fastapi_pytorch_batch \
    --model-name rubert_topic_classifier_pytorch \
    --hardware "NVIDIA GeForce RTX 3080" \
    --notes "FastAPI + PyTorch batch=$B"
done
```

### Run ONNX Batch Sweep

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench

for B in 1 2 4 8 16; do
  python -m benchmark.runner \
    --base-url http://127.0.0.1:8002 \
    --endpoint /predict_batch \
    --batch-size "$B" \
    --num-requests 160 \
    --concurrency 4 \
    --warmup-requests 160 \
    --serving-mode fastapi_onnx_batch \
    --model-name rubert_topic_classifier_onnx \
    --hardware "NVIDIA GeForce RTX 3080" \
    --notes "FastAPI + ONNX batch=$B"
done
```
