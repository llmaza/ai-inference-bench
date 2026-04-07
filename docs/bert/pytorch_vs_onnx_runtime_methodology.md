# PyTorch vs ONNX Runtime Methodology

## Purpose

This document describes the methodology used to compare direct BERT inference performance between PyTorch Runtime and ONNX Runtime.

This comparison is backend-only. It isolates model execution and does not include:
- FastAPI request handling
- HTTP/network overhead
- JSON serialization
- end-to-end service latency

The goal is to understand whether ONNX Runtime executes the same BERT classifier more efficiently than PyTorch under the same input workload.

## Scope of Comparison

The benchmark compares these backends:
- PyTorch CPU
- PyTorch CUDA
- ONNX Runtime CPU
- ONNX Runtime CUDA

## Model and Artifacts

- Repo root:
  - `/home/user/projects/ai-inference-bench`
- Source BERT model used for Phase A1:
  - `/home/user/projects/ai-inference-bench/workloads/bert_classifier/artifacts/model`
- Exported ONNX model:
  - `/home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/bert_classifier.onnx`
- Benchmark scenario file:
  - `/home/user/projects/ai-inference-bench/benchmark/scenarios/bert_inputs.jsonl`

## Environment

The project uses a repo-local environment path:
- `/home/user/projects/ai-inference-bench/aienv`

At the moment, `aienv` points to the existing working environment that already contains:
- `torch`
- `transformers`
- `onnx`
- `onnxruntime`
- `onnxruntime-gpu`

Activate it with:

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
```

## ONNX Export and Validation

Before running the runtime comparison, the PyTorch model was exported to ONNX and then checked for parity.

### Export Command

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench
python workloads/bert_classifier/onnx_export/export_to_onnx.py \
  --model-dir /home/user/projects/ai-inference-bench/workloads/bert_classifier/artifacts/model \
  --output-path /home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/bert_classifier.onnx \
  --max-length 256 \
  --opset 17
```

### Validation Command

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench
python workloads/bert_classifier/onnx_export/validate_onnx.py \
  --model-dir /home/user/projects/ai-inference-bench/workloads/bert_classifier/artifacts/model \
  --onnx-path /home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/bert_classifier.onnx \
  --scenario-path /home/user/projects/ai-inference-bench/benchmark/scenarios/bert_inputs.jsonl \
  --max-length 256 \
  --num-samples 8
```

### Validation Outcome

The ONNX export was accepted as valid after parity checks showed:
- class parity: `8/8`
- class parity rate: `1.0`
- max absolute difference: `1.430511474609375e-06`

This indicates that the ONNX model preserved prediction behavior for the tested samples and that numerical drift was negligible.

## Benchmark Inputs

- Input dataset:
  - `benchmark/scenarios/bert_inputs.jsonl`
- Number of texts:
  - `100`
- Maximum sequence length:
  - `256`

## Batch Sizes

The runtime benchmark was executed with these batch sizes:
- `1`
- `4`
- `8`
- `16`

These batch sizes were chosen to show how each runtime behaves for:
- single-request inference
- small manual batches
- larger batched execution

## Warmup Strategy

Warmup was applied before timed measurement:
- warmup batches per backend and batch size: `4`

Warmup was used to reduce distortion from:
- initial graph/runtime setup
- CUDA kernel warmup
- first-run memory allocation effects

## Benchmark Command

The runtime comparison was run with:

```bash
source /home/user/projects/ai-inference-bench/aienv/bin/activate
cd /home/user/projects/ai-inference-bench
python -m benchmark.compare_backends \
  --model-dir /home/user/projects/ai-inference-bench/workloads/bert_classifier/artifacts/model \
  --onnx-path /home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/bert_classifier.onnx \
  --dataset-path /home/user/projects/ai-inference-bench/benchmark/scenarios/bert_inputs.jsonl \
  --num-texts 100 \
  --batch-sizes 1 4 8 16 \
  --warmup-batches 4 \
  --backends pytorch_cpu pytorch_cuda onnx_cpu onnx_cuda
```

## Metrics Reported

For each backend and batch size, the benchmark reports:
- mean latency
- p50 latency
- p95 latency
- p99 latency
- minimum latency
- maximum latency
- throughput in items per second

## Output Files

The benchmark writes:

- raw comparison rows:
  - `results/raw/backend_comparison_<run_id>.jsonl`
- aggregate comparison table:
  - `results/tables/backend_comparison.csv`

## Interpretation Notes

- These measurements compare runtime engines, not API stacks.
- They are useful for determining whether ONNX Runtime is faster than PyTorch for the same model and inputs.
- They should not be interpreted as end-to-end FastAPI latency measurements.
- A later service-level comparison such as `FastAPI + PyTorch` versus `FastAPI + ONNX Runtime` is still needed to quantify end-to-end serving gains.

## Current Conclusion

At this stage, the methodology supports one specific conclusion:
- ONNX Runtime can be evaluated independently of HTTP serving overhead
- this allows the project to separate model-engine effects from service-framework effects

That separation is useful because it makes later benchmarking phases easier to explain:
- runtime-only comparison first
- service-level comparison next
- Triton and dynamic batching after that
