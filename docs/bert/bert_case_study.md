# BERT Case Study

## Scope

This section collects the Phase A1 benchmark work for the RuBERT topic-classification workload in `ai-inference-bench`.

Current implemented scope:

- baseline FastAPI service with PyTorch
- ONNX export and parity validation
- ONNX-backed FastAPI service
- Triton-backed FastAPI service
- Triton direct client and no-batching benchmark
- direct runtime benchmarking
- end-to-end FastAPI benchmarking
- batched FastAPI benchmarking
- serving architecture note

Triton comparison is not covered in this section yet.

## Reading Order

### 1. PyTorch vs ONNX Runtime

This compares the model backends directly, without FastAPI or HTTP overhead.

- report: [pytorch_vs_onnx_runtime.md](/home/user/projects/ai-inference-bench/docs/bert/pytorch_vs_onnx_runtime.md)
- methodology: [pytorch_vs_onnx_runtime_methodology.md](/home/user/projects/ai-inference-bench/docs/bert/pytorch_vs_onnx_runtime_methodology.md)

Use this when you want to isolate inference-engine behavior and understand whether ONNX Runtime is faster than PyTorch for the same model.

### 2. FastAPI + PyTorch vs FastAPI + ONNX Runtime

This compares end-to-end API serving for single-message requests.

- report: [fastapi_pytorch_vs_fastapi_onnx.md](/home/user/projects/ai-inference-bench/docs/bert/fastapi_pytorch_vs_fastapi_onnx.md)

Use this when you want to see whether the ONNX advantage survives real HTTP serving overhead.

### 3. FastAPI Batch Sweep

This compares batched API serving across batch sizes `1, 2, 4, 8, 16`.

- report: [fastapi_batch_sweep_pytorch_vs_onnx.md](/home/user/projects/ai-inference-bench/docs/bert/fastapi_batch_sweep_pytorch_vs_onnx.md)

Use this when you want to understand how API-layer batching changes latency, throughput, and the relative advantage of ONNX over PyTorch.

## Current Findings

- ONNX Runtime CUDA outperformed PyTorch CUDA in direct backend benchmarking.
- The ONNX export passed parity validation against PyTorch with perfect class parity on the sampled test set.
- In end-to-end FastAPI serving, ONNX Runtime substantially outperformed PyTorch for single-message requests.
- In batched FastAPI serving, ONNX remained faster than PyTorch across the full batch sweep.
- Triton direct no-batching benchmark is implemented and logged as a tensor-level serving comparison.
- Best observed operating ranges in the current API batch experiments:
  - PyTorch FastAPI: batch `4-8`
  - ONNX FastAPI: batch `2-4`

## Suggested Reading Flow

If you are new to this case study, read in this order:

1. [pytorch_vs_onnx_runtime.md](/home/user/projects/ai-inference-bench/docs/bert/pytorch_vs_onnx_runtime.md)
2. [fastapi_pytorch_vs_fastapi_onnx.md](/home/user/projects/ai-inference-bench/docs/bert/fastapi_pytorch_vs_fastapi_onnx.md)
3. [fastapi_batch_sweep_pytorch_vs_onnx.md](/home/user/projects/ai-inference-bench/docs/bert/fastapi_batch_sweep_pytorch_vs_onnx.md)

That order moves from:

- backend-only comparison
- to end-to-end serving comparison
- to API-level batching behavior

## Next Step

The next logical extension of this case study is a `FastAPI + Triton(ONNX)` adapter service so `FastAPI + ONNX Runtime` can be compared against a text-API-equivalent Triton-backed service. After that baseline is established, Triton dynamic batching can be enabled and measured separately.
