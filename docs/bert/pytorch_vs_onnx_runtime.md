# BERT Case Study

## Runtime Comparison

### CPU Comparison

| Batch | PyTorch CPU mean ms | ONNX CPU mean ms | ONNX advantage |
|---|---:|---:|---:|
| 1 | 58.81 | 42.33 | ~1.39x faster |
| 4 | 214.66 | 209.38 | ~1.03x faster |
| 8 | 496.57 | 608.53 | PyTorch faster |
| 16 | 1059.49 | 1513.79 | PyTorch faster |

Observed:
- At `batch=1`, ONNX CPU is clearly faster. This fits the idea that ONNX Runtime reduces some inference-engine overhead and executes the static graph more efficiently than PyTorch on CPU.
- At `batch=4`, the two are close, so ONNX’s advantage mostly disappears.
- At `batch=8` and `16`, PyTorch CPU becomes faster. On CPU, larger batches do not give the same kind of scaling benefit you usually expect on GPU. The CPU becomes compute-bound, and ONNX is not automatically better for large batched workloads.
- So the CPU story is: ONNX helps most for small single-request inference, but it is not universally faster for large CPU batches.

### CUDA Comparison

| Batch | PyTorch CUDA mean ms | ONNX CUDA mean ms | ONNX advantage |
|---|---:|---:|---:|
| 1 | 5.83 | 2.73 | ~2.14x faster |
| 4 | 9.37 | 7.57 | ~1.24x faster |
| 8 | 18.57 | 15.22 | ~1.22x faster |
| 16 | 37.32 | 32.50 | ~1.15x faster |

Observed:
- ONNX CUDA is faster at every tested batch size.
- The biggest gain is at `batch=1`. That suggests ONNX Runtime is reducing inference overhead and running the graph more efficiently for small online-style requests.
- As batch size increases, ONNX still wins, but the gap gets smaller. That is normal because batching improves hardware utilization for both backends, so the relative engine advantage narrows.
- GPU is also where batching makes the most sense. Unlike CPU, the GPU benefits much more from larger tensor workloads, so both runtimes scale better there.
- So the CUDA story is: ONNX gives a real inference-engine improvement, especially for small request-per-request serving, while still staying ahead at larger batch sizes.

## Overall Observation

- CPU: ONNX can help for small inference calls, but large CPU batching is not always favorable.
- CUDA: ONNX consistently improves execution, and batching helps both backends, though ONNX still stays ahead.

These measurements are backend-only comparisons. They compare direct model execution in PyTorch Runtime versus ONNX Runtime and do not yet include FastAPI serving overhead. This makes them useful for isolating model-engine performance before moving into end-to-end service comparisons such as `FastAPI + PyTorch` versus `FastAPI + ONNX Runtime`.
