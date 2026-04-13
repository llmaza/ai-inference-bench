# Reranker ONNX Experiment

## Goal

This experiment converted only the reranker stage to an ONNX path while leaving:

- embeddings unchanged
- dense retrieval unchanged
- final local LLM unchanged

The purpose was to test the fastest practical optimization after the earlier timing study showed:

- `llm_ms` dominates full direct RAG
- `rerank_ms` is the second largest bottleneck

## Backend Options

The direct RAG pipeline now supports two reranker backends:

- `torch`
  - local `transformers` sequence-classification reranker
- `onnx`
  - ONNX-exported reranker executed through ONNX Runtime

The backend is switchable without changing retrieval or LLM code.

## Files Added / Updated

Added:

- `workloads/rag/reranker/backend.py`
- `workloads/rag/reranker/onnx_backend.py`
- `workloads/rag/benchmarks/benchmark_reranker_backend.py`

Updated:

- `workloads/rag/direct/rag_pipeline.py`
- `workloads/rag/direct/rag_ui.py`
- `workloads/rag/benchmarks/benchmark_direct_rag.py`

## Isolated Reranker Benchmark

Artifact:

- `workloads/rag/benchmarks/results/20260409T145151Z_reranker_backend_2a6806f0-53cb-4854-82a7-eb84ad113e64_summary.json`

Setup:

- query set: `queries.labor_eval.jsonl`
- repeats: `3`
- backends: `torch`, `onnx`
- successful executions: `72`

### Results

| Backend | Mean ms | p50 ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|
| `torch` | 1099.263 | 1046.294 | 1261.122 | 2589.911 |
| `onnx` | 860.938 | 798.760 | 957.719 | 2698.828 |

### Interpretation

The ONNX reranker is clearly faster in isolation.

- mean reranker latency improved by about `21.7%`
- p50 reranker latency improved by about `23.7%`
- p95 reranker latency improved by about `24.1%`

So the ONNX path is a real improvement at the stage level.

## End-to-End Full Direct RAG Comparison

Clean full-run artifacts:

- ONNX reranker:
  - `workloads/rag/benchmarks/results/20260409T145524Z_full_1c7df4e9-2fbd-495c-a210-5f63ed82269e_summary.json`
- Torch reranker:
  - `workloads/rag/benchmarks/results/20260409T145717Z_full_1a3f02be-234e-42d8-bc33-f8e67693fdb1_summary.json`

Setup:

- query set: `queries.labor_eval.jsonl`
- repeats: `1`
- `max_context_chars=6000`
- `llm_max_tokens=512`

### Full-Path Results

| Backend | rerank mean ms | rerank p50 ms | total mean ms | total p50 ms | success rate |
|---|---:|---:|---:|---:|---:|
| `torch` | 1207.819 | 1080.130 | 7932.871 | 8151.081 | 1.0 |
| `onnx` | 983.727 | 815.275 | 7604.908 | 7823.620 | 1.0 |

### End-to-End Interpretation

The ONNX reranker also improves the full direct RAG path:

- `rerank_ms`
  - mean improvement: about `18.6%`
  - p50 improvement: about `24.5%`
- `total_ms`
  - mean improvement: about `4.1%`
  - p50 improvement: about `4.0%`

This is exactly what we would expect from the earlier bottleneck study:

- reranking got meaningfully faster
- but total latency improved only modestly because the local LLM still dominates the full path

## Practical Diagnosis

This ONNX reranker experiment was successful:

- the backend is switchable
- isolated reranker latency improved materially
- end-to-end direct RAG latency improved modestly but consistently

However, the overall system is still dominated by `llm_ms`, so the ONNX reranker is a useful second-order optimization, not a first-order breakthrough.

## Recommendation

For the next practical optimization step:

1. keep the ONNX reranker path as the better reranking option
2. preserve the torch backend for fallback / comparison
3. move next to the final local LLM stage, because that is still the main latency budget consumer

## Notes

Some earlier full benchmark attempts failed due GPU memory pressure when multiple heavy runs overlapped. The clean comparison above uses the successful sequential runs only.
