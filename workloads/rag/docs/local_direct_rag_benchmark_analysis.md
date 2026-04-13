# Local Direct RAG Benchmark Analysis

## Scope

This note summarizes the completed benchmark runs for the local direct/classic RAG workload.

Successful benchmark artifacts analyzed:

- `retrieval`
  - `workloads/rag/benchmarks/results/20260409T135617Z_retrieval_7935fd7a-753c-4cbb-9b2c-d51167b65e67.jsonl`
- `retrieval_rerank`
  - `workloads/rag/benchmarks/results/20260409T135734Z_retrieval_rerank_b12c1ac4-587e-4dad-af02-5d1ea109a242.jsonl`
- `full`
  - `workloads/rag/benchmarks/results/20260409T140158Z_full_022f43c8-c7ea-469e-8ec9-b82395d1ce6c.jsonl`

Older failed runs were excluded from the timing diagnosis.

## Run Counts

- `retrieval`
  - `60` successful query executions
  - `12` unique queries
  - `5` repeats
- `retrieval_rerank`
  - `60` successful query executions
  - `12` unique queries
  - `5` repeats
- `full`
  - `36` successful query executions
  - `12` unique queries
  - `3` repeats

## Stage Timing Summary

### Retrieval

| Stage | Mean ms | p50 ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|
| `bm25_ms` | 1.20 | 1.13 | 2.08 | 2.49 |
| `dense_ms` | 62.88 | 14.77 | 19.53 | 2873.68 |
| `merge_ms` | 0.03 | 0.02 | 0.05 | 0.07 |
| `total_ms` | 64.15 | 15.96 | 21.33 | 2874.49 |

Average share of total latency:

- `dense_ms`: `98.0%`
- `bm25_ms`: `1.9%`
- `merge_ms`: effectively `0%`

### Retrieval + Rerank

| Stage | Mean ms | p50 ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|
| `bm25_ms` | 1.21 | 1.13 | 2.15 | 2.39 |
| `dense_ms` | 60.82 | 14.35 | 21.33 | 2737.75 |
| `merge_ms` | 0.03 | 0.02 | 0.04 | 0.05 |
| `rerank_ms` | 975.91 | 938.84 | 1103.28 | 2462.39 |
| `total_ms` | 1038.01 | 952.85 | 1119.98 | 5201.18 |

Average share of total latency:

- `rerank_ms`: `94.0%`
- `dense_ms`: `5.9%`
- `bm25_ms`: `0.1%`
- `merge_ms`: effectively `0%`

### Full Direct RAG

| Stage | Mean ms | p50 ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|
| `bm25_ms` | 1.15 | 1.15 | 2.00 | 2.08 |
| `dense_ms` | 81.23 | 14.94 | 19.27 | 2385.07 |
| `merge_ms` | 0.03 | 0.03 | 0.05 | 0.08 |
| `rerank_ms` | 988.44 | 953.53 | 1140.75 | 2501.31 |
| `context_build_ms` | 0.04 | 0.04 | 0.07 | 0.09 |
| `llm_ms` | 5908.86 | 6482.51 | 9715.66 | 9948.01 |
| `total_ms` | 6979.81 | 7441.32 | 11105.74 | 12012.51 |

Average share of total latency:

- `llm_ms`: `84.7%`
- `rerank_ms`: `14.2%`
- `dense_ms`: `1.2%`
- `bm25_ms`, `merge_ms`, `context_build_ms`: effectively `0%`

## Bottleneck Diagnosis

### Primary bottleneck

The main bottleneck in full direct RAG is the final local LLM stage.

- mean `llm_ms`: `5908.86`
- p50 `llm_ms`: `6482.51`
- p95 `llm_ms`: `9715.66`

This stage dominates end-to-end latency and is the main reason the full path lands around `7-11` seconds.

### Secondary bottleneck

The reranker is the second major cost.

- mean `rerank_ms`: about `976-988`
- p50 `rerank_ms`: about `939-954`
- p95 `rerank_ms`: about `1103-1141`

It dominates the `retrieval_rerank` mode and remains the second-largest contributor in full mode.

### Cheap stages

The following stages are currently cheap and are not worth optimizing first:

- `bm25_ms`
- `merge_ms`
- `context_build_ms`

These stages are already in the `~0-2 ms` range.

## Variance Notes

Dense retrieval has high variance relative to its normal median.

- typical p50 `dense_ms`: about `14-15 ms`
- worst observed outliers: `2.3-2.8 s`

This suggests dense retrieval is usually fast, but occasionally suffers from large spikes. That points more toward warmup, transient contention, cache effects, or Qdrant/embedding jitter than toward a steady-state throughput issue.

The reranker also shows variance, but it is consistently expensive even outside outliers.

The LLM shows both large absolute latency and meaningful variance, which makes it the most impactful optimization target overall.

## Recommended Optimization Order

1. Optimize or replace the final LLM stage.
   - Highest impact on full-path latency.
   - Best lever for reducing total response time.

2. Optimize the reranker.
   - Strong second-order win.
   - Likely best options are fewer candidates, a smaller reranker, or a faster backend.

3. Investigate dense retrieval variance.
   - Median is already good.
   - Focus should be on eliminating large outliers rather than lowering the steady-state p50.

4. Leave BM25, merge, and context-build alone for now.
   - They are already cheap enough that further tuning will not materially improve end-to-end latency.

## Bottom Line

For the current local direct RAG setup:

- `retrieval` is effectively gated by dense retrieval, but only because everything else is negligible
- `retrieval_rerank` is dominated by the reranker
- `full` is dominated by the final local LLM, with reranking as the next biggest contributor

If the goal is meaningful latency reduction, the optimization priority should be:

`LLM -> reranker -> dense retrieval variance`
