# Direct RAG Benchmarks

Input format is JSONL with one query per line:

```json
{"query_id":"q1","query":"Какой срок испытательного срока?","notes":"optional"}
```

Benchmark modes:
- `retrieval`
- `retrieval_rerank`
- `full`

Example commands:

```bash
python workloads/rag/benchmarks/benchmark_direct_rag.py --mode retrieval
python workloads/rag/benchmarks/benchmark_direct_rag.py --mode retrieval_rerank
python workloads/rag/benchmarks/benchmark_direct_rag.py --mode full
```

Results are written to `workloads/rag/benchmarks/results/`:
- raw per-query JSONL
- per-run summary JSON
