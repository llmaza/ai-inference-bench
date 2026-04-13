# AI Inference Bench

AI Inference Bench is a local benchmarking workspace for comparing inference stacks across multiple AI workloads on the same machine. The repo is organized around workload-specific implementations and shared serving components so we can measure latency, throughput, batching behavior, and system tradeoffs with repeatable prompts and controlled configurations.

## Current Scope

- BERT topic-classification serving benchmarks
- Direct/classic RAG benchmarking
- Shared small-LLM serving and benchmark baselines
- Placeholder structure for OCR and future workloads

## Workloads

- [workloads/bert_classifier](/home/user/projects/ai-inference-bench/workloads/bert_classifier)
  BERT classifier workload with baseline FastAPI, ONNX Runtime FastAPI, Triton direct, and FastAPI-to-Triton paths.
- [workloads/hybrid_bert_regex](/home/user/projects/ai-inference-bench/workloads/hybrid_bert_regex)
  Hybrid BERT plus rule-based workload area.
- [workloads/rag](/home/user/projects/ai-inference-bench/workloads/rag)
  Direct RAG pipeline, local embedder, local reranker, shared small-LLM integration, benchmarks, and docs.
- [workloads/small_llm](/home/user/projects/ai-inference-bench/workloads/small_llm)
  Shared small-LLM service and benchmark paths.
- [workloads/ocr](/home/user/projects/ai-inference-bench/workloads/ocr)
  Placeholder for future OCR integration.

## Benchmarking Areas

- [benchmark](/home/user/projects/ai-inference-bench/benchmark)
  Shared benchmark harnesses and metrics helpers.
- [docs/bert](/home/user/projects/ai-inference-bench/docs/bert)
  BERT benchmark reports and case-study notes.
- [workloads/rag/docs](/home/user/projects/ai-inference-bench/workloads/rag/docs)
  RAG benchmark analysis and experiment summaries.

## Current Findings

- FastAPI + ONNX Runtime is currently the strongest BERT serving path in this repo.
- Triton was benchmarked for BERT, including batching experiments, but did not beat the current ONNX in-process path on this machine.
- The direct RAG workload is instrumented by stage:
  - retrieval
  - retrieval + rerank
  - full direct RAG
- Current RAG bottlenecks are:
  - local LLM generation first
  - reranker second
  - dense retrieval variance is worth tracking, but it is not the first steady-state optimization target
- Shared `small_llm` stages are currently framed as:
  - Stage A: local FastAPI baseline
  - Stage B: TensorRT-LLM direct
  - Stage C: Triton + TensorRT-LLM

## Quick Start

Create and activate the project environment:

```bash
cd /home/user/projects/ai-inference-bench
python3 -m venv aienv
source aienv/bin/activate
python -m pip install -r requirements.txt
```

Run the BERT ONNX FastAPI service:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
python -m uvicorn workloads.bert_classifier.onnx_fastapi.app.main:app --host 0.0.0.0 --port 8004
```

Run the shared small-LLM FastAPI baseline:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
MODEL_KEY=qwen_1_5b_instruct SMALL_LLM_SERVING_KEY=baseline_fastapi \
python -m uvicorn workloads.small_llm.app.main:app --host 0.0.0.0 --port 8010
```

Benchmark the shared small-LLM baseline:

```bash
cd /home/user/projects/ai-inference-bench
source aienv/bin/activate
python workloads/small_llm/benchmarks/run_baseline.py \
  --base-url http://127.0.0.1:8010 \
  --prompts workloads/small_llm/benchmarks/prompts.labor_sample.jsonl \
  --repeats 3 \
  --concurrency 1 \
  --timeout-sec 180
```

## Important Notes

- Local model downloads should not be committed to the repo. Root-level Hugging Face artifacts are ignored in `.gitignore`.
- Some workloads depend on Docker, NVIDIA GPU support, Triton, Qdrant, or TensorRT-LLM environments that need separate setup.
- TensorRT-LLM direct setup exists in the repo, but actual execution depends on the container/runtime environment being prepared correctly.

## Suggested Read Order

- [workloads/small_llm/README.md](/home/user/projects/ai-inference-bench/workloads/small_llm/README.md)
- [workloads/rag/docs/local_direct_rag_benchmark_analysis.md](/home/user/projects/ai-inference-bench/workloads/rag/docs/local_direct_rag_benchmark_analysis.md)
- [docs/bert/bert_case_study.md](/home/user/projects/ai-inference-bench/docs/bert/bert_case_study.md)

## License

See [LICENSE](/home/user/projects/ai-inference-bench/LICENSE).
