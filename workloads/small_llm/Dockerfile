FROM ai-bench-python-ml-base:0.1.0

WORKDIR /app

COPY workloads /app/workloads

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV AI_BENCH_REPO_ROOT=/app
ENV MODEL_KEY=qwen_1_5b_instruct
ENV SMALL_LLM_SERVING_KEY=baseline_fastapi
ENV SMALL_LLM_LOG_DIR=/outputs/stage_a_baseline/requests
ENV SMALL_LLM_REQUEST_LOG_PATH=/outputs/stage_a_baseline/requests/requests_small_llm.jsonl

RUN mkdir -p /outputs/stage_a_baseline/requests

CMD ["uvicorn", "workloads.small_llm.app.main:app", "--host", "0.0.0.0", "--port", "8010"]
