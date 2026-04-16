FROM ai-bench-python-ml-base:0.1.0

WORKDIR /app

COPY app /app/app

ENV PYTHONUNBUFFERED=1
ENV BERT_MODEL_DIR=/models/bert_classifier
ENV BERT_ONNX_PATH=/models/bert_classifier_onnx/bert_classifier.onnx
ENV BERT_LOG_DIR=/outputs/onnx_fastapi

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
