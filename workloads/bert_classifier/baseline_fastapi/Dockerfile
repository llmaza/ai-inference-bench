FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app /app/app

ENV PYTHONUNBUFFERED=1
ENV BERT_MODEL_DIR=/models/bert_classifier
ENV BERT_LOG_DIR=/outputs/baseline_fastapi

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
