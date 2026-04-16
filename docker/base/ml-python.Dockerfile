FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY docker/base/ml-python-requirements.txt /tmp/ml-python-requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/ml-python-requirements.txt
