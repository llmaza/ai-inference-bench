# BERT Case Study

## Current State

The BERT workload in `ai-inference-bench` currently has three implemented layers:

- baseline FastAPI service using PyTorch
- ONNX export and parity validation
- Triton model repository layout for the exported ONNX model

Detailed benchmark writeups live under:

- [docs/bert/bert_case_study.md](/home/user/projects/ai-inference-bench/docs/bert/bert_case_study.md)

## Baseline Service

The baseline service is a FastAPI application that loads the fine-tuned RuBERT sequence classifier and exposes:

- `GET /health`
- `POST /predict`
- `POST /predict_batch`

That baseline exists in:

- [main.py](/home/user/projects/ai-inference-bench/workloads/bert_classifier/baseline_fastapi/app/main.py)

## ONNX Export

The same classifier has been exported to ONNX and validated against PyTorch for prediction parity.

Relevant files:

- [export_to_onnx.py](/home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/export_to_onnx.py)
- [validate_onnx.py](/home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/validate_onnx.py)
- [model_versions.md](/home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/model_versions.md)

## Triton Deployment Plan

The Triton repository structure is now in place for the exported ONNX model:

- [config.pbtxt](/home/user/projects/ai-inference-bench/workloads/bert_classifier/triton/model_repository/bert_classifier/config.pbtxt)
- [model.onnx](/home/user/projects/ai-inference-bench/workloads/bert_classifier/triton/model_repository/bert_classifier/1/model.onnx)

The current Triton setup is intentionally minimal:

- ONNX Runtime backend
- one model version
- dynamic sequence length preserved
- no Triton dynamic batching yet

That keeps the next Triton phase focused on establishing a clean serving baseline before adding Triton scheduler features such as dynamic batching.
