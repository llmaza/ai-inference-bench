# Triton BERT Model Repository

This directory contains the Triton-serving layout for the exported BERT ONNX classifier.

## Model Layout

- `model_repository/bert_classifier/config.pbtxt`
- `model_repository/bert_classifier/1/model.onnx`

## Current Phase A1 Assumptions

- backend: ONNX Runtime
- model format: exported ONNX sequence-classification model
- tokenizer remains outside Triton for now
- manual API batching has already been measured in FastAPI
- Triton dynamic batching has not been enabled yet

## Why Dynamic Batching Is Not Enabled Yet

This repository is intentionally configured conservatively so the first Triton step isolates serving-stack differences before introducing scheduler-level optimizations. Once the base Triton path is validated, `dynamic_batching` can be added and benchmarked explicitly against the existing FastAPI batch results.
