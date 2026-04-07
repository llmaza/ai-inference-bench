# BERT Serving Architecture Layers

This note explains the serving-layer boundaries used in the BERT benchmarking work.

## 1. FastAPI + ONNX Runtime

Flow:

`Client -> FastAPI -> tokenizer -> ONNX Runtime -> logits -> postprocess -> JSON response`

### Responsibility split

- Client sends raw text
- FastAPI handles:
  - request parsing
  - validation
  - tokenization
  - inference call
  - postprocessing
  - JSON response formatting
- ONNX Runtime runs inside the FastAPI process

### Meaning

This is an application-style serving setup. One Python service owns the full request lifecycle.

## 2. FastAPI + Triton(ONNX)

Flow:

`Client -> FastAPI -> tokenizer -> Triton -> ONNX model -> logits -> FastAPI postprocess -> JSON response`

### Responsibility split

- Client still sends raw text
- FastAPI handles:
  - request parsing
  - validation
  - tokenization
  - sending tensors to Triton
  - postprocessing
  - JSON response formatting
- Triton handles:
  - model execution
  - ONNX backend serving

### Meaning

This keeps the same text API boundary while moving inference execution out of the FastAPI process and into Triton.

## 3. Triton Direct

Flow:

`Client -> tokenizer -> Triton -> ONNX model -> logits`

### Responsibility split

- Client handles:
  - text preprocessing
  - tokenization
  - Triton request formatting
  - optional postprocessing after logits return
- Triton handles:
  - model inference only

### Meaning

This is a lower-level model-serving setup. Triton receives tensors rather than raw text.

## Why These Comparisons Are Different

### Not fully apples-to-apples

`FastAPI + ONNX Runtime` vs `Triton Direct`

- FastAPI path includes text tokenization and API logic in the measured server path
- Triton direct path receives already-tokenized tensors

So this comparison is useful, but it does not use the same API boundary.

### More apples-to-apples

`FastAPI + ONNX Runtime` vs `FastAPI + Triton(ONNX)`

- both accept raw text
- both perform tokenization
- both return the same style of JSON response
- the main difference is the inference backend path

### Same tensor boundary, but not identical overhead

`Direct ONNX Runtime` vs `Triton Direct`

- both operate on tensors
- both exclude application-layer API work
- the main difference is the inference-serving/runtime path
- direct ONNX Runtime is in-process
- Triton Direct still includes server and HTTP overhead

## Practical Interpretation

- `FastAPI + ONNX Runtime` is a full text-serving API.
- `FastAPI + Triton(ONNX)` is a full text-serving API backed by Triton.
- `Triton Direct` is a model-serving interface, not a full application API by itself.
- `Direct ONNX Runtime` is the lowest-overhead in-process tensor path.

## Short Interview Framing

FastAPI is the application/API layer. Triton is the inference-serving layer. A direct Triton benchmark measures tensor-level model serving with server overhead still included, while a FastAPI-plus-Triton design keeps the same text API boundary and replaces only the inference backend.
