# BERT ONNX Export Notes

## Source Model

- Source model path:
  - `/home/user/projects/ai-inference-bench/workloads/bert_classifier/artifacts/model`
- Model type:
  - local Hugging Face `AutoModelForSequenceClassification`
- Tokenizer:
  - local Hugging Face `AutoTokenizer` loaded from the same model directory

## Export Assumptions

- Max sequence length:
  - `256` by default
- Padding mode:
  - `padding="max_length"` during export and validation
- Output:
  - classifier `logits`
- Shape strategy:
  - dynamic batch and dynamic sequence axes by default
- Fixed-shape option:
  - available via `--fixed-shape`
- Default ONNX opset:
  - `17`
- Exporter mode:
  - legacy exporter by default for compatibility
  - dynamo exporter available via `--use-dynamo-exporter`

## Export Command

Run from the repo root with the existing BERT environment activated:

```bash
source /home/user/projects/ai-inference-bench/.venv/bin/activate
cd /home/user/projects/ai-inference-bench
python workloads/bert_classifier/onnx_export/export_to_onnx.py \
  --model-dir /home/user/projects/ai-inference-bench/workloads/bert_classifier/artifacts/model \
  --output-path /home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/bert_classifier.onnx \
  --max-length 256 \
  --opset 17
```

## Validation Command

```bash
source /home/user/projects/ai-inference-bench/.venv/bin/activate
cd /home/user/projects/ai-inference-bench
python workloads/bert_classifier/onnx_export/validate_onnx.py \
  --model-dir /home/user/projects/ai-inference-bench/workloads/bert_classifier/artifacts/model \
  --onnx-path /home/user/projects/ai-inference-bench/workloads/bert_classifier/onnx_export/bert_classifier.onnx \
  --scenario-path /home/user/projects/ai-inference-bench/benchmark/scenarios/bert_inputs.jsonl \
  --max-length 256 \
  --num-samples 8
```

## Validation Outputs

The validation script reports:
- class parity count
- class parity rate
- maximum absolute difference between PyTorch and ONNX logits

## Environment Note

The repo-local environment should include:

```bash
pip install -r /home/user/projects/ai-inference-bench/requirements.txt
```

If you explicitly choose the newer dynamo exporter, you may also need:

```bash
pip install onnxscript
```
