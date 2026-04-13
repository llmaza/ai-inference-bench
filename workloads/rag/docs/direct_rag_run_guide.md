# Direct RAG Run Guide

This guide covers the migrated direct/classic RAG path under `workloads/rag/`.

## Main entrypoints

- Ingestion / document structuring:
  - `workloads/rag/direct/pdf_to_structured_labor_code.py`
- Chunking:
  - `workloads/rag/direct/chunk_labor_code_for_rag.py`
- Vector indexing:
  - `workloads/rag/embed/embed_chunks_to_qdrant.py`
- Direct question answering:
  - `workloads/rag/direct/rag_pipeline.py`
- Optional UI:
  - `workloads/rag/direct/rag_ui.py`

## Required env vars

Create `workloads/rag/.env` from `workloads/rag/.env.example`.

Required:
- `QDRANT_URL`

Optional:
- `QDRANT_API_KEY`
- `HF_RERANKER_MODEL`
- `HF_LLM_MODEL`
- `HF_CHAT_BASE_URL`
- `EMBEDDING_MODEL`
- `EMBEDDING_VECTOR_SIZE`
- `LOCAL_EMBED_DEVICE`
- `LOCAL_EMBED_BATCH_SIZE`
- `LOCAL_RERANK_DEVICE`
- `LOCAL_LLM_DEVICE`
- `QDRANT_COLLECTION`

Default collection if unset:
- `labor_code_tk_e5`

## Required Python packages

The repo root `requirements.txt` does not yet include all RAG dependencies.
Install these in the active environment before running the migrated direct path:

- `huggingface_hub`
- `qdrant_client`
- `rank_bm25`
- `httpx`
- `numpy`
- `python-dotenv`
- `tiktoken`
- `PyMuPDF`
- `streamlit` for the optional UI only

The local embedding backend uses repo-standard model libraries already present in the bench stack:
- `torch`
- `transformers`

The local reranker backend uses the same local model stack:
- `torch`
- `transformers`

The shared small LLM backend under `workloads/small_llm/llm_inference.py` also uses:
- `torch`
- `transformers`

## Command sequence

From the repo root:

```bash
cd /home/user/projects/ai-inference-bench
```

1. Prepare `workloads/rag/.env`

```bash
cp workloads/rag/.env.example workloads/rag/.env
```

2. Optional: rebuild structured labor-code files from the PDF

```bash
python workloads/rag/direct/pdf_to_structured_labor_code.py
```

Expected outputs in `workloads/rag/direct/assets/`:
- `Трудовой_Кодекс_структурированный.md`
- `Трудовой_Кодекс_структурированный.jsonl`
- `Трудовой_Кодекс_структурированный.json`

3. Optional: rebuild chunks from the structured JSON

```bash
python workloads/rag/direct/chunk_labor_code_for_rag.py
```

Expected output:
- `workloads/rag/direct/assets/Трудовой_Кодекс_chunks.jsonl`

4. Create or refresh the Qdrant vector collection

```bash
python workloads/rag/embed/embed_chunks_to_qdrant.py --collection labor_code_tk_e5
```

Useful options:

```bash
python workloads/rag/embed/embed_chunks_to_qdrant.py --dry-run
python workloads/rag/embed/embed_chunks_to_qdrant.py --recreate --collection labor_code_tk_e5
```

Expected result:
- Qdrant collection populated with chunk vectors and payloads
- script prints local embedding timing:
  - `batch_embed_ms`
  - `avg_embed_ms`
  - `total_embed_ms`

5. Ask a direct RAG question from CLI

```bash
python workloads/rag/direct/rag_pipeline.py "Какой срок испытательного срока?"
```

Useful options:

```bash
python workloads/rag/direct/rag_pipeline.py \
  "Какой срок испытательного срока?" \
  --verbose \
  --max-context-chars 24000
```

Expected output:
- retrieved + reranked context used internally
- final single-answer LLM response printed to stdout

6. Optional: run the Streamlit UI

```bash
streamlit run workloads/rag/direct/rag_ui.py
```

## Notes on behavior

- Retrieval remains hybrid:
  - BM25
  - dense Qdrant retrieval using local E5 query embeddings
- Reranking is now local
- Final answer generation is now local
- The local answer stage now imports the shared `workloads/small_llm` component
- No agentic flow was added
- Embeddings are local
- Embeddings only are now local:
  - indexed text keeps the `passage: ...` E5 convention
  - search queries keep the `query: ...` E5 convention
- Reranker is local:
  - candidate scoring stays in the same rerank stage
  - `rerank_ms` in benchmark logs remains the timing to compare later
- Final LLM is local and shared:
  - one answer-generation call per request
  - `llm_ms` in benchmark logs remains the latency to compare later
  - default local model: `Qwen/Qwen2.5-1.5B-Instruct`

## Better benchmarking entrypoint

For benchmarking and automation, prefer:
- `workloads/rag/direct/rag_pipeline.py`

It is simpler and more deterministic than the Streamlit UI and is the best direct programmatic entrypoint for scripted evaluation.
