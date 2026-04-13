# RAG Workloads

This directory contains the conservative direct/classic RAG workload migrated from `indrive_v1`.

Current scope:
- direct retrieval + rerank + context build + single LLM answer
- shared small-LLM inference reused from `workloads/small_llm`
- main programmatic entrypoint: `workloads/rag/direct/rag_pipeline.py`
- optional UI entrypoint: `workloads/rag/direct/rag_ui.py`

Not included yet:
- agentic path
- broader RAG restructuring
