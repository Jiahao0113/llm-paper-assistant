# Roadmap

| Step | Topic | Deliverable | Done |
|---|---|---|---|
| 0 | LLM basics: tokenizer, sampling, API | `step0_llm_basics/` | [x] |
| 1 | Prompting, structured output, function calling | `step1_prompting/` | [x] |
| 2 | Embeddings + vector search, chunking | `src/copilot/ingest.py`, `retrieval.py` | [x] |
| 3 | Full RAG with citations | `src/copilot/rag.py` (v1) | [x] |
| 4 | Evaluation: test set, recall@k, faithfulness | `eval/` (retrieval baseline done; generation eval TODO) | [~] |
| 5 | Hybrid (BM25 + dense), reranker, query rewriting | `bm25.py`, `rerank.py` (query rewriting TODO) | [~] |
| 6 | Agent: hand-written ReAct loop, tool use | v3 | [ ] |
| 7 | Agentic RAG, MCP, tracing | v3+ | [ ] |
| 8 | FastAPI + Docker + README with results | deploy | [ ] |
