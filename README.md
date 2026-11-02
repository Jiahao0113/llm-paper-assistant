# LLM Paper Assistant

A question-answering system over five arXiv papers (Transformer, RAG, LoRA, QLoRA, ReAct). It grows from plain RAG to an agentic RAG loop, and every step is measured with an evaluation harness. Answers carry citations, and the system refuses when the papers do not contain the answer.

The point of the project is the measurement as much as the system: what each component buys, and where the evaluation itself is weak.

## Architecture

```
PDFs -> ingest (structure-aware chunks, 300 tok, 50 overlap) -> 348 chunks
                                  |
question -> BM25 (top 100) + dense MiniLM (top 100) -> RRF fusion -> top 30
         -> cross-encoder rerank -> top 5 -> LLM (Claude Haiku 5.5) -> answer + citations / refusal

Agentic version: the LLM calls search_papers(query, paper?) in a loop (max 6 steps),
reads the results, may rewrite the query and search again, then calls final_answer.
```

| Part | Code |
|---|---|
| PDF parsing and chunking (section metadata, appendix handling, table residue filter) | `src/copilot/ingest.py` |
| Dense retrieval (all-MiniLM-L6-v2, normalized vectors, brute-force cosine) | `src/copilot/retrieval.py` |
| BM25 (hand-written, k1=1.2, b=0.75) and RRF | `src/copilot/bm25.py` |
| Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) | `src/copilot/rerank.py` |
| Hybrid retrieval pipeline | `src/copilot/pipeline.py` |
| Grounded RAG with structured output | `src/copilot/rag.py` |
| Agent (hand-written tool-use loop, strict `final_answer` schema) | `src/copilot/agent.py` |
| Evaluation scripts | `eval/` |

## Results

All numbers are small-sample. Read the limitations before quoting them.

**Retrieval** (recall@5 on 60 synthetic questions; `eval/`)

| Retriever | recall@5 | MRR |
|---|---|---|
| Dense | 0.783 | 0.688 |
| BM25 | 0.933 | 0.839 |
| Hybrid (RRF) | 0.933 | 0.789 |

The synthetic questions reuse wording from the gold chunk, which favours BM25. On 15 hand-written paraphrased questions the order flips and dense does better, so no retriever is declared the winner. Hybrid is the hedge.

**Reranking** (hybrid top 30 -> cross-encoder)

| Test set | recall@1 before -> after | MRR before -> after |
|---|---|---|
| Synthetic (n=60) | 0.683 -> 0.767 | 0.789 -> 0.855 |
| Hand-written (n=15) | 0.467 -> 0.600 | 0.592 -> 0.717 |

**Generation** (15 answerable + 5 unanswerable hand-written questions, Claude Sonnet 5.5 as judge, different from the Haiku generator)

| | Correct (answerable) | Wrongly refused | Unanswerable refused | Searches | Input tokens |
|---|---|---|---|---|---|
| Fixed RAG | 13/15 | 1-2 | 5/5 | 1 | ~1.5-2k |
| Agent | 15/15 | 0 | 5/5 | ~2 | ~5.5-6k |

The agent answers more questions correctly at roughly three times the token cost. The model is not deterministic, so numbers vary between runs; a one-question difference on n=20 is noise.

**Ablation: does the "rephrase the query" prompt sentence matter?** I predicted it would. Removing it changed almost nothing (1.90 vs 1.95 searches per question, same gold recall), so query rewriting comes mostly from the model and the search results, not from that sentence. One run per condition; this means "no effect detected", not "no effect". See `eval/ablate_rewrite.py`.

## Limitations

- **Small, biased test sets.** 60 synthetic questions with lexical leakage and 15 hand-written ones written by one person. More questions would not remove that bias.
- **The agent was developed against the same 20 questions it is evaluated on.** There is no held-out set, so its numbers are optimistic.
- **The agent-versus-RAG comparison is confounded.** The agent sees more chunks (about 10 across two searches) than fixed RAG sees (5). It also benefits from the model's prior knowledge of these famous papers; private documents may behave differently.
- **Citation accuracy is not measured.** The judge sees passages numbered `[1]..[5]`, while the agent cites chunk ids like `c278`, so the judge cannot check the citations. It flagged this in its reasoning.
- **The judge is imperfect.** I read its reasoning and found real bugs in the evaluation (truncated passages, missing metadata), which I fixed; I did not validate it against a second human labeller.
- **One over-generalisation seen:** on an ambiguous question ("What optimizer ... was used to train the model?") the agent answered for three papers and described one loosely; the judge marked it unfaithful.
- **PDF parsing is a baseline.** Formulas, tables and some figure text (ReAct trajectories) are garbled or dropped.
- **No latency measurement**, no tracing, and the index is small enough that brute-force search is fine; it would not scale as written.

## Run it

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and an Anthropic API key in a `.env` file (`ANTHROPIC_API_KEY=...`, gitignored).

```bash
uv sync
# place the five PDFs in data/raw/ (arXiv ids: 1706.03762, 2005.11401, 2106.09685, 2305.14314, 2210.03629)
uv run python -c "from src.copilot.retrieval import DenseIndex; DenseIndex().build()"   # build the index
uv run python -m eval.eval_agent          # agent on the 20 questions, with the LLM judge
uv run python -m eval.ablate_rewrite      # the prompt ablation
```

**API.** `uv run uvicorn src.copilot.api:app --port 8000`, then:

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What number format is used for the frozen weights in QLoRA?"}'
```

The response has `answerable`, `answer`, `sources` (only passages that were really retrieved, with paper, section and page), the search queries the agent used, steps and token count.

**Docker** (the image holds the code and dependencies; the index and the key are supplied at run time):

```bash
docker build -t paper-assistant .
docker run -p 8000:8000 --env-file .env -v "$PWD/data:/app/data" paper-assistant
```

## How it was built

Step by step, with notes on each concept (BPE, sampling, tool use, embeddings, chunking, BM25, reranking, LLM-as-judge, agents) in `notes/NOTES.md` and the plan in `ROADMAP.md`.
