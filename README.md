# LLM Paper Assistant

A question-answering system over a library of 59 arXiv papers on LLMs, RAG, agents, fine-tuning and efficiency (the list is `data/papers.json`). It grows from plain RAG to an agentic RAG loop, and every step is measured with an evaluation harness. Answers carry citations, and the system refuses when the papers do not contain the answer.

The point of the project is the measurement as much as the system: what each component buys, and where the evaluation itself is weak.

## Architecture

```
PDFs -> ingest (structure-aware chunks, 300 tok, 50 overlap) -> 4,645 chunks (59 papers)
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

All numbers are small-sample. Read the limitations before quoting them. The project started with five papers (Transformer, RAG, LoRA, QLoRA, ReAct) and the evaluation sets were written for them; I later grew the library to 59 papers and re-ran everything. Tables show both where I have both.

**Retrieval** (60 synthetic questions whose gold chunk is in the original five papers; `eval/`)

| Retriever | recall@5, 5 papers | recall@5, 59 papers | MRR, 5 papers | MRR, 59 papers |
|---|---|---|---|---|
| Dense | 0.783 | 0.700 | 0.688 | 0.615 |
| BM25 | 0.933 | 0.917 | 0.839 | 0.790 |
| Hybrid (RRF) | 0.933 | 0.833 | 0.789 | 0.728 |

With 13x more chunks (348 -> 4,645) every retriever gets worse, dense most and BM25 least: exact-term matching holds up better against many distractors than a small embedding model does.

The synthetic questions reuse wording from the gold chunk, which favours BM25. On 15 hand-written paraphrased questions the order flips and dense does better, so no retriever is declared the winner. Hybrid is the hedge.

**Reranking** (hybrid top 30 -> cross-encoder)

| Test set | recall@1 before -> after | MRR before -> after |
|---|---|---|
| Synthetic (n=60), 5 papers | 0.683 -> 0.767 | 0.789 -> 0.855 |
| Hand-written (n=15), 5 papers | 0.467 -> 0.600 | 0.592 -> 0.717 |
| Synthetic (n=60), 59 papers | 0.617 -> 0.733 | 0.728 -> 0.822 |
| Hand-written (n=15), 59 papers | 0.200 -> 0.400 | 0.354 -> 0.498 |

The reranker helps in all four settings. The hand-written drop at 59 papers is partly a label problem, not only a retrieval problem: see Limitations.

**Generation** (15 answerable + 5 unanswerable hand-written questions, Claude Sonnet 5.5 as judge, different from the Haiku generator)

| | Correct (answerable) | Wrongly refused | Unanswerable refused | Searches | Input tokens |
|---|---|---|---|---|---|
| Fixed RAG | 13/15 | 1-2 | 5/5 | 1 | ~1.5-2k |
| Agent, 5 papers | 15/15 | 0 | 5/5 | ~2 | ~5.5-6k |
| Agent, 59 papers | 15/15 | 0 | 5/5 | 2.0 | ~10.7k |

The agent answers more questions correctly at roughly three times the token cost, and about twice that again with the larger library (59 titles in the system prompt and tool schema, plus longer results; I did not isolate which contributes how much). The model is not deterministic, so numbers vary between runs; a one-question difference on n=20 is noise.

**Ablation: does the "rephrase the query" prompt sentence matter?** I predicted it would. Removing it changed almost nothing (1.90 vs 1.95 searches per question, same gold recall), so query rewriting comes mostly from the model and the search results, not from that sentence. One run per condition; this means "no effect detected", not "no effect". See `eval/ablate_rewrite.py`.

## Limitations

- **Small, biased test sets.** 60 synthetic questions with lexical leakage and 15 hand-written ones written by one person. More questions would not remove that bias.
- **The agent was developed against the same 20 questions it is evaluated on.** There is no held-out set, so its numbers are optimistic.
- **The agent-versus-RAG comparison is confounded.** The agent sees more chunks (about 10 across two searches) than fixed RAG sees (5). It also benefits from the model's prior knowledge of these famous papers; private documents may behave differently.
- **Citation accuracy is not measured.** The judge sees passages numbered `[1]..[5]`, while the agent cites chunk ids like `c278`, so the judge cannot check the citations. It flagged this in its reasoning.
- **The judge is imperfect.** I read its reasoning and found real bugs in the evaluation (truncated passages, missing metadata), which I fixed; I did not validate it against a second human labeller.
- **One over-generalisation seen:** on an ambiguous question ("What optimizer ... was used to train the model?") the agent answered for three papers and described one loosely; the judge marked it unfaithful.
- **Hand-written gold labels are not valid for a bigger library.** Questions like "What optimizer was used to train the model?" or "How does the system decide which documents to fetch?" were written when only five papers existed. With 59, other papers (DPR, Atlas, REALM, ...) may answer them correctly, but they count as misses. Part of the recall drop at 59 papers is therefore false negatives in the labels (likely; I did not read each case). The end-to-end agent numbers are not affected the same way because the judge reads the answer, not the chunk id.
- **The unanswerable questions had to be protected.** I left out papers that would answer them (Llama 3, Mixtral, DPO, Chinchilla, GPT-4, and the two mixture-of-experts papers). Near-miss papers remain (Kaplan scaling laws for the Chinchilla question), which makes refusal harder, and the agent still refused 5/5.
- **The paper list goes into the prompt.** That works for 59 titles; it does not for thousands. A real deployment would retrieve at paper level or drop the list.
- **PDF parsing is a baseline.** Formulas, tables and some figure text (ReAct trajectories) are garbled or dropped.
- **No latency measurement**, no tracing. Brute-force dense search over 4,645 vectors is fine; it would not be at millions of chunks.

## Run it

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and an Anthropic API key in a `.env` file (`ANTHROPIC_API_KEY=...`, gitignored).

```bash
uv sync
uv run python -c "from src.copilot.papers import load_papers, download; download(list(load_papers()))"   # fetch the PDFs listed in data/papers.json (~160 MB, polite 3 s delay)
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
