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

**A harder set written for the 59-paper library** (`eval/retrieval_set_v3.json`: 12 single-paper questions on the new papers, 5 questions that need two papers, 4 unanswerable). Gold is "paper id + a phrase that must appear in a retrieved chunk", so it does not depend on chunk indices. This also controls the earlier confound: fixed RAG is run with top-5 and top-10, the agent sees about 10.

| Type | System | All gold retrieved | Judged correct | Faithful | Refused | Input tokens |
|---|---|---|---|---|---|---|
| single (n=12) | RAG top-5 | 10/12 | 9/12 | 11/12 | 2 | 2.5k |
| | RAG top-10 | 11/12 | 11/12 | 12/12 | 1 | 4.5k |
| | Agent | 11/12 | 12/12 | 12/12 | 0 | 10.2k |
| cross-paper (n=5) | RAG top-5 | 0/5 | 2/5 | 4/5 | 2 | 2.6k |
| | RAG top-10 | 1/5 | 4/5 | 5/5 | 1 | 4.5k |
| | Agent | 4/5 | 5/5 | 4/5 | 0 | 11.0k |
| unanswerable (n=4) | RAG top-5 | - | 4/4 | 4/4 | 4 | 2.2k |
| | RAG top-10 | - | 4/4 | 4/4 | 4 | 3.9k |
| | Agent | - | 4/4 | 4/4 | 3 | 10.3k |

What this says, with n this small (one question is 8% on the single-paper rows, 20% on the cross-paper rows):
- Most of the earlier "agent beats RAG" gap is explained by seeing more passages: going from top-5 to top-10 fixes about as many questions as switching to the agent (single-paper 9 -> 11 -> 12 correct). The agent is one question ahead of top-10 RAG at 2-2.5x the tokens, which I cannot distinguish from noise.
- The one place the agent clearly retrieves better is questions that need two papers: it finds both papers' evidence in 4 of 5 cases, top-10 RAG in 1 of 5. Its multi-search behaviour matters there; I would want more than 5 such questions before claiming it.
- "Judged correct" is looser than "all gold retrieved" on cross-paper questions: the judge accepts an answer that is consistent with the reference and on topic, even when it did not retrieve both papers. Treat the gold-coverage column as the stricter measure.

**Ablation: does the "rephrase the query" prompt sentence matter?** I predicted it would. Removing it changed almost nothing (1.90 vs 1.95 searches per question, same gold recall), so query rewriting comes mostly from the model and the search results, not from that sentence. One run per condition; this means "no effect detected", not "no effect". See `eval/ablate_rewrite.py`.

**Vector database (Chroma / HNSW) vs numpy brute force.** The dense stage can run on either (`Retriever(dense_backend="numpy" | "chroma")`; the embeddings are loaded into Chroma without re-encoding). On the 60-question set the two give identical end-to-end results (recall@5 0.900, recall@1 0.733); the approximate search returned 99.7% of the exact top-10 and 96.1% of the exact top-100. At 4,645 vectors latency is the same (~10 ms, almost all of it query encoding), so the database brings no benefit at this size. To see where it would, `eval/ann_scaling.py` measures single-query search time on real embeddings plus noise (realistic cluster structure):

| Vectors | Exact (numpy) | HNSW ef=50 | HNSW recall@10 | HNSW build |
|---|---|---|---|---|
| 5,000 | 0.08 ms | 0.20 ms | 0.987 | 0.3 s |
| 50,000 | 0.91 ms | 0.27 ms | 1.000 | 5.7 s |
| 500,000 | 9.28 ms | 0.26 ms | 0.990 | 66.8 s |

Brute force grows linearly with the number of vectors, HNSW barely moves, at a recall cost that `ef` tunes (ef=20: 0.95 at 500k; ef=200: 1.000 but 0.59 ms). Below roughly 10k vectors, exact search is faster. A first run on purely random vectors gave HNSW recall of 0.02 at 500k: random vectors in 384 dimensions are almost equidistant, which is the worst case for ANN, so those numbers say nothing about text embeddings. The scaling data is synthetic (noise added to real embeddings), not a measurement of real 500k-chunk corpora.

## Limitations

- **Small, biased test sets.** 60 synthetic questions with lexical leakage and 15 hand-written ones written by one person. More questions would not remove that bias.
- **Small sets, written by one author.** The v3 set has 21 questions; a single question swings a row by 8-20%. The agent was developed against the 20 older questions, not against v3, but v3 is still too small to rank systems with confidence. I replaced one v3 "unanswerable" question after seeing a result, because it was mislabelled (the GQA paper does mention speculative sampling, so the system was right to answer); the replacement was checked against the corpus first.
- **The agent-versus-RAG comparison is only partly controlled.** On v3 I added a top-10 RAG baseline, which closes most of the gap (see above); the older 20-question comparison still pits the agent against top-5 only. The model also knows these famous papers from pretraining, so results on private documents may differ.
- **Citation accuracy is not measured.** The judge sees passages numbered `[1]..[5]`, while the agent cites chunk ids like `c278`, so the judge cannot check the citations. It flagged this in its reasoning.
- **Structured flag and text can disagree.** On one unanswerable question the agent wrote "the library does not contain the Llama 3 paper" and then described Llama 2, while setting `answerable=true`. The judge graded the text.
- **One unexplained crash.** An agent run once failed with a missing `answer` field in the `final_answer` call despite `strict` mode. I suspect truncation at `max_tokens=1200` but did not confirm it; I raised the limit to 2500 and made truncation raise an explicit error. It did not recur in the rerun.
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
