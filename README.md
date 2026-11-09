# LLM Paper Assistant

A research agent over a library of 59 arXiv papers on LLMs, RAG, agents, fine-tuning and efficiency (`data/papers.json`). Given a research task such as "compare LoRA, QLoRA, adapters and prefix-tuning", it plans sub-questions, searches the library, writes a cited brief, checks its own citations and repairs the ones the evidence does not support. It says what the library does not cover instead of inventing it.

The point of the project is the measurement as much as the system: what each component buys, what it costs, and where the evaluation itself is weak. It started as plain RAG and was turned into an agent step by step; every step has numbers and the weak spots are listed under Limitations.

## Architecture

```
                 +---------------- retrieval tool (search_papers) ----------------+
PDFs -> chunks ->|  BM25 top 100 + dense MiniLM top 100 -> RRF -> cross-encoder -> top 5 |
(4,645, 59 papers)+------------------------------------------------------------+
                                           ^
research task -> plan (3-6 sub-questions) -> research each sub-question (tool-use agent, in parallel in the LangGraph version)
              -> write cited brief -> verify every cited sentence against its passages
              -> if unsupported claims: find more evidence, revise (up to 2 rounds) -> brief + sources
```

Five ways to produce the same brief are implemented and compared (`src/copilot/research.py`, `research_graph.py`): `single_shot` (one retrieval, one generation), `react` (the tool-use loop asked to write a report), `staged` (the pipeline above), `staged_noverify` (ablation without the verifier) and `graph` (the staged pipeline as a LangGraph `StateGraph`).

| Part | Code |
|---|---|
| PDF parsing and chunking (section metadata, appendix handling, table residue filter) | `src/copilot/ingest.py` |
| Dense retrieval (all-MiniLM-L6-v2) with a numpy or Chroma/HNSW backend, BM25 (hand-written), RRF, cross-encoder reranker | `retrieval.py`, `vectorstore.py`, `bm25.py`, `rerank.py`, `pipeline.py` |
| Tool-use agent loop (strict `final_answer` schema) | `src/copilot/agent.py` |
| Research systems and shared step functions | `src/copilot/research.py` |
| The same pipeline in LangGraph (parallel fan-out, conditional repair loop, human approval of the plan) | `src/copilot/research_graph.py` |
| FastAPI service (`/ask`, `/research`), API-key gate, CloudWatch metrics line | `src/copilot/api.py` |
| Fixed RAG baseline | `src/copilot/rag.py` |
| Evaluation | `eval/` (research briefs: `research_tasks.json`, `eval_research.py`, `analyze_research.py`) |
| Tests, CI, Lambda deployment | `tests/`, `.github/workflows/ci.yml`, `Dockerfile`, `template.yaml`, `docs/DEPLOY.md` |

## Results

### Research briefs (the agent task)

15 comparison tasks that need 2-6 papers each, plus 2 "gap" tasks that ask about papers the library does not contain (DPO, Switch Transformer/Mixtral routing). Each key point in a task is tied to a phrase that exists in the corpus, so the task is answerable from the library. A Sonnet 5.5 judge (not the Haiku model that writes the briefs) scores every brief:

- **support**: share of factual sentences whose claim is stated by the passages they cite (higher is better);
- **uncited**: factual sentences with no citation at all (lower is better);
- **key points**: share of the task's expected points the brief states; **papers**: share of required papers cited at least once.

| System | Support | Uncited / brief | Key points | Papers | LLM calls | Cost / brief | Time |
|---|---|---|---|---|---|---|---|
| `single_shot` (top-10 retrieval, one call) | 0.78 | 3.5 | 0.49 | 0.67 | 1 | $0.001 | 6 s |
| `react` (tool-use loop) | 0.76 | 5.3 | 0.96 | 1.00 | 2.1 | $0.003 | 9 s |
| `staged_noverify` (plan -> research -> write) | 0.83 | 3.9 | 0.98 | 1.00 | 11 | $0.011 | 37 s |
| `staged` (+ verify and repair citations) | **0.92** | **1.6** | 0.94 | 1.00 | 16 | $0.021 | 59 s |
| `graph` (staged in LangGraph, parallel research) | 0.92 | 1.1 | 0.94 | 1.00 | 16 | $0.022 | **37 s** |

Paired differences (same task and run; wins / ties / losses over about 28 pairs, `eval/analyze_research.py`):

- Searching more than once is what matters most: `react` vs `single_shot` raises key-point coverage by 0.48 (14 wins, 0 losses). A single retrieval cannot cover four or five papers.
- Planning separate sub-questions adds a little citation support over the plain loop (`staged_noverify` vs `react`: +0.06, 19 wins / 8 losses).
- The citation verifier adds more (`staged` vs `staged_noverify`: +0.07 support, 15 wins / 8 losses, and 2.0 fewer uncited sentences per brief), at the price of about 2x the calls. It slightly lowers key-point coverage (-0.05, 0 wins / 5 losses): when it deletes an unsupported claim, a key point sometimes goes with it.
- The `react` loop is the cost-efficient choice if citation rigor does not matter: it reaches the same coverage for 1/8 of the cost of `staged`. `staged` is worth it when unsupported claims are expensive.
- LangGraph gives the same quality as the hand-written pipeline (support -0.01, within noise) and is 37 s instead of 59 s, because the sub-questions are researched in parallel. It also gives checkpointing and a human-approval step for free: the run pauses after planning, a person edits the sub-questions, and it resumes.
- On both gap tasks, every system said what was missing in every run and none invented details (acknowledged every time, fabricated 0; 2 tasks x 1-2 runs per system).

How much to trust this: see Limitations. The effects are modest and the sample is small; read these as evidence, not proof.

### Question answering (the earlier RAG stages)

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

**Research-brief evaluation**
- **15 tasks plus 2 gap tasks, written by one author**, and the same tasks are used in both runs, so the paired comparisons are not independent samples. A third run was planned and cut short when the Anthropic account hit its monthly API usage limit; only runs 0 and 1 are used (about 28 pairs per comparison). Failed attempts from that period were removed from the results file.
- **The judge is one model (Sonnet 5.5) with a strictness I set.** "Unsupported" depends on its reading of the passages; I did not validate it against a human labeller. The verifier inside `staged` is Haiku checking Haiku's own text, which can be lenient or share blind spots; the Sonnet judge is what the table reports.
- **Sentence splitting is a heuristic.** Sentences under 26 characters are not checked by either the verifier or the judge (pinned by a test). `uncited` counts factual sentences the judge found; the same brief can score differently on a re-judge.
- **Token limits caused failed runs** (briefs with 5-6 items exceeded 1,500-2,500 output tokens); the limits were raised and the failed runs retried. Runs that completed before the change were not re-run.
- **Retrieval was not thread-safe.** Concurrent torch calls hung a test, so retrieval is serialised with a lock; the parallel speed-up in the LangGraph version comes from overlapping the model calls, not the retrieval.
- **Cost is the model API only** (Haiku 5.5 at its list price; judge cost not included). Latency is wall-clock on a laptop with up to 3 briefs running at once.
- **Deployment is untested.** The Dockerfile, SAM template and CI workflow have not been built or deployed (no Docker or AWS credentials where I wrote them). See `docs/DEPLOY.md`.

**Question-answering stages**

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

**Tests.** `uv run python -m pytest` (11 unit and API tests, no network or API key needed).

**Research briefs from the command line:** `uv run python -m eval.eval_research staged graph --runs 1` writes `eval/research_results.json`; `uv run python -m eval.analyze_research` prints the tables above without calling any API.

**API.** `uv run uvicorn src.copilot.api:app --port 8000` (set `APP_API_KEY` to require an `x-api-key` header), then:

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What number format is used for the frozen weights in QLoRA?"}'
```

`POST /research` with `{"task": "...", "system": "staged"}` returns the brief, its sources, cost, calls and a trace of the steps. Deploying to AWS Lambda is described in `docs/DEPLOY.md`.

The `/ask` response has `answerable`, `answer`, `sources` (only passages that were really retrieved, with paper, section and page), the search queries the agent used, steps and token count.

**Docker** (the image holds the code and dependencies; the index and the key are supplied at run time):

```bash
docker build -t paper-assistant .
docker run -p 8000:8000 --env-file .env -v "$PWD/data:/app/data" paper-assistant
```

## How it was built

Step by step, with notes on each concept (BPE, sampling, tool use, embeddings, chunking, BM25, reranking, LLM-as-judge, agents) in `notes/NOTES.md` and the plan in `ROADMAP.md`.
