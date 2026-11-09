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

## How the design followed the measurements

| Stage | Question | What was measured | Decision |
|---|---|---|---|
| 1. Baseline retrieval | Dense, BM25 or hybrid? | BM25 won on the synthetic set (its questions reuse the chunk's wording), dense won on 15 hand-written paraphrases | Hybrid (RRF) as the hedge |
| 2. Reranking | Does a cross-encoder help? | recall@1 improved in all four settings tried | Add the reranker |
| 3. RAG vs agent for QA | Does the agent beat fixed RAG? | top-5 RAG 9/12, top-10 RAG 11/12, agent 12/12 (n=12): most of the gap was "sees more passages"; clear only for questions needing two papers | Keep the agent for multi-paper work |
| 4. Scale | Does it hold at 59 papers? | hybrid recall@5 0.93 -> 0.83 with 13x more chunks | Keep hybrid + rerank |
| 5. Vector database | Is Chroma/HNSW worth it? | identical results at 4.6k vectors; HNSW only wins above ~10k | numpy default, Chroma optional |
| 6. New task: research briefs | Is one retrieval enough? | single retrieval covers 0.48 of the key points; a tool-use loop 0.96 | Agent loop |
| 7. Planning and verification | What do planner and citation checker add? | planner +0.08 support, checker +0.06 more (CIs below) | `staged` pipeline |
| 8. Repair rounds | Are two repair rounds needed? | one round is indistinguishable from two, 20% cheaper | Default to one round (chosen after seeing the data, see Limitations) |
| 9. Framework | Does LangGraph help? | same quality, 59 s -> 38 s from parallel research, plus pause/resume | Offered alongside the hand-written version |

## Results

### Research briefs (the agent task)

15 comparison tasks that need 2-6 papers each, plus 2 "gap" tasks that ask about papers the library does not contain (DPO, Switch Transformer/Mixtral routing). Each expected key point is tied to a phrase that exists in the corpus, so every task is answerable from the library. Every system was run 3 times per task (`staged_r1`: 2 times). A Sonnet 5.5 judge, a different model from the Haiku 5.5 that writes the briefs, scores each brief:

- **support**: share of factual sentences whose claim is stated by the passages they cite (uncited factual sentences count as unsupported);
- **uncited**: factual sentences with no citation (lower is better);
- **key points**: share of the task's expected points the brief states; **papers**: share of required papers cited at least once.

| System | Support | Uncited / brief | Key points | Papers | LLM calls | Cost / brief | Time |
|---|---|---|---|---|---|---|---|
| `single_shot` (top-10 retrieval, one call) | 0.77 | 3.4 | 0.48 | 0.67 | 1 | $0.001 | 5 s |
| `react` (tool-use loop) | 0.75 | 5.4 | 0.96 | 1.00 | 2 | $0.003 | 9 s |
| `staged_noverify` (plan -> research -> write) | 0.83 | 3.6 | 0.98 | 1.00 | 11 | $0.011 | 37 s |
| `staged_r1` (+ verify, 1 repair round) | 0.90 | 2.0 | 0.97 | 1.00 | 14.5 | $0.017 | 50 s |
| `staged` (+ verify, 2 repair rounds) | 0.89 | 1.9 | 0.95 | 1.00 | 16 | $0.022 | 59 s |
| `graph` (`staged` in LangGraph) | **0.92** | **1.4** | 0.95 | 1.00 | 16 | $0.022 | 38 s |

Run-to-run spread is real: `react` support was 0.83, 0.70, 0.73 across its three runs, `staged` 0.93, 0.90, 0.85. So the comparisons below are paired by task and use a bootstrap over tasks (the same tasks are re-run, so runs are not independent samples).

| Comparison (a vs b) | Metric | Mean difference | 95% CI | Tasks a better / tied / worse |
|---|---|---|---|---|
| `react` vs `single_shot` | key points | **+0.48** | [+0.38, +0.56] | 14 / 1 / 0 |
| `react` vs `single_shot` | support | -0.02 | [-0.07, +0.03] | 8 / 0 / 7 |
| `staged_noverify` vs `react` (planning) | support | **+0.08** | [+0.03, +0.11] | 14 / 0 / 1 |
| `staged` vs `staged_noverify` (verifier) | support | **+0.06** | [+0.02, +0.11] | 11 / 0 / 4 |
| `staged` vs `staged_noverify` | uncited sentences | **-1.7** | [-2.6, -0.7] | 11 / 0 / 4 |
| `staged` vs `staged_r1` (2nd repair round) | support | +0.01 | [-0.04, +0.06] | 10 / 0 / 5 |
| `graph` vs `staged` | support | +0.03 | [-0.03, +0.09] | 9 / 0 / 6 |
| `staged` vs `react` | support | **+0.14** | [+0.09, +0.19] | 14 / 0 / 1 |

What the evidence supports:

- **Searching more than once fixes coverage, not rigor.** A tool-use loop raises key-point coverage from 0.48 to 0.96 but leaves citation support unchanged. One retrieval simply cannot reach four or five papers.
- **Planning and verification each add citation support**, and the pipeline's gain over the plain loop is large and consistent (14 of 15 tasks). Verification also removes about 1.7 uncited factual sentences per brief.
- **Support is two things.** On the reports that kept per-sentence verdicts, the share of factual sentences carrying a citation rose from 76-79% (`single_shot`, `react`) to 88-93% (`staged`, `graph`), and the share of cited sentences that were actually supported rose from about 94-95% to about 98%. Part of the gain is simply more citing; part is more accurate citing. (The subsets differ in size across systems, so treat the split as indicative.)
- **A second repair round did not help measurably** (+0.01, CI spans zero) and costs 20% more, so `staged_r1` is the default in the API.
- **LangGraph matched the hand-written pipeline** on quality and cut wall-clock time from 59 s to 38 s because the sub-questions are researched in parallel. It also provides checkpointing and a human-approval step: the run pauses after planning, a person edits the sub-questions, and it resumes (tested on one task).
- **Cost frontier.** `react` reaches the same coverage for roughly 1/9 of the cost of `staged`; if citation rigor is not needed, it is the better buy. The pipeline is worth its cost when an unsupported claim is expensive.
- **The pipeline's own verifier is noisy.** Its Haiku checker flagged 35-45% of sentences on the first pass and still about 37-41% after repair; no brief ended with nothing flagged. The independent judge finds far fewer unsupported sentences, so the in-pipeline verifier over-flags. Repair still improved the judged result, but a calibrated or stronger verifier is the obvious next step.
- **Gap tasks:** in every run (6 per system, 4 for `staged_r1`) the brief said what the library lacks, and none invented details about the missing papers.

Reproduce without any API call: `uv run python -m eval.analyze_research`.

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
- **15 tasks plus 2 gap tasks, written by one author.** Tasks were fixed before any system was run, but the confidence intervals only cover task-to-task variation, not "other people would write other tasks". Repeated runs of the same tasks are not independent, which the bootstrap over tasks accounts for.
- **The judge is one model (Sonnet 5.5) with a strictness I set, and it has not been validated against a human yet.** `eval/audit_sheet.py` produces a blinded sample of 30 judged sentences with their cited passages for a person to label, and `eval/audit_score.py` computes agreement and kappa; until those labels exist, treat the absolute numbers as the judge's opinion. Relative comparisons between systems are more trustworthy than the absolute values because the same judge scores all of them.
- **The `support` metric mixes citation coverage and citation accuracy**, and counts uncited connective or "the library does not cover X" sentences as unsupported. Error analysis of a sample of flagged sentences found many uncited summary or limitation sentences that were not wrong. The split above separates the two, but only for reports that stored per-sentence verdicts.
- **`staged_r1` was chosen after seeing the data.** Comparing one and two repair rounds on the same 15 tasks, then making one round the default, is selection on the evaluation set. The difference was not significant either way; the cost saving is the reason for the choice.
- **The in-pipeline verifier over-flags** (see above) and is the same model family as the writer, so its flags are weak evidence; the reported numbers come from the independent judge.
- **Sentence splitting is a heuristic.** Sentences under 26 characters are not checked by either the verifier or the judge (pinned by a test). `uncited` counts factual sentences the judge found; the same brief can score differently on a re-judge.
- **Token limits caused failed runs** (briefs with 5-6 items exceeded 1,500-2,500 output tokens); the limits were raised and the failed runs retried. Runs that completed before the change were not re-run. An API spending limit also interrupted one batch; those failures were removed and the runs repeated.
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
- **No latency measurement for the QA stages and no distributed tracing** (the research systems record a step-by-step trace and per-brief timing). Brute-force dense search over 4,645 vectors is fine; it would not be at millions of chunks.

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
