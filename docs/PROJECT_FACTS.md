# Project facts (for anyone summarising this repository, e.g. into a resume)

This file separates what was done and measured from what was only written or planned. Please do not claim anything in the "not done" list. All numbers are from this repository's own evaluation; see the README for the tables and `eval/` for the code that produced them.

## What it is

A research agent over a library of 59 arXiv papers (LLMs, RAG, agents, fine-tuning, efficiency). Given a research task, it plans sub-questions, searches the library with a hybrid retrieval tool, writes a brief with citations, checks its own citations against the cited passages, and revises unsupported claims. It states what the library does not cover instead of inventing it. The project started as plain RAG question answering and was developed step by step into the agent, with an evaluation harness at every step.

## How it was built (be accurate about this)

- Solo, personal project. Not a work project; no users; no production traffic.
- **Developed with an AI coding assistant (Claude Code).** The owner chose the scope and direction, reviewed results and made decisions; the assistant wrote much of the code and wrote the notes. The owner should be prepared to explain every component and every number in an interview, and should describe their own contribution truthfully.
- Written in Python 3.13; about 1,000 lines in `src/`, 800 in `eval/`, 100 in `tests/`; about 30 commits.

## Technical content (all implemented and run locally)

| Area | What exists |
|---|---|
| Ingestion | PDF parsing with PyMuPDF; structure-aware chunking (300 tokens, 50 overlap, section and page metadata); 59 papers -> 4,645 chunks; a paper manifest with stable chunk ids |
| Retrieval | Dense embeddings (all-MiniLM-L6-v2), hand-written BM25, reciprocal rank fusion, cross-encoder reranker; optional Chroma/HNSW vector database backend |
| Agent | Hand-written tool-use loop with Anthropic's Messages API (tool use, strict tool schema, step cap, forced final answer) |
| Research pipeline | Planner, per-sub-question researchers, writer, citation verifier, repair loop; also implemented as a LangGraph `StateGraph` (parallel fan-out with `Send`, conditional edges, reducer, checkpointing, human-approval interrupt) |
| Service | FastAPI with `/ask` and `/research`, API-key check, input validation, CloudWatch-style metrics line; retrieval serialised with a lock |
| Evaluation | Retrieval metrics (recall@k, MRR) on a synthetic and a hand-written set; LLM-as-judge (a different, larger model than the generator) for answers and for research briefs; paired comparisons with bootstrap confidence intervals; ablations; failure analysis; cost and latency per brief |
| Engineering | 11 pytest tests; GitHub Actions workflow; pinned requirements; secrets kept out of git |

## Verified results (research briefs: 15 tasks x 3 runs per system, plus 2 gap tasks)

- A single retrieval covered 48% of the expected key points; a tool-use loop covered 96% (difference +0.48, 95% CI +0.38 to +0.56).
- The planning-plus-verification pipeline raised the judged citation-support rate from 0.75 (plain loop) to 0.89-0.92; the paired difference was +0.14 (95% CI +0.09 to +0.19), better on 14 of 15 tasks. Planning alone contributed about +0.08 and the citation verifier about +0.06 more.
- Cited sentences were accurate about 94% of the time for the baselines and about 98% for the verified pipeline, and more sentences carried citations (about 76-79% -> 88-93%).
- A second repair round gave no measurable gain (+0.01, CI spans zero) at about 20% higher cost, so one round is the default.
- The LangGraph version matched the hand-written pipeline's quality and cut wall-clock time from 59 s to 38 s by researching sub-questions in parallel.
- Cost per brief (model API): about $0.001 single retrieval, $0.003 plain loop, $0.017-0.022 for the verified pipeline.
- Earlier stages: hybrid retrieval and a reranker were compared across two test sets, with the finding that the best retriever changed with how the test set was written; growing the library from 5 to 59 papers lowered recall for every retriever (hybrid recall@5 0.93 -> 0.83); a fair top-10 RAG baseline closed most of the apparent advantage of the agent on single-paper questions.
- Chroma/HNSW matched exact search on results at 4.6k vectors; an ANN scaling experiment on synthetic data showed exact search faster at 5k vectors and HNSW faster at 50k and 500k (the crossover was not located).

## Things found and fixed along the way (good interview material, all real)

A thread-safety hang when running retrieval concurrently (fixed with a lock; it would also have hit the first API version); judge output truncated by a token limit; briefs truncated by output limits (checked `stop_reason`); a manifest ordering mistake that would have silently invalidated all older gold labels (caught by an id-stability check, now a test); a mislabelled "unanswerable" test question; a first ANN benchmark on random vectors that gave misleading recall; an evaluation metric that mixed citation coverage and citation accuracy until it was decomposed.

## Honest limitations (also in the README)

Small test sets written by one person; one LLM judge; the judge's strictness is set by the author; the pipeline's own verifier over-flags; one design choice (one repair round) was made after seeing the evaluation data; support rates are on the judge's reading.

## NOT done - do not claim

- **Not deployed to AWS.** A Dockerfile, a SAM/CloudFormation template and a CI workflow were written, and dependency resolution for Linux was checked, but the image has not been built locally and nothing has been deployed. Whether CI builds the image successfully is unknown until it has run on GitHub.
- **No human validation of the LLM judge.** A 30-sentence audit was done by an AI model that is the same model as the judge; it is not independent and not human. The human audit tooling exists but has no human labels.
- No SageMaker, no model training or fine-tuning, no GPU work, no Kubernetes, no knowledge graph, no streaming/Kafka, no MCP server, no distributed tracing.
- No production usage, users, scale or SLAs. Latency and cost figures are from a laptop and the model API list prices.
- The test sets are small (15 research tasks, 21 older QA questions, 60 synthetic retrieval questions). The statistical statements above hold for these tasks only.
- Claude API access was used directly, not Amazon Bedrock.

## Accurate phrasing examples (adapt freely, keep the numbers and caveats true)

- Built a research agent over 59 arXiv papers (hybrid BM25 + dense retrieval with cross-encoder reranking, plan-research-write-verify pipeline) and evaluated it with an LLM judge, paired bootstrap comparisons and ablations.
- Showed that adding a citation verifier raised judged citation support by about 0.06 (95% CI 0.02-0.11) and that planning raised it by about 0.08; found a second repair round added nothing measurable at +20% cost.
- Re-implemented the pipeline in LangGraph (parallel fan-out, conditional repair loop, human-approval checkpoint), matching quality while cutting latency from 59 s to 38 s.
- Diagnosed and fixed concurrency, truncation and evaluation-validity bugs (thread-unsafe inference, output-limit truncation, metric conflating coverage and accuracy).
- Wrote a FastAPI service with API-key auth, tests (pytest), a CI workflow, and a Dockerfile/SAM template for AWS Lambda (written, not deployed).

## Mapping to common job-posting terms (what is honest to say)

| Posting term | Status |
|---|---|
| LLMs, prompt/tool use, structured output | Done (Anthropic Messages API, strict tool schemas) |
| Advanced RAG, semantic search, retrieval systems | Done (hybrid retrieval, reranking, evaluation) |
| Vector databases | Experiment only (Chroma/HNSW backend, ANN scaling study); default search is exact numpy |
| Agentic AI frameworks | Done in LangGraph, plus a hand-written agent loop |
| Evaluation / LLM-as-judge / model quality | Done, a strength of the project |
| Python, FastAPI, testing, version control | Done |
| Docker, CI/CD | Written, not verified end to end |
| AWS (Lambda, SSM, CloudWatch, SAM) | Prepared, not deployed |
| SageMaker, MLOps, Kubernetes, knowledge graphs, Kafka | Not done |
