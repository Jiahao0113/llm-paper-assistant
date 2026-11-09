# LLM Study Notes

Running notes, one section per concept. Newest concepts at the bottom.

---

## 1. Tokenization and BPE (Byte Pair Encoding)

### Why a tokenizer
Models only process numbers, so text must become integer IDs. Three options:

| Approach | Vocab size | Problem |
|---|---|---|
| Character-level | small (hundreds to 150k+ depending on languages) | sequences too long; attention cost is O(n^2); single characters carry little meaning |
| Word-level | huge | OOV (out-of-vocabulary): unseen words cannot be represented |
| **Subword** | tunable (e.g. 100k) | compromise: common words = 1 token, rare words split into pieces |

BPE is the most common algorithm for building a subword vocabulary.

### Training BPE
Core idea: **repeatedly merge the most frequent adjacent pair into a new token.**

1. Start with base symbols (bytes or characters).
2. Count all adjacent pairs in the corpus.
3. Merge the most frequent pair into one new token; record it as a *merge rule*.
4. Repeat until the vocabulary reaches the target size.

Toy example (corpus: low x5, lower x2, newest x6, widest x3):
`e s` (9 times) -> `es`; then `es t` -> `est`; then `l o` -> `lo`; then `lo w` -> `low` ...

Output of training = **vocabulary** (token <-> ID) + **ordered merge rules**.

### Encoding new text
Split into base symbols, then apply merge rules **in training order**.
`lowest` -> `l o w e s t` -> ... -> `low` + `est`. Unseen words still work.

### Byte-level BPE (GPT family)
- Base symbols are the 256 UTF-8 bytes, not characters. Any text in any language can be represented, so no OOV.
- Rare characters get split into multiple byte tokens (that is the `�` seen in `tokenizer_demo.py` for "世").
- English-heavy training data => English is compressed well, Chinese less so => more tokens per sentence => higher API cost and faster context-window usage.

### Key facts to remember
- The tokenizer is trained **separately and before** the model, then frozen. No gradient descent involved.
- Model and tokenizer must match; a different tokenizer means different ID meanings.
- Vocab size trade-off: larger = shorter sequences, but bigger embedding matrix and under-trained rare tokens. Smaller = longer sequences.
- Variants: WordPiece (BERT), SentencePiece/Unigram (Llama, T5). BPE is enough for interviews.
- LLMs fail at letter-counting ("how many r in strawberry") because they see tokens (`str`+`aw`+`berry`), not letters.

### Self-check Q&A
1. **What is the merge criterion?** The most frequent adjacent pair in the corpus.
2. **Why no unseen words?** Any word can fall back to base symbols (bytes), so everything is representable even if never merged.
3. **Vocab shrinks 100k -> 1k, what happens to sequence length?** It gets longer. Fewer merges means each token covers less text, so the same sentence needs more tokens (in the extreme, one token per byte). Longer sequences cost more compute (attention is O(n^2)) and fill the context window faster.

Code: `step0_llm_basics/tokenizer_demo.py`

---

## 2. How an LLM generates text: sampling

### What the model outputs
- Each step the model outputs **logits** (one raw score per vocab token), then **softmax** turns them into a **probability distribution** over the whole vocab.
- So an LLM is a ~100k-class classifier run once per step. Training loss = cross-entropy, same as ordinary classification.
- The token is **chosen from** the distribution by a sampling rule: `model -> distribution -> (sampling) -> token`.
- **Autoregressive generation**: append the chosen token to the input, predict again, repeat until the end-of-sequence token.

### Decoding strategies
| Method | Rule | Pros / cons |
|---|---|---|
| Greedy | always take the highest-probability token | deterministic, but dull/repetitive; locally best != globally best |
| Random sampling | draw from the distribution | diverse, but the long tail (many tiny probabilities summing to a lot) occasionally yields nonsense |

### Temperature (T)
`probability = softmax(logits / T)`
- T < 1: sharper distribution, more conservative (exploitation).
- T = 1: original distribution.
- T > 1: flatter distribution, more diverse but riskier (exploration).
- T = 0 is equivalent to greedy (implemented as argmax). Same input + same weights => usually same output. "Not always" because GPU floating-point ordering can flip near-tied tokens.
- Analogy: simulated annealing temperature (but LLM T is usually fixed during a generation).
- Use low T (0 to 0.3) for extraction, code, factual QA, RAG; higher T (0.7 to 1.0) for creative writing.

### Top-p (nucleus sampling) and top-k
- Top-p: sort tokens by probability, keep the smallest set whose cumulative probability >= p, drop the rest, renormalize, then sample.
- p = 1.0 means no filtering. Very small p keeps only a few tokens (if the top token already has probability >= p, only that one).
- The set size is **adaptive per step**: model confident => few tokens kept; model unsure => many. The user sets p once.
- Top-k keeps a fixed k tokens regardless of the distribution, so it is less flexible.
- Temperature changes the **shape**; top-p/top-k cut the **tail**. Usually tune only one of them.

### Entropy and the shape of the distribution
- **Entropy** measures how uncertain the model is at a step. Low entropy = sharp distribution (e.g. after "The capital of France is", " Paris" gets most of the mass). High entropy = flat distribution (e.g. after "The cat sat on the", many valid continuations).
- Observed with GPT-2: at T=1.0 the top token " floor" has only p=0.076, so random sampling is naturally diverse here.
- The probability is conditional on the context, NOT the word's overall frequency in the corpus.
- top-p=0.9 keeps 728 of 50257 tokens: the long tail is very long, but most of those tokens carry almost no probability.
- Provider note: newer Claude models no longer expose `temperature`/`top_p` in the Messages API (anthropic SDK 2.3.0), so control output via prompting. Always check the provider docs.
- For RAG, prefer T=0 (or near 0) and no top-p: answers should be grounded in retrieved text, reproducible and easy to evaluate.

Code: `step0_llm_basics/gpt2_sampling.py`

---

## 3. Using an LLM API (Step 1)

### Messages and system prompt
- Roles: `user`, `assistant`; the `system` prompt is a separate parameter in the Claude API.
- The API is **stateless**: the model remembers nothing between requests. The application resends the whole history each time, so cost and context usage grow with conversation length (per-request cost ~linear, cumulative cost ~quadratic).
- A system prompt sets role, rules and output format. For RAG it will say "answer only from the provided documents and cite sources" to reduce **hallucination**.
- A response `content` is a **list of blocks** (`text`, `thinking`, later `tool_use`). Always filter by `block.type`; never assume `content[0].text`.

### Zero-shot vs few-shot (in-context learning)
- Few-shot = put input/output examples in the prompt (as earlier user/assistant turns). No weights change, only the input does.
- Why it works: the model computes P(next token | context), so a different context gives a different distribution; pretraining on repetitive text taught it to continue patterns (induction heads are one known mechanism).
- It shifts probabilities, it does not guarantee anything. Main value: stabilizing the output format.
- Few-shot vs fine-tuning: prompt changes are fast, cheap and temporary; fine-tuning changes weights, costs training, is permanent. Try prompting first.

### Structured output
- Pass a JSON schema; the API guarantees the output matches it (**constrained decoding**: tokens that would violate the schema get probability 0).
- Prompt-only JSON failed in two ways: (1) markdown code fence around the JSON -> `json.loads` raises (loud failure); (2) the model invented a category value outside the allowed set -> later `if category == ...` silently never matches (silent failure, worse).
- Pydantic `BaseModel`: define the shape once; `model_json_schema()` produces the schema, `model_validate_json()` parses and validates. `Literal[...]` becomes an enum. `extra="forbid"` is needed because the API requires `additionalProperties: false`.
- A schema guarantees **format**, not **correctness**. Correctness is measured by evaluation (Step 4).

Code: `step1_prompting/` (`messages_demo.py`, `few_shot_demo.py`, `structured_output_demo.py`)

### Tool use (function calling)
Flow: question + tool list -> model replies `stop_reason="tool_use"` with `tool_use` blocks (name + arguments) -> **our code** runs the function -> we send back a `tool_result` block (matched by `tool_use_id`) inside a user message -> model continues -> `stop_reason="end_turn"` means the final answer.
- The model never executes anything; it only emits a structured request. We control what tools can do (security is our responsibility).
- Tool `name`, `description` and `input_schema` (JSON schema) are effectively prompt text; bad descriptions lead to wrong or missing tool use.
- The full history (assistant tool_use + user tool_result) must be resent each turn (stateless API).
- The model may issue **parallel tool calls** (e.g. search "QLoRA" and "LoRA" in one turn).
- `while stop_reason == "tool_use"` is the skeleton of an **agent loop**. Real agents also need a max-iteration cap (avoid infinite loops and runaway cost) and error handling for failed tools.
- **Grounding problem**: with retrieved text the model still mixed in facts from its own memory (e.g. "roughly 4x", "16-bit") that were not in the results. It cannot tell retrieved facts from remembered ones, so unsupported claims may slip in with no citation. Mitigations: system prompt rules ("answer only from the documents, cite each claim") and measuring **faithfulness** in evaluation (Step 4).

Code: `step1_prompting/tool_use_demo.py`

---

## 4. Embeddings and vector search (Step 2)

### Embedding
- An **embedding** maps a piece of text to a fixed-length vector (384 dims for all-MiniLM-L6-v2) so that texts with similar meaning have nearby vectors. Sentence models pool token vectors (e.g. mean pooling) into one vector.
- Trained with **contrastive learning**: pull related pairs (question/answer) together, push unrelated ones apart (cross-entropy over "which candidate is the correct pair").
- Idea behind it: a word's meaning is determined by the contexts it appears in, so `memory` and `RAM` land close together. Keyword matching compares strings and cannot see this.
- **Cosine similarity** = a.b / (|a||b|), range -1..1. With vectors normalized to length 1, cosine == dot product.

### Brute-force vector search (`step2_retrieval/mini_retriever.py`)
- Store chunk vectors in a matrix `(num_chunks, dim)`. For a query vector q (dim,), `matrix @ q` gives all similarities at once, shape `(num_chunks,)`. `argsort` then picks the top-k.
- Complexity O(N*d) per query. Fine up to ~100k chunks. At 100M chunks x 384 float32 it is ~150 GB of memory plus slow scans, so vector databases use **ANN** (approximate nearest neighbor) indexes such as HNSW, trading a little recall for large speedups.

### Limits seen in experiments (why RAG needs more than embeddings)
- A small model ranked an unrelated sentence containing the word "memory" above a truly relevant one; "Dropout" ranked 2nd for a fine-tuning memory question.
- Scores are not comparable across queries (best score 0.28 for one query, 0.50 for another), so a fixed relevance threshold is unreliable.
- Remedies planned: hybrid search (BM25 + dense), reranker (Step 5), and measuring retrieval quality with evaluation (Step 4).
- RAG is garbage-in/garbage-out: bad retrieval makes a good generator answer wrongly, so retrieval and generation are evaluated separately.

Code: `step2_retrieval/embedding_demo.py`, `step2_retrieval/mini_retriever.py`

---

## 5. Chunking and ingestion (Step 2, continued)

### Why chunk
- One vector cannot represent a whole paper (averaging blurs meaning), embedding models truncate long input (~256 tokens for MiniLM), and stuffing whole papers into the prompt is costly and distracts the LLM.
- Too small: not enough context (e.g. "therefore we default to rank 8" without the antecedent), many chunks. Too large: vague embedding, wasted prompt tokens. Typical start: 200-500 tokens; tune with evaluation.

### Techniques
- **Structure-aware chunking**: split along sections/paragraphs, with min/max size (merge tiny paragraphs, split huge ones); recursive splitting tries separators in order (heading, blank line, newline, sentence, space).
- **Overlap** (10-20%): a sentence at a boundary appears in both neighbors so neither chunk loses its context.
- **Metadata** per chunk (paper, section, page) enables citations; prepending the section title to the text before embedding adds context.
- **Small-to-big**: retrieve small chunks, give the LLM the larger surrounding context (to be covered later).

### Parsing real PDFs is messy (`src/copilot/ingest.py`)
- Tables extract as a column of numbers, ligatures (`ﬁ`) and line-break hyphens break words, formulas are flattened, headers/footers/page numbers mix into the text, some fonts decode to garbage (ReAct paper).
- Baseline cleaning: NFKC normalization, de-hyphenation, drop repeated headers and bare page numbers, drop table residue, cut at References.
- Principle: build a baseline, then let **measurement** decide what to improve (avoid premature optimization). Always inspect the actual output of a pipeline.
- Prioritize defects by downstream impact: (1) **information loss** (e.g. appendices dropped) cannot be repaired later; (2) **noise** (garbled or boilerplate chunks) can partly be fixed by reranking/filtering; (3) **wrong metadata** only affects citation display.
- Known issues (v1): garbled text in ReAct figures, fake "sections" from table numbers, wrong title for the Attention paper, appendices after References dropped, paragraphs longer than 300 tokens are not split, tables/formulas not handled.

### First retrieval results (`src/copilot/retrieval.py`, 5 papers -> 231 chunks)
- Good on definitional questions (multi-head attention, RAG, optimal LoRA rank).
- Failure: "What is the ReAct prompting format?" ranked a "Reproducibility statement" chunk first. Eyeballing 5 queries does not scale, which motivates a labeled test set and recall@k (Step 4).

---

## 6. Evaluation and better retrieval (Step 4-5)

### Retrieval metrics
- **recall@k**: fraction of questions whose gold chunk is in the top k. **MRR**: mean of 1/rank of the gold chunk (rewards ranking it high). High recall@5 with low MRR => answer is in the candidates but ranked low => use a reranker.
- Synthetic test set: an LLM writes a question per sampled chunk; that chunk is the gold label. Cheap, but **lexical leakage** (61% of question content words appeared verbatim in the gold chunk) favored keyword matching. Gold labels are also incomplete (several chunks can answer a question), so allow multiple gold chunks.
- **Variance vs bias**: more questions reduce random error but not systematic bias from how the set was built. A significant result on a biased set is not a general result.
- A realistic test set needs: real user phrasing, varied styles (keywords / full sentences), with and without entity names, mixed question types (fact, compare, summarize, multi-hop), difficulty spread, and unanswerable questions.

### BM25 vs dense
- **TF-IDF** = Term Frequency x Inverse Document Frequency. **BM25** = "Best Matching 25" (25th ranking function variant of the Okapi system). BM25 improves TF-IDF with term-frequency saturation (k1) and document-length normalization (b); IDF makes rare terms count more.
- BM25 matches exact words (names, symbols, acronyms); dense matches meaning (paraphrases). Their weaknesses are opposite, so combine them: **hybrid** with **RRF** (score = sum of 1/(60+rank) over systems; uses ranks only because score scales differ).
- Results flipped between test sets (BM25 won on the leaky synthetic set; dense won on hand-written paraphrased questions). Hybrid is a hedge: robust, not always best.

### Reranking (cross-encoder)
- Bi-encoder: query and chunk are encoded separately, vectors can be precomputed (fast, lossy). Cross-encoder: query and chunk go through one Transformer together, giving a precise relevance score, but needs one forward pass per pair (cannot precompute).
- Standard two-stage pipeline: cheap retrieval of top-N candidates (hybrid, N=30), then cross-encoder reranks, then top-5 go to the LLM.
- Result (cross-encoder/ms-marco-MiniLM-L-6-v2): synthetic n=60: recall@1 0.683 -> 0.767, MRR 0.789 -> 0.855; hand-written n=15: recall@1 0.467 -> 0.600, MRR 0.592 -> 0.717. Same direction on both sets, but n is small.

Code: `src/copilot/bm25.py`, `src/copilot/rerank.py`, `eval/compare_retrievers.py`, `eval/compare_v2.py`, `eval/compare_rerank.py`

---

## 7. Generation evaluation and the agent (Step 4 end, Step 6)

### Generation evaluation (`eval/eval_generation.py`)
- Dimensions: **correctness** (answer right), **faithfulness / groundedness** (every claim supported by the retrieved passages), **citation accuracy**, **abstention** (refuse unanswerable questions, do not refuse answerable ones). Faithful-but-wrong happens when retrieval gives the wrong passage; correct-but-unfaithful happens when the model answers from memory.
- **LLM-as-judge**: a different model from the generator (limits self-preference), structured output with a reasoning field, reference passages for correctness. Known issues: self-preference, length/position bias, plain mistakes, instability. Always validate the judge against human reading.
- Real bugs found by reading judge reasoning: passages truncated for the judge (faithful 11/15 -> 14/15 after the fix) and missing paper metadata. **Evaluation code needs evaluating too.**
- Generation is non-deterministic (new Claude models expose no temperature), so repeat runs; on n=20 a 1-question difference is noise.
- Failure attribution: if the gold chunk is not in the top-k and the model refuses, that is a **retrieval** failure; fix retrieval, not the prompt. Final recall@k <= first-stage candidate recall@N.

### Agent (`src/copilot/agent.py`)
- Agent = LLM + tools + loop + stop condition; control flow is decided by the model instead of a fixed pipeline. Tools: `search_papers(query, paper?)` (hybrid + rerank) and `final_answer(...)`. Max-step cap with a forced final answer.
- Agent vs fixed RAG: more capable (rephrases queries, restricts to a paper, splits comparisons) but ~3x tokens, less predictable, harder to debug.
- Tool arguments are **not guaranteed** to follow the schema unless the tool is `strict` (a run crashed with `KeyError: 'citations'`); see Strict tool use below. Validate tool inputs otherwise.
- Results on 20 questions, 3 runs: all 15 answerable questions answered correctly (RAG: 13/15), wrongly refused 0 (RAG: 1-2), unanswerable refused 5/5 in every run. Cost: ~2 searches/question, ~5.5-6k input tokens vs ~1.5-2k for RAG. Caveats: the agent sees more chunks than top-5; the set is small and was used while developing; agent queries benefit from the model's prior knowledge of the domain.

### Strict tool use
- `"strict": true` on a tool turns on **constrained decoding for the tool input**, the same mechanism as structured output: invalid tokens are masked (logit -> -inf), so required fields and types are guaranteed. It needs `additionalProperties: false`. We enabled it on `final_answer` after the `KeyError: 'citations'` crash. Without it, tool input is only "usually" valid, so validate it.

### How the agent rewrites queries
- There is no separate rewrite module. The `query` argument of `search_papers` is text the model generates (next-token prediction), like any other output. The user question is just context.
- Rewriting is adaptive: the previous search's results come back as an **observation** in `messages`, so the model can see the miss and write a better query (e.g. `NF4`, `normal float`, `information theoretically optimal`). It also uses its pretraining knowledge of the papers, which is a confounder; private documents would weaken this.
- Dedicated alternatives (one-shot, before retrieval): **query rewriting**, **multi-query**, **HyDE** (retrieve with a hypothetical answer), **query decomposition**. They are predictable and cheap; the agent version is flexible but costs steps and tokens and is not guaranteed to happen.

### Ablation: does the "rephrase" sentence matter? (`eval/ablate_rewrite.py`)
- **Ablation** = remove one component, measure the change. Prediction: without "rephrase the query, use technical terms", the agent searches less and rewrites less. **Result: no visible change.**

| | avg searches | questions with >1 search | consecutive-query Jaccard | gold retrieved | avg tokens |
|---|---|---|---|---|---|
| with hint | 1.90 | 18/20 | 0.20 | 15/15 | 5465 |
| no hint | 1.95 | 19/20 | 0.20 | 15/15 | 5414 |

- Rewriting seems to come mostly from the model, not that sentence. Likely driven by the observation of the first search, but the experiment does not separate that from prior knowledge.
- Caveats: n=20, one run per condition (a 0.05 difference is noise); the tool description and "Be efficient" still nudge the model; only behaviour was measured, not quality (gold is 15/15 in both, so the set cannot discriminate); the papers are famous. "No effect detected" is not "no effect".

---

## 8. Scaling the library from 5 to 59 papers

- **Stable ids matter.** Gold labels were chunk indices. Adding papers in alphabetical order would shift every index and silently invalidate the labels. Fix: build in manifest order (original five first, in their old order) and verify the first 348 chunks are identical to the old index. The check caught a real mistake: I had listed two of the five papers in the wrong order.
- **More distractors => every retriever degrades** (recall@5 hybrid 0.933 -> 0.833; dense most, BM25 least). The reranker still helped everywhere. This is the realistic setting a RAG system faces; 5 papers was easy mode.
- **Gold labels can go stale.** A question written for a small library can be legitimately answered by new documents; recall then undercounts. Labels need re-validation when the corpus changes. The same applies to "unanswerable" questions: I excluded papers that would answer them.
- **Prompt cost scales with library size** when the paper list sits in the system prompt (input tokens ~5.5k -> ~10.7k per question). Prompt caching would cut the repeated part.
- **Evaluation bug found again:** the judge's `max_tokens=500` truncated its JSON once answers got longer. Now 1500, plus an assertion on `stop_reason`.

## 9. A harder test set, and what a fair baseline does to a result

- **Gold as (paper, phrase)** instead of a chunk index: robust to re-chunking and to adding papers. Check each phrase exists in the corpus, and keep it specific (a phrase present in 40 chunks of a paper is a loose label).
- **Mislabelled "unanswerable" question:** "How does speculative decoding work?" looked out of scope, but one library paper (GQA) mentions it; all three systems answered, and were right. Lesson: verify unanswerable questions by searching the corpus, not by intuition.
- **Fair baseline changes the story.** The earlier claim "agent beats RAG" compared an agent that sees ~10 chunks with RAG that sees 5. With RAG top-10, the gap shrinks to about one question (single-paper correct: 9 -> 11 -> 12 for RAG-5, RAG-10, agent). The result that remains: on questions needing two papers, the agent retrieves both papers' evidence far more often (4/5 vs 1/5) because it can search once per paper. Always ask: what is the strongest simple baseline?
- **Two metrics, different strictness:** "all gold retrieved" is strict; the LLM judge's `correct` is lenient on multi-part questions. Report both.
- **Structured fields vs free text can disagree** (`answerable=true` while the text refuses). Decide which one is the source of truth.
- **Failure handling:** a tool call truncated by `max_tokens` has an incomplete input. Check `stop_reason` and fail loudly.

## 10. Vector databases and ANN

- **Exact kNN** costs O(N x d) per query (one dot product with every vector). Fine up to ~10k vectors; at 500k it was 9 ms/query in our test and keeps growing linearly. Tree indexes (KD-tree) stop helping in hundreds of dimensions (curse of dimensionality), so large systems use **ANN (approximate nearest neighbor)**: give up a little recall for sublinear search time.
- **HNSW** (Hierarchical Navigable Small World): vectors are nodes in a graph linked to near neighbors. Search = greedy walk toward the query, keeping a candidate list of size `ef`; upper layers are sparse with long links (coarse jumps, like a skip list), lower layers are dense. Knobs: `M` (links per node), `ef_construction`, `ef_search` (bigger = higher recall, slower). Costs: memory for the graph, slow build (67 s for 500k), updates/deletes are awkward.
- Other ANN families: **IVF** (cluster vectors, search only the nearest clusters), **PQ** (compress vectors into short codes), **LSH** (hash similar vectors into the same bucket); real systems combine them.
- **Vector database** = ANN index + persistence + metadata filtering + CRUD + (often) a server and scaling. FAISS is a library, not a database.
- Experiment lessons: at 4.6k vectors the database changed nothing (same recall, same latency); HNSW was *slower* than numpy below ~10k vectors. And a benchmark on pure random vectors gave recall 0.02, because random high-dimensional vectors have no neighbor structure: a bad benchmark can mislead as much as a bad system. Check recall before trusting a latency number.

---

## 11. From question answering to a research agent

- **QA vs research task.** A question has one short answer that one retrieval can usually find. A research task ("compare A, B, C, D") needs evidence from several places, a structure, and honesty about gaps. That is where an agent loop earns its cost. Measured: one top-10 retrieval covered 0.48 of the key points; a tool-use loop covered 0.96 (CI of the difference +0.38 to +0.56).
- **Workflow vs agent.** A *workflow* has code-defined control flow (plan -> research -> write -> verify). An *agent* lets the model choose the next step. Our `staged` pipeline is mostly a workflow with an agent inside each research step; `react` is a plain agent. Start with the simplest that works; add structure only where a measurement says it helps.
- **Patterns used:** *planner* (split the task into sub-questions), *researcher* (tool-use loop per sub-question), *writer* (compose only from the evidence pool), *critic / verifier* (check each cited sentence against its passages), *repair* (retrieve more evidence for flagged claims, then revise). Related published ideas: ReAct (reason + act interleaved), Reflexion (feedback turned into verbal reflection), Self-RAG (critique tokens).
- **Grounding rules in the prompt:** cite a passage id for every factual sentence; say explicitly when the library does not cover something. The gap tasks (DPO, Switch/Mixtral routing) test the second rule: all systems passed.

## 12. Evaluating an agent's output

- **Metrics for a long answer** (`eval/eval_research.py`): *citation support* (does the cited passage state the claim), *uncited factual sentences*, *key-point coverage*, *required-paper coverage*, *gap handling*, plus cost, calls, latency. Key points are tied to (paper, phrase) pairs checked against the corpus, so a miss is the system's fault, not an impossible task.
- **Judge hygiene:** the judge (Sonnet) is a different model from the writer (Haiku). The verifier inside the pipeline is the same family as the writer, so its leniency is a risk; the reported numbers come from the independent judge.
- **Paired comparison with a bootstrap over tasks.** Compare systems on the same task and run, average the differences within each task, then resample TASKS to get a 95% interval. (The same tasks are re-run, so runs are not independent; resampling runs would give intervals that are too narrow.) Report tasks better / tied / worse next to the mean. A CI that spans zero means 'not distinguishable', not 'equal'.
- **Ablation = remove one component.** `staged` vs `staged_noverify` isolates the verifier; `staged_noverify` vs `react` isolates planning. Result (3 runs x 15 tasks): planning +0.08 support [+0.03, +0.11], verifier +0.06 more [+0.02, +0.11] at about 1.5x the calls; the verifier lowers key-point coverage slightly (-0.03, CI touches zero) because deleting an unsupported claim can delete a key point.
- **Cost-quality frontier.** `react`: $0.003, support 0.75. `staged_r1`: $0.017, 0.90. `staged`: $0.022, 0.89. Report all; the right choice depends on how costly an unsupported claim is. A second repair round was not distinguishable from one (+0.01, CI spans zero) and cost 20% more.
- **Failures are data.** Runs that failed (output truncated, API limit) were recorded with their error, retried, and counted, not silently dropped. When two runs differ only because the token limit changed, say so.
- **Variance:** repeated runs of the same system on the same tasks differed (react support 0.83, 0.70, 0.73; staged 0.93, 0.90, 0.85). One run is not a result.

## 13. LangGraph

- **Core ideas:** a `StateGraph` over a typed `State`; *nodes* are functions that return partial state updates; *edges* define order; *conditional edges* choose the next node from the state (our verify -> repair loop); a *reducer* (`Annotated[list, fn]`) defines how parallel updates merge (our evidence pool); `Send` fans out one task per item (one research branch per sub-question); a *checkpointer* saves state after each step; `interrupt()` pauses the run and `Command(resume=...)` continues it (human approval of the plan).
- **What it bought here:** same quality (support 0.92 vs 0.89, difference +0.03 with a CI spanning zero), latency 59 s -> 38 s from parallel research, and a pause/resume point with no extra code. What it did not buy: better answers. The step functions are shared with the hand-written version, so the comparison measures only orchestration.
- **When a framework is worth it:** branching, parallel fan-out, long-running state that must survive a crash or wait for a human. For a straight loop, the hand-written version (about 30 lines) is clearer. Writing it by hand first is how you know what the framework is doing for you.
- **Thread-safety gotcha:** parallel nodes run in threads; shared counters need a lock (`Report.add_counts`), and any shared non-thread-safe resource (here, torch inference) needs serialising.

## 14. Production engineering lessons from this stage

- **Concurrency bug found by a test, not by thought.** Four threads calling the encoder and reranker hung for minutes at ~400% CPU. A lock around retrieval fixed it (6 s for the same test). This also affected the first FastAPI version, which ran the agent in a thread pool: two concurrent requests could have hung the server.
- **Check `stop_reason`.** A response cut at `max_tokens` may contain an incomplete tool call. Raise a clear error that names the step; do not let it surface as a `KeyError`. Then size the limits from real outputs (reports with 5-6 items needed 4,000, not 1,500).
- **Serialization:** numpy integers are not JSON serializable; cast at the boundary (`int(c)`) rather than patching every `json.dumps`.
- **Tests that earn their place:** id-stability of the paper manifest (a past bug), citation parsing, API auth and "only return sources that were really retrieved", and that upstream errors become a 502 without leaking internals. Unit tests need no network or key (a placeholder key is set in `conftest.py` because the client is created at import).
- **Secrets and abuse:** a public URL that spends model credits needs a gate. The API requires an `x-api-key` header (constant-time compare). On AWS the keys live in SSM Parameter Store (SecureString) and are read at cold start, never baked into the image or committed.
- **Serverless fit:** low, bursty traffic -> Lambda container image behind a Function URL, with the Lambda Web Adapter so the FastAPI app is unchanged; index and models baked into the image so there is no setup or network at runtime; arm64 (cheaper, matches Apple Silicon builds); idle cost near zero. The cost is a cold start of several seconds. An always-on Fargate task would cost more per month than the whole budget.
- **Observability without a vendor:** one CloudWatch Embedded Metric Format line per request (cost, latency, calls, tokens). Add a traffic alarm and a budget alert.
- **Within a company on AWS** the usual model access is Amazon Bedrock (traffic and billing stay in the account); the change is the client class.
- **Operational reality:** the Anthropic account has a monthly API usage limit. Hitting it returns a 400 until the reset date. Budget the evaluation (`judge cost this invocation` is printed), keep results cached so reruns skip finished work, and make failure records distinguishable from results.

## 15. Agent vocabulary (quick reference)

| Term | Meaning |
|---|---|
| Tool / function calling | A function the model can request; your code runs it and returns the result (`search_papers`) |
| Agent loop | Call the model, run any tools it requests, feed results back, stop on a final answer or a step cap |
| MCP (Model Context Protocol) | A standard way to expose tools and data sources to any compatible application |
| Skill | A folder with a `SKILL.md` (name + description + instructions, optional scripts) loaded on demand; knowledge about *how* to do a task, not a callable function. Progressive disclosure: only the description is always in context |
| Subagent / multi-agent | Separate model loops with their own context, used to isolate reading-heavy work or specialise roles |
| Human-in-the-loop | The run pauses at a checkpoint for a person to approve or edit (our plan approval) |
| Checkpointing | Saving graph state after each step so a run can resume or be inspected |
| Self-verification / critic | A step that checks the system's own output against evidence (our citation verifier) |
| Prompt injection | Retrieved text that tries to give the model instructions; treat retrieved content as data, never as instructions |

## 16. What the error analysis and the decomposition taught (research-brief evaluation)

- **Read the failures, not just the score.** Sampling the sentences the judge called unsupported showed that many were uncited connective or limitation sentences ("The library's evidence does not include a cost comparison...") that were not wrong. A single `support` number was hiding two different things.
- **Decompose a composite metric.** `support` = (share of factual sentences that carry a citation) x (share of cited sentences that are actually supported). Baselines: about 76-79% cited, about 94% of those supported. With planning and verification: 88-93% cited, about 98% supported. Both parts improved, and they have different fixes (prompting for coverage, verification for accuracy).
- **Your own verifier is not ground truth.** The in-pipeline Haiku checker flagged 35-45% of sentences and still ~40% after repair, while the independent judge found far fewer unsupported sentences. A critic from the same model family as the writer is noisy; use it to drive repair, measure with something independent, and say so.
- **Cost-aware design:** removing the second repair round cut cost by 20% with no measurable loss. Ask of every extra loop iteration: what does the measurement say it buys? Choosing the default after seeing the data is selection on the evaluation set: disclose it.
- **Validate the judge with a human.** `eval/audit_sheet.py` samples judged sentences without the verdict; a person labels them; `eval/audit_score.py` reports agreement and Cohen's kappa (agreement beyond chance). "Judge too lenient" and "judge too strict" are different failures and are reported separately.
- **Practical:** write results to disk as each job finishes (not at the end), treat failed runs as data (recorded, retried), keep a spending limit, and print the judge's cost so a budget can be managed.
