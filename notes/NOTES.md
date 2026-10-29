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
