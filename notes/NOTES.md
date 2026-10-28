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
