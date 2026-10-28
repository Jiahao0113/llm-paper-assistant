import tiktoken

# Load a pretrained BPE vocabulary (used by GPT-3.5/4). Claude's tokenizer differs, but the idea is the same.
enc = tiktoken.get_encoding("cl100k_base")

texts = [
    "Hello world",
    "unbelievable",
    "你好，世界",
    "def foo(x): return x",
    "strawberry",
]

for text in texts:
    ids = enc.encode(text)                    # text -> list of token IDs (what the model actually sees)
    pieces = [enc.decode([i]) for i in ids]   # decode each ID alone to see which text chunk it covers
    print(f"{text!r}")
    print(f"  ids:    {ids}")
    print(f"  pieces: {pieces}")
    print(f"  chars={len(text)}  tokens={len(ids)}\n")

print("vocab size:", enc.n_vocab)
