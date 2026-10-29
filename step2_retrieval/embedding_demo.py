import numpy as np
from sentence_transformers import SentenceTransformer

# A small pretrained embedding model (~90MB, 384 dimensions), runs locally and free.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "QLoRA reduces memory usage during fine-tuning.",      # 0
    "This method cuts GPU RAM consumption when training.",  # 1  same meaning, different words
    "The recipe needs two cups of flour and one egg.",      # 2  unrelated
    "Memory is a faculty of the human brain.",              # 3  shares the word "memory", different meaning
]

emb = model.encode(sentences)  # numpy array, shape (4, 384): one vector per sentence
print("embedding shape:", emb.shape)
print("first 5 numbers of sentence 0:", np.round(emb[0][:5], 3))
print("vector length (L2 norm):", np.round(np.linalg.norm(emb[0]), 3))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\ncosine similarity with sentence 0:")
for i in range(1, len(sentences)):
    print(f"  [{i}] {cosine(emb[0], emb[i]):.3f}  {sentences[i]}")

# Now do a tiny search: which sentence best matches a query?
query = "How can I train a large model with less GPU memory?"
q = model.encode(query)
scores = [cosine(q, e) for e in emb]
print(f"\nquery: {query!r}")
for i in np.argsort(scores)[::-1]:
    print(f"  {scores[i]:.3f}  {sentences[i]}")
