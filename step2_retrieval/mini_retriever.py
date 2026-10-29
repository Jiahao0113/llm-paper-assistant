import numpy as np
from sentence_transformers import SentenceTransformer


class MiniRetriever:
    """Brute-force vector search: what a vector database does, minus the indexing tricks."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[str] = []
        self.matrix = None  # shape (num_chunks, dim); each row is one normalized chunk vector

    def add(self, chunks: list[str]):
        self.chunks += chunks
        vecs = self.model.encode(chunks, normalize_embeddings=True)  # unit length -> dot product == cosine
        self.matrix = vecs if self.matrix is None else np.vstack([self.matrix, vecs])

    def search(self, query: str, k: int = 3):
        q = self.model.encode(query, normalize_embeddings=True)  # shape (dim,)
        scores = self.matrix @ q                                 # one matrix-vector product scores ALL chunks
        top = np.argsort(scores)[::-1][:k]                       # indices of the k highest scores
        return [(float(scores[i]), self.chunks[i]) for i in top]


if __name__ == "__main__":
    r = MiniRetriever()
    r.add([
        "LoRA freezes the pretrained weights and trains small low-rank matrices injected into attention layers.",
        "QLoRA lowers GPU memory needs during training by quantizing the base model to 4 bits.",
        "The Transformer replaces recurrence with self-attention, allowing parallel training over the sequence.",
        "BPE builds a subword vocabulary by repeatedly merging the most frequent adjacent pair of symbols.",
        "RLHF fine-tunes a language model using a reward model trained on human preference comparisons.",
        "Dropout randomly zeroes activations during training to reduce overfitting.",
        "Flash attention reduces memory traffic by computing attention in tiles that fit in on-chip SRAM.",
        "Beam search keeps the k most probable partial sequences at each decoding step.",
    ])
    for query in [
        "How to shrink the footprint of fine-tuning?",
        "How do models learn from human feedback?",
        "How is a tokenizer vocabulary built?",
    ]:
        print(f"\nQ: {query}")
        for score, chunk in r.search(query):
            print(f"  {score:.3f}  {chunk[:80]}")
