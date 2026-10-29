"""Dense retrieval over all chunks: embed once, cache to disk, brute-force cosine search."""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import Chunk, chunk_paper

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = Path("data/index")


class DenseIndex:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def build(self, pdf_dir="data/raw"):
        for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
            self.chunks += chunk_paper(pdf)
        # Prepend the section name so an isolated paragraph carries some context.
        texts = [f"{c.section}. {c.text}" for c in self.chunks]
        self.matrix = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(CACHE / "matrix.npy", self.matrix)
        (CACHE / "chunks.json").write_text(json.dumps([c.__dict__ for c in self.chunks]))

    def load(self):
        self.matrix = np.load(CACHE / "matrix.npy")
        self.chunks = [Chunk(**d) for d in json.loads((CACHE / "chunks.json").read_text())]

    def search(self, query: str, k: int = 3):
        q = self.model.encode(query, normalize_embeddings=True)
        scores = self.matrix @ q
        return [(float(scores[i]), self.chunks[i]) for i in np.argsort(scores)[::-1][:k]]
