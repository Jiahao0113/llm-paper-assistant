"""Retrieval pipeline: hybrid (dense + BM25 via RRF) -> cross-encoder rerank -> top-k."""
import numpy as np

from .bm25 import BM25, rrf
from .rerank import rerank
from .retrieval import DenseIndex


class Retriever:
    def __init__(self, top_n: int = 30):
        self.idx = DenseIndex()
        self.idx.load()
        self.top_n = top_n
        self.texts = [f"{c.section}. {c.text}" for c in self.idx.chunks]  # same text we embedded
        self.bm25 = BM25(self.texts)

    def _dense(self, q):
        return list(np.argsort(self.idx.matrix @ self.idx.model.encode(q, normalize_embeddings=True))[::-1])

    def _bm25(self, q):
        return list(np.argsort(self.bm25.scores(q))[::-1])

    def retrieve_ids(self, q: str, k: int = 5, paper: str | None = None):
        cands = rrf([self._dense(q)[:100], self._bm25(q)[:100]])
        if paper:
            cands = [c for c in cands if self.idx.chunks[c].paper == paper]
        cands = cands[: self.top_n]
        return rerank(q, cands, [self.texts[c] for c in cands])[:k]

    def retrieve(self, q: str, k: int = 5):
        return [(0.0, self.idx.chunks[c]) for c in self.retrieve_ids(q, k)]  # score slot kept for compatibility with rag.build_prompt
