"""BM25 keyword retrieval, written from scratch."""
import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]        # term frequencies per doc
        self.len = [sum(c.values()) for c in self.tf]
        self.avgdl = sum(self.len) / len(docs)
        df = Counter(t for c in self.tf for t in c)           # in how many docs does each term appear
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    norm = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / norm
            out.append(s)
        return out


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion over several best-first rankings of chunk ids."""
    fused = Counter()
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] += 1 / (k + rank)
    return [d for d, _ in fused.most_common()]
