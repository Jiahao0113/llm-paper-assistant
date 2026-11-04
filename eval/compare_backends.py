"""numpy brute force (exact) vs Chroma/HNSW (approximate) for the dense stage: agreement and end-to-end retrieval quality."""
import json
import time

import numpy as np

from src.copilot.pipeline import Retriever

data = json.load(open("eval/retrieval_set.json"))
qs = [x["question"] for x in data]
gold = [x["gold_chunk"] for x in data]
exact, ann = Retriever(), Retriever(dense_backend="chroma")


def timed(fn, qs):
    t = time.perf_counter()
    out = [fn(q) for q in qs]
    return out, (time.perf_counter() - t) / len(qs) * 1000


E, t_e = timed(exact._dense, qs)
A, t_a = timed(ann._dense, qs)
for k in (10, 100):
    ov = np.mean([len(set(e[:k]) & set(a[:k])) / k for e, a in zip(E, A)])
    print(f"dense stage: top-{k} overlap exact vs HNSW = {ov:.3f}")
print(f"dense stage latency (includes query encoding): exact {t_e:.1f} ms, chroma {t_a:.1f} ms per query")
for name, r in (("numpy exact", exact), ("chroma HNSW", ann)):
    ids = [r.retrieve_ids(q, k=5) for q in qs]
    print(f"end-to-end hybrid+rerank, {name}: recall@5 = {np.mean([g in i for g, i in zip(gold, ids)]):.3f}, "
          f"recall@1 = {np.mean([i[0] == g for g, i in zip(gold, ids)]):.3f}")
