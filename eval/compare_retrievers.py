import json
from pathlib import Path

import numpy as np

from eval.run_retrieval import KS, evaluate, summarize
from src.copilot.bm25 import BM25, rrf
from src.copilot.retrieval import DenseIndex

rows = json.loads(Path("eval/retrieval_set.json").read_text())
idx = DenseIndex()
idx.load()
bm25 = BM25([f"{c.section}. {c.text}" for c in idx.chunks])  # same text we embedded


def dense_order(q):
    return list(np.argsort(idx.matrix @ idx.model.encode(q, normalize_embeddings=True))[::-1])


def bm25_order(q):
    return list(np.argsort(bm25.scores(q))[::-1])


def hybrid_order(q):
    return rrf([dense_order(q)[:100], bm25_order(q)[:100]])


results = {}
for name, fn in [("dense", dense_order), ("bm25", bm25_order), ("hybrid (RRF)", hybrid_order)]:
    ranks, _ = evaluate(fn, rows)
    results[name] = summarize(ranks)

print(f"{'':14}" + "".join(f"{'recall@'+str(k):>11}" for k in KS) + f"{'MRR':>8}")
for name, m in results.items():
    print(f"{name:14}" + "".join(f"{m[f'recall@{k}']:>11.3f}" for k in KS) + f"{m['MRR']:>8.3f}")
