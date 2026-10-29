"""Score a retriever on eval/retrieval_set.json: recall@k and MRR."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.copilot.retrieval import DenseIndex

KS = [1, 3, 5, 10, 20]


def evaluate(search_ranks, rows):
    """search_ranks(question) -> list of chunk ids, best first."""
    ranks, per_paper = [], defaultdict(list)
    for r in rows:
        order = search_ranks(r["question"])
        rank = order.index(r["gold_chunk"]) + 1 if r["gold_chunk"] in order else None
        ranks.append(rank)
        per_paper[r["paper"]].append(rank)
    return ranks, per_paper


def summarize(ranks):
    out = {f"recall@{k}": np.mean([r is not None and r <= k for r in ranks]) for k in KS}
    out["MRR"] = np.mean([1 / r if r else 0 for r in ranks])
    return out


if __name__ == "__main__":
    rows = json.loads(Path("eval/retrieval_set.json").read_text())
    idx = DenseIndex()
    idx.load()

    def dense_order(q):
        scores = idx.matrix @ idx.model.encode(q, normalize_embeddings=True)
        return list(np.argsort(scores)[::-1])

    ranks, per_paper = evaluate(dense_order, rows)
    print("Dense (all-MiniLM-L6-v2), n =", len(rows))
    for k, v in summarize(ranks).items():
        print(f"  {k}: {v:.3f}")
    print("  median rank of gold chunk:", int(np.median([r for r in ranks])))
    print("\nrecall@5 by paper:")
    for p, rs in per_paper.items():
        print(f"  {p}: {np.mean([r <= 5 for r in rs]):.2f}")
    worst = sorted(zip(ranks, rows), key=lambda x: -x[0])[:3]
    print("\nworst 3 (rank, question):")
    for rk, r in worst:
        print(f"  rank {rk}: [{r['paper']} p{r['page']}] {r['question'][:110]}")
