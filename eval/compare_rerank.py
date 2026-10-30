import json
from pathlib import Path

import numpy as np

from eval.compare_retrievers import hybrid_order, idx, rows as rows_v1
from eval.compare_v2 import rows as rows_v2
from src.copilot.rerank import rerank

TOP_N = 30


def hybrid_rerank_order(q):
    cands = hybrid_order(q)[:TOP_N]
    texts = [f"{idx.chunks[c].section}. {idx.chunks[c].text}" for c in cands]
    return rerank(q, cands, texts) + hybrid_order(q)[TOP_N:]  # rest keeps hybrid order


def report(name, rows, gold_of, q_of):
    out = {}
    for label, fn in [("hybrid", hybrid_order), (f"hybrid+rerank(top{TOP_N})", hybrid_rerank_order)]:
        rk = []
        for r in rows:
            pos = {c: i for i, c in enumerate(fn(q_of(r)))}
            rk.append(min(pos.get(g, 10**6) for g in gold_of(r)) + 1)
        out[label] = rk
    print(f"\n{name} (n={len(rows)})")
    print(f"{'':26}{'r@1':>7}{'r@3':>7}{'r@5':>7}{'r@10':>7}{'MRR':>8}")
    for label, rk in out.items():
        print(f"{label:26}" + "".join(f"{np.mean([x <= k for x in rk]):>7.3f}" for k in (1, 3, 5, 10)) + f"{np.mean([1/x for x in rk]):>8.3f}")


report("v1 synthetic", rows_v1, lambda r: [r["gold_chunk"]], lambda r: r["question"])
report("v2 hand-written", rows_v2, lambda r: r["gold"], lambda r: r["q"])
