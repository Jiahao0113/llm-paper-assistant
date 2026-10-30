import json
from pathlib import Path

import numpy as np

from eval.compare_retrievers import bm25_order, dense_order, hybrid_order
from eval.run_retrieval import KS
from src.copilot.bm25 import tokenize

data = json.loads(Path("eval/retrieval_set_v2.json").read_text())
rows = data["answerable"]


def ranks(fn):
    out = []
    for r in rows:
        order = list(fn(r["q"]))
        pos = {c: i for i, c in enumerate(order)}
        out.append(min(pos.get(g, 10**6) for g in r["gold"]) + 1)  # best rank among gold chunks; absent = never retrieved
    return out


print(f"v2 hand-written set, n={len(rows)} (hit on ANY gold chunk counts)")
print(f"{'':14}" + "".join(f"{'recall@'+str(k):>11}" for k in KS) + f"{'MRR':>8}")
res = {}
for name, fn in [("dense", dense_order), ("bm25", bm25_order), ("hybrid (RRF)", hybrid_order)]:
    rk = ranks(fn)
    res[name] = rk
    print(f"{name:14}" + "".join(f"{np.mean([x <= k for x in rk]):>11.3f}" for k in KS) + f"{np.mean([1/x for x in rk]):>8.3f}")
print("\nper-question rank (dense / bm25 / hybrid):")
for r, d, b, h in zip(rows, res["dense"], res["bm25"], res["hybrid (RRF)"]):
    print(f"  {d:>3} {b:>3} {h:>3}  {r['q'][:80]}")
