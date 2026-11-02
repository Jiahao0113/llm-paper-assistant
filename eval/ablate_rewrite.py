"""Ablation: does the 'rephrase / use technical terms' sentence in the system prompt change agent behaviour?
Behaviour only (no judge): number of searches, how different consecutive queries are, whether the paper filter is used."""
import json
from pathlib import Path

import numpy as np

from src.copilot import agent
from src.copilot.pipeline import Retriever

HINT = """You may search several times: rephrase the query, use the technical terms
a paper would use, restrict to one paper, or split a comparison question into one search per paper."""
assert HINT in agent.SYSTEM
CONDITIONS = {"with_hint": agent.SYSTEM, "no_hint": agent.SYSTEM.replace(HINT, "Use search_papers to find evidence.")}


def jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / len(a | b)


data = json.loads(Path("eval/retrieval_set_v2.json").read_text())
items = [(x["q"], x["gold"]) for x in data["answerable"]] + [(q, []) for q in data["unanswerable"]]
r = Retriever()
report = {}
for name, system in CONDITIONS.items():
    agent.SYSTEM = system
    rows = []
    for q, gold in items:
        out = agent.run_agent(r, q)
        qs = [s["query"] for s in out.searches]
        rows.append({"q": q, "queries": qs, "papers": [s.get("paper") for s in out.searches],
                     "gold_hit": any(g in out.retrieved_ids for g in gold), "answerable_q": bool(gold),
                     "tokens": out.input_tokens})
    report[name] = rows
    n = [len(x["queries"]) for x in rows]
    multi = [x for x in rows if len(x["queries"]) > 1]
    sims = [jaccard(x["queries"][i], x["queries"][i + 1]) for x in multi for i in range(len(x["queries"]) - 1)]
    A = [x for x in rows if x["answerable_q"]]
    print(f"{name:9s} searches avg {np.mean(n):.2f} | questions with >1 search {len(multi)}/{len(rows)} | "
          f"consecutive-query Jaccard {np.mean(sims) if sims else float('nan'):.2f} | "
          f"paper filter used {sum(p is not None for x in rows for p in x['papers'])}/{sum(n)} | "
          f"gold retrieved {sum(x['gold_hit'] for x in A)}/{len(A)} | avg tokens {np.mean([x['tokens'] for x in rows]):.0f}")
Path("eval/ablate_rewrite_results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
