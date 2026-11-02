"""v3 set on the 59-paper library: fixed RAG with top-5, fixed RAG with top-10, and the agent, same judge.
Top-10 RAG controls for the confound that the agent sees more chunks than top-5 RAG."""
import json
from pathlib import Path
from types import SimpleNamespace

from eval.eval_generation import judge
from src.copilot.agent import run_agent
from src.copilot.pipeline import Retriever
from src.copilot.rag import answer

data = json.loads(Path("eval/retrieval_set_v3.json").read_text())
r = Retriever()
chunks = r.idx.chunks


def gold_hit(g, ids):
    return any(chunks[i].paper == g["paper"] and g["phrase"].lower() in chunks[i].text.lower() for i in ids)


def reference(gold):
    """Up to 2 chunks per gold entry, for the judge's reference passages."""
    out = []
    for g in gold:
        out += [c for c in chunks if c.paper == g["paper"] and g["phrase"].lower() in c.text.lower()][:2]
    return out


items = [(x["type"], x["q"], x["gold"]) for x in data["answerable"]] + [("unanswerable", q, []) for q in data["unanswerable"]]
results = []
for typ, q, gold in items:
    for system in ("rag_k5", "rag_k10", "agent"):
        if system == "agent":
            out = run_agent(r, q)
            ids, ans, tokens, nsearch = out.retrieved_ids, SimpleNamespace(answerable=out.answerable, answer=out.answer), out.input_tokens, len(out.searches)
            hits = [(0.0, chunks[i]) for i in ids]
        else:
            hits = r.retrieve(q, k=int(system[5:]))
            ids = [chunks.index(c) for _, c in hits]
            ans, _, usage = answer(q, hits)
            tokens, nsearch = usage.input_tokens, 1
        v = judge(q, hits, ans, reference(gold))
        results.append({"type": typ, "system": system, "q": q, "gold_covered": sum(gold_hit(g, ids) for g in gold), "n_gold": len(gold),
                        "model_answerable": ans.answerable, "faithful": v.faithful, "correct": v.correct, "tokens": tokens,
                        "searches": nsearch, "answer": ans.answer, "judge_reasoning": v.reasoning})
    print("done:", q[:60], flush=True)
Path("eval/v3_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))

print(f"\n{'type':13s}{'system':9s}{'n':>3s}{'all gold':>9s}{'correct':>9s}{'faithful':>9s}{'refused':>8s}{'tokens':>8s}")
for typ in ("single", "cross", "unanswerable"):
    for system in ("rag_k5", "rag_k10", "agent"):
        R = [x for x in results if x["type"] == typ and x["system"] == system]
        gold_all = "-" if typ == "unanswerable" else f"{sum(x['gold_covered'] == x['n_gold'] for x in R)}/{len(R)}"
        refused = sum(not x["model_answerable"] for x in R)
        print(f"{typ:13s}{system:9s}{len(R):3d}{gold_all:>9s}{sum(x['correct'] for x in R):>6d}/{len(R):<2d}{sum(x['faithful'] for x in R):>6d}/{len(R):<2d}{refused:>8d}{sum(x['tokens'] for x in R)//len(R):>8d}")
