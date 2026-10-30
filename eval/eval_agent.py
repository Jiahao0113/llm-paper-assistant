"""Same 20 questions and the same judge as eval_generation.py, but answered by the agent."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eval.eval_generation import judge
from src.copilot.agent import run_agent
from src.copilot.pipeline import Retriever

data = json.loads(Path("eval/retrieval_set_v2.json").read_text())
r = Retriever()
items = [(x["q"], x["gold"]) for x in data["answerable"]] + [(q, []) for q in data["unanswerable"]]
results = []
for q, gold in items:
    out = run_agent(r, q)
    hits = [(0.0, r.idx.chunks[i]) for i in out.retrieved_ids]
    ans = SimpleNamespace(answerable=out.answerable, answer=out.answer)
    v = judge(q, hits, ans, [r.idx.chunks[g] for g in gold])
    results.append({"q": q, "answerable_q": bool(gold), "retrieved_gold": any(g in out.retrieved_ids for g in gold),
                    "model_answerable": out.answerable, "faithful": v.faithful, "correct": v.correct,
                    "steps": out.steps, "tokens": out.input_tokens, "searches": out.searches,
                    "hit_step_limit": out.hit_step_limit, "answer": out.answer, "judge_reasoning": v.reasoning})
Path("eval/agent_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))

A = [x for x in results if x["answerable_q"]]
U = [x for x in results if not x["answerable_q"]]
print(f"answerable (n={len(A)}): gold retrieved {sum(x['retrieved_gold'] for x in A)}, correct {sum(x['correct'] for x in A)}, "
      f"faithful {sum(x['faithful'] for x in A)}, wrongly refused {sum(not x['model_answerable'] for x in A)}")
print(f"unanswerable (n={len(U)}): correctly refused {sum(x['correct'] for x in U)}, faithful {sum(x['faithful'] for x in U)}")
print(f"avg searches {np.mean([len(x['searches']) for x in results]):.1f}, avg steps {np.mean([x['steps'] for x in results]):.1f}, "
      f"avg input tokens {np.mean([x['tokens'] for x in results]):.0f}, hit step limit {sum(x['hit_step_limit'] for x in results)}")
