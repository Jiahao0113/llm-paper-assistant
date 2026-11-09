"""Evaluate research-brief systems. Metrics per report:
  support      - share of factual sentences whose claim is supported by the passages they cite (judge: Sonnet, not the Haiku generator)
  uncited      - factual sentences with no citation at all
  bad_cites    - citations to passage ids that do not exist
  key_points   - share of the task's key points the brief states
  papers       - share of required papers cited at least once
  gap tasks    - acknowledged the missing coverage / fabricated details about the missing papers
Usage: python -m eval.eval_research [system ...] [--runs N]; results accumulate in eval/research_results.json."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from src.copilot.pipeline import Retriever
from src.copilot.research import cited_ids, evidence_block, split_sentences
from src.copilot.systems import SYSTEMS

load_dotenv()
client = anthropic.Anthropic()
JUDGE = "claude-sonnet-5-5"
PRICE_IN, PRICE_OUT = 2.0 / 1e6, 10.0 / 1e6
OUT = Path("eval/research_results.json")
WORKERS = int(os.environ.get("WORKERS", "1"))  # >1 ran into a silent crash, see notes


class SentenceVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factual: list[int]    # indexes of sentences that state a factual claim about a paper (not headings, transitions, or "the library does not cover X")
    supported: list[int]  # subset of factual: claim is stated by, or clearly implied by, the passages that sentence cites


class KeyPoints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    covered: list[int]    # indexes of the key points the brief states (wording may differ)


class GapVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    acknowledged_gap: bool  # the brief says the requested paper/topic is not covered by the library
    fabricated: bool        # the brief states specifics about the missing paper/topic as if known (even if cited to unrelated passages)
    reasoning: str


calls = {"in": 0, "out": 0}


def ask(system: str, user: str, schema):
    resp = client.messages.create(model=JUDGE, max_tokens=2500, thinking={"type": "between_tools"}, system=system,
                                  messages=[{"role": "user", "content": user}],
                                  output_config={"format": {"type": "json_schema", "schema": schema.model_json_schema()}})
    assert resp.stop_reason != "max_tokens", "judge output truncated"
    calls["in"] += resp.usage.input_tokens
    calls["out"] += resp.usage.output_tokens
    return schema.model_validate_json("".join(b.text for b in resp.content if b.type == "text"))


def score(r: Retriever, task: dict, text: str) -> dict:
    n_chunks = len(r.idx.chunks)
    sents = split_sentences(text)
    blocks = []
    for n, s in enumerate(sents):
        ids = [i for i in cited_ids(s) if i < n_chunks]
        blocks.append(f"({n}) {s}\n   cited passages:\n" + (evidence_block(r, ids) if ids else "   (none)"))
    v = ask("You grade citation support in a research brief. Be strict: a sentence is supported only if its cited passages state or clearly imply the claim. A sentence with no cited passage that makes a factual claim is not supported.",
            "\n\n".join(blocks), SentenceVerdicts)
    factual = [i for i in set(v.factual) if 0 <= i < len(sents)]
    supported = [i for i in set(v.supported) if i in factual]
    all_ids = cited_ids(text)
    out = {"n_sentences": len(sents), "n_factual": len(factual), "support": len(supported) / len(factual) if factual else None,
           "uncited": sum(1 for i in factual if not cited_ids(sents[i])), "bad_cites": sum(1 for i in all_ids if i >= n_chunks),
           "unsupported_sentences": [sents[i] for i in factual if i not in supported],
           # every factual sentence with the judge's verdict: the raw material for the human audit (eval/audit_sheet.py)
           "sentence_verdicts": [{"text": sents[i], "cited": [c for c in cited_ids(sents[i]) if c < n_chunks], "supported": i in supported} for i in sorted(factual)]}
    kps = task["key_points"]
    if kps:
        listing = "\n".join(f"({n}) {k['point']}" for n, k in enumerate(kps))
        kv = ask("You check which key points a research brief states. A point counts if the brief says it, even in different words; vague or wrong statements do not count.",
                 f"Key points:\n{listing}\n\nBrief:\n{text}", KeyPoints)
        out["key_points"] = len({i for i in kv.covered if 0 <= i < len(kps)}) / len(kps)
    req = task["required_papers"]
    if req:
        cited_papers = {r.idx.chunks[i].paper for i in all_ids if i < n_chunks}
        out["papers"] = sum(p in cited_papers for p in req) / len(req)
    if task["type"] == "gap":
        g = ask("You check how a research brief handles a gap in its source library.", f"Task: {task['q']}\nKnown gap: {task['gap_note']}\n\nBrief:\n{text}", GapVerdict)
        out.update(acknowledged_gap=g.acknowledged_gap, fabricated=g.fabricated, gap_reasoning=g.reasoning)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1
    args = [a for a in args if a not in {str(runs)}]
    systems = [a for a in args if a in SYSTEMS] or list(SYSTEMS)
    tasks = json.loads(Path("eval/research_tasks.json").read_text())["tasks"]
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    results = [x for x in results if "error" not in x]  # failed runs are retried, not kept as "done"
    done = {(x["task"], x["system"], x["run"]) for x in results}
    r = Retriever()
    jobs = [(t, s, k) for k in range(runs) for t in tasks for s in systems if (t["id"], s, k) not in done]

    def job(j):
        t, s, k = j
        try:
            rep = SYSTEMS[s](r, t["q"])
            sc = score(r, t, rep.text)
        except Exception as e:  # a failed run is a result (counted below), not a reason to stop the evaluation
            print("FAILED", t["id"], s, k, type(e).__name__, str(e)[:120], flush=True)
            return {"task": t["id"], "type": t["type"], "system": s, "run": k, "error": f"{type(e).__name__}: {e}"[:300]}
        print("done", t["id"], s, k, flush=True)
        return {"task": t["id"], "type": t["type"], "system": s, "run": k, "text": rep.text, "cost": rep.cost, "llm_calls": rep.llm_calls,
                "seconds": rep.seconds, "input_tokens": rep.input_tokens, "output_tokens": rep.output_tokens, "trace": rep.trace, **sc}

    with ThreadPoolExecutor(WORKERS) as ex:
        for fut in as_completed([ex.submit(job, j) for j in jobs]):  # save each report as soon as it finishes
            results.append(fut.result())
            OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=int))
    print(f"judge cost this invocation: ${calls['in'] * PRICE_IN + calls['out'] * PRICE_OUT:.2f}")
    summarize(results)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def per_run(R, key):
    """Mean of `key` within each run, so the spread between repeated runs is visible."""
    runs = sorted({x["run"] for x in R})
    return [mean([x.get(key) for x in R if x["run"] == k]) for k in runs]


def summarize(results):
    print(f"\n{'system':16s}{'reports':>8s}{'support':>9s}{'uncited':>9s}{'keypts':>8s}{'papers':>8s}{'calls':>7s}{'$/task':>9s}{'sec':>6s}  failed")
    for s in SYSTEMS:
        R = [x for x in results if x["system"] == s and x["type"] != "gap" and "error" not in x]
        failed = sum(1 for x in results if x["system"] == s and "error" in x)
        if R:
            print(f"{s:16s}{len(R):8d}{mean([x['support'] for x in R]):9.3f}{mean([x['uncited'] for x in R]):9.2f}{mean([x.get('key_points') for x in R]):8.3f}"
                  f"{mean([x.get('papers') for x in R]):8.3f}{mean([x['llm_calls'] for x in R]):7.1f}{mean([x['cost'] for x in R]):9.4f}{mean([x['seconds'] for x in R]):6.0f}  {failed}")
            if len({x["run"] for x in R}) > 1:
                print(f"{'':16s}  per-run support {', '.join(f'{v:.3f}' for v in per_run(R, 'support'))} | key points {', '.join(f'{v:.3f}' for v in per_run(R, 'key_points'))}")
    print("\ngap tasks (acknowledged / fabricated, all runs):")
    for s in SYSTEMS:
        G = [x for x in results if x["system"] == s and x["type"] == "gap" and "error" not in x]
        if G:
            print(f"  {s:16s} acknowledged {sum(x['acknowledged_gap'] for x in G)}/{len(G)}, fabricated {sum(x['fabricated'] for x in G)}/{len(G)}")


if __name__ == "__main__":
    main()
