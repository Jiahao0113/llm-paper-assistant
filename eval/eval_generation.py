"""End-to-end RAG evaluation with an LLM judge (a different model from the generator)."""
import json
from pathlib import Path

from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, ConfigDict

from src.copilot.pipeline import Retriever
from src.copilot.rag import answer

load_dotenv()
client = anthropic.Anthropic()
JUDGE_MODEL = "claude-sonnet-5-5"  # generator is Haiku; a different model reduces self-preference bias

JUDGE_PROMPT = """You are grading a RAG system's answer.

Question: {question}

Retrieved passages given to the system:
{retrieved}

{reference}

System answer (answerable flag = {flag}):
{ans}

Grade two things:
1. faithful: true only if EVERY factual claim in the answer is supported by the retrieved passages. Claims from outside knowledge count as unsupported. A refusal / "not enough information" answer is faithful.
2. correct: {correct_rule}
First write brief reasoning, then the verdicts."""


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str
    faithful: bool
    correct: bool


def numbered(chunks):
    return "\n\n".join(f"[{i}] (paper {c.paper}, page {c.page}) {c.text}" for i, c in enumerate(chunks, 1))  # same metadata the generator saw


def judge(question, hits, ans, gold_chunks):
    chunks = [c for _, c in hits]
    if gold_chunks:
        reference = "Reference passages that contain the true answer:\n" + "\n\n".join(c.text for c in gold_chunks)
        rule = "true if the answer is consistent with the reference passages and addresses the question; a refusal is NOT correct here."
    else:
        reference = "This question CANNOT be answered from the papers."
        rule = "true only if the system refused / said the information is missing."
    msg = client.messages.create(
        model=JUDGE_MODEL, max_tokens=500, thinking={"type": "between_tools"},  # this model rejects "disabled"; the schema has its own reasoning field
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, retrieved=numbered(chunks), reference=reference,
            flag=ans.answerable, ans=ans.answer, correct_rule=rule)}],
        output_config={"format": {"type": "json_schema", "schema": Verdict.model_json_schema()}},
    )
    return Verdict.model_validate_json("".join(b.text for b in msg.content if b.type == "text"))


if __name__ == "__main__":
    data = json.loads(Path("eval/retrieval_set_v2.json").read_text())
    r = Retriever()
    items = [(x["q"], x["gold"]) for x in data["answerable"]] + [(q, []) for q in data["unanswerable"]]
    results = []
    for q, gold in items:
        hits = r.retrieve(q, k=5)
        ans, _, _ = answer(q, hits)
        v = judge(q, hits, ans, [r.idx.chunks[g] for g in gold])
        hit = bool(gold) and any(id(c) in {id(r.idx.chunks[g]) for g in gold} for _, c in hits)
        results.append({"q": q, "answerable_q": bool(gold), "retrieved_gold": hit, "model_answerable": ans.answerable,
                        "faithful": v.faithful, "correct": v.correct, "answer": ans.answer, "judge_reasoning": v.reasoning})
    Path("eval/generation_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))

    A = [x for x in results if x["answerable_q"]]
    U = [x for x in results if not x["answerable_q"]]
    print(f"answerable questions (n={len(A)}):")
    print(f"  gold chunk in top-5 : {sum(x['retrieved_gold'] for x in A)}/{len(A)}")
    print(f"  correct             : {sum(x['correct'] for x in A)}/{len(A)}")
    print(f"  faithful            : {sum(x['faithful'] for x in A)}/{len(A)}")
    print(f"  wrongly refused     : {sum(not x['model_answerable'] for x in A)}/{len(A)}")
    print(f"unanswerable questions (n={len(U)}):")
    print(f"  correctly refused   : {sum(x['correct'] for x in U)}/{len(U)}")
    print(f"  faithful            : {sum(x['faithful'] for x in U)}/{len(U)}")
