"""RAG v1: retrieve top-k chunks, ask the LLM to answer ONLY from them, with citations."""
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, ConfigDict

from .retrieval import DenseIndex

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"

SYSTEM = """You are a research assistant answering questions about machine learning papers.
Rules:
- Use ONLY the numbered context passages provided. Do not use outside knowledge, even if you know the answer.
- After each claim, cite the passage numbers that support it, like [1] or [2][3].
- If the passages do not contain enough information to answer, set answerable=false and say what is missing. Do not guess.
- Be concise."""


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answerable: bool
    answer: str
    citations: list[int]  # passage numbers actually used


def build_prompt(question: str, hits) -> str:
    passages = "\n\n".join(
        f"[{i}] (paper {c.paper}, page {c.page}, section: {c.section})\n{c.text}"
        for i, (_, c) in enumerate(hits, start=1)
    )
    return f"Context passages:\n\n{passages}\n\nQuestion: {question}"


def answer(question: str, hits):
    msg = client.messages.create(
        model=MODEL, max_tokens=600, thinking={"type": "disabled"}, system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
        output_config={"format": {"type": "json_schema", "schema": Answer.model_json_schema()}},
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    ans = Answer.model_validate_json(text)
    sources = [(n, hits[n - 1][1]) for n in ans.citations if 1 <= n <= len(hits)]
    return ans, sources, msg.usage


def ask(index: DenseIndex, question: str, k: int = 5):
    return answer(question, index.search(question, k))


if __name__ == "__main__":
    idx = DenseIndex()
    idx.load()
    for q in [
        "How does LoRA reduce the number of trainable parameters?",
        "What is double quantization in QLoRA?",
        "How many parameters does GPT-4 have?",  # not in our papers: should refuse
    ]:
        ans, sources, usage = ask(idx, q)
        print(f"\nQ: {q}\nanswerable={ans.answerable}  (input tokens: {usage.input_tokens})")
        print("A:", ans.answer)
        for n, c in sources:
            print(f"   [{n}] {c.paper} p{c.page} {c.section[:40]!r}")
