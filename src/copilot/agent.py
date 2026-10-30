"""Agentic RAG: the model decides what to search, how often, and when to stop."""
from dataclasses import dataclass, field

from dotenv import load_dotenv
import anthropic

from .pipeline import Retriever

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"
MAX_STEPS = 6

PAPERS = {
    "1706.03762": "Attention Is All You Need (Transformer)",
    "2005.11401": "Retrieval-Augmented Generation (RAG)",
    "2106.09685": "LoRA",
    "2305.14314": "QLoRA",
    "2210.03629": "ReAct",
}

SYSTEM = f"""You are a research assistant answering questions using a small library of ML papers:
{chr(10).join(f'- {k}: {v}' for k, v in PAPERS.items())}

Use search_papers to find evidence. You may search several times: rephrase the query, use the technical terms
a paper would use, restrict to one paper, or split a comparison question into one search per paper.
Rules:
- Answer ONLY from passages you retrieved. Do not use outside knowledge.
- Cite the passage ids you used, like [c12].
- If after searching the evidence is still insufficient, call final_answer with answerable=false and say what is missing.
- Be efficient: stop searching once you have enough evidence."""

TOOLS = [
    {
        "name": "search_papers",
        "description": "Search the paper library. Returns the 5 most relevant passages with ids like c12. Optionally restrict to one paper id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "paper": {"type": "string", "enum": list(PAPERS), "description": "Optional paper id to restrict the search"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "final_answer",
        "description": "Submit the final answer. Call exactly once when done.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answerable": {"type": "boolean"},
                "answer": {"type": "string", "description": "Answer with inline citations like [c12]"},
                "citations": {"type": "array", "items": {"type": "integer"}, "description": "Chunk numbers used (12 for c12)"},
            },
            "required": ["answerable", "answer", "citations"],
        },
    },
]


@dataclass
class AgentResult:
    answerable: bool
    answer: str
    citations: list[int]
    searches: list[dict] = field(default_factory=list)  # the trajectory: every tool call the model made
    retrieved_ids: list[int] = field(default_factory=list)
    steps: int = 0
    input_tokens: int = 0
    hit_step_limit: bool = False


def run_agent(retriever: Retriever, question: str, max_steps: int = MAX_STEPS) -> AgentResult:
    messages = [{"role": "user", "content": question}]
    res = AgentResult(False, "", [])
    for step in range(1, max_steps + 1):
        resp = client.messages.create(
            model=MODEL, max_tokens=1200, system=SYSTEM, tools=TOOLS, messages=messages,
            thinking={"type": "disabled"},
        )
        res.steps, res.input_tokens = step, res.input_tokens + resp.usage.input_tokens
        messages.append({"role": "assistant", "content": resp.content})
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:  # model answered in plain text without the tool: treat as a final answer
            res.answerable, res.answer = True, "".join(b.text for b in resp.content if b.type == "text")
            return res
        results = []
        for call in calls:
            if call.name == "final_answer":
                res.answerable, res.answer, res.citations = call.input["answerable"], call.input["answer"], call.input.get("citations", [])
                return res
            ids = retriever.retrieve_ids(call.input["query"], k=5, paper=call.input.get("paper"))
            res.searches.append(call.input)
            res.retrieved_ids += [i for i in ids if i not in res.retrieved_ids]
            text = "\n\n".join(
                f"[c{i}] (paper {retriever.idx.chunks[i].paper}, page {retriever.idx.chunks[i].page}) {retriever.idx.chunks[i].text}"
                for i in ids
            )
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": text})
        messages.append({"role": "user", "content": results})
    # Step limit reached: force a final answer from what we have.
    res.hit_step_limit = True
    messages.append({"role": "user", "content": "Step limit reached. Call final_answer now with what you have."})
    resp = client.messages.create(model=MODEL, max_tokens=1200, system=SYSTEM, tools=TOOLS, messages=messages,
                                  tool_choice={"type": "tool", "name": "final_answer"}, thinking={"type": "disabled"})
    for b in resp.content:
        if b.type == "tool_use":
            res.answerable, res.answer, res.citations = b.input["answerable"], b.input["answer"], b.input.get("citations", [])
    return res


if __name__ == "__main__":
    r = Retriever()
    for q in ["What problems of pure step-by-step reasoning does acting in an environment fix?",
              "What number format is used for the frozen weights, and why does it suit the weight distribution?"]:
        out = run_agent(r, q)
        print(f"\nQ: {q}\nsteps={out.steps} tokens={out.input_tokens} answerable={out.answerable}")
        for s in out.searches:
            print("  search:", s)
        print("A:", out.answer[:500])
