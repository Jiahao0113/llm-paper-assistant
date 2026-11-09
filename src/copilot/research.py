"""Research-brief systems over the paper library. Three ways to answer the same task, to be compared:
  single_shot  - one retrieval (top 10) with the task text, one generation
  react_report - the tool-use agent loop (agent.py) with a report-writing prompt
  staged       - plan -> research each sub-question (agent loop) -> write -> verify citations -> repair
All return a Report with the text, what was cited, token usage and a trace of the steps taken."""
import json
import re
import threading
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from .agent import PAPERS, client, run_agent
from .pipeline import Retriever

MODEL = "claude-haiku-5-5"
REPORT_MAX_TOKENS = 4000  # 1500 truncated 3 of 17 staged briefs on tasks with 5-6 items
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6  # USD per token, Haiku 5.5 (prompts up to 100K tokens)

REPORT_RULES = """Write a short research brief that answers the task.
- Organise it by the items or questions in the task (a short section or bullet per item), then a brief comparison or conclusion.
- Use ONLY the evidence passages. Every factual sentence must end with citations to the passage ids that support it, like [c12] or [c12][c40].
- Do not state anything you cannot cite. If the library does not cover part of the task (a paper or topic is missing), say so explicitly instead of filling the gap from memory.
- Be concise: about 250-450 words."""

PAPER_LIST = "\n".join(f"- {k}: {v}" for k, v in PAPERS.items())


@dataclass
class Report:
    text: str
    system: str
    retrieved_ids: list[int] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    seconds: float = 0.0
    trace: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # parallel graph branches update the counters

    @property
    def cost(self) -> float:
        return self.input_tokens * PRICE_IN + self.output_tokens * PRICE_OUT

    def add_counts(self, input_tokens: int, output_tokens: int, calls: int):
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.llm_calls += calls

    def add(self, usage):
        self.add_counts(usage.input_tokens, usage.output_tokens, 1)


def cited_ids(text: str) -> list[int]:
    ids = []
    for group in re.findall(r"\[([^\]]*c\d+[^\]]*)\]", text):
        ids += [int(i) for i in re.findall(r"c(\d+)", group)]
    return list(dict.fromkeys(ids))


def evidence_block(r: Retriever, ids: list[int]) -> str:
    return "\n\n".join(f"[c{i}] (paper {r.idx.chunks[i].paper}, p{r.idx.chunks[i].page}) {r.idx.chunks[i].text}" for i in ids)


def text_of(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


def llm(rep: Report, system: str, user: str, max_tokens: int = REPORT_MAX_TOKENS, schema: type[BaseModel] | None = None):
    kw = {"output_config": {"format": {"type": "json_schema", "schema": schema.model_json_schema()}}} if schema else {}
    resp = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system, thinking={"type": "disabled"},
                                  messages=[{"role": "user", "content": user}], **kw)
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("output truncated")
    rep.add(resp.usage)
    return schema.model_validate_json(text_of(resp)) if schema else text_of(resp)


# --- system 1: one retrieval, one generation -------------------------------------------------
def single_shot(r: Retriever, task: str, k: int = 10) -> Report:
    t0, rep = time.time(), Report("", "single_shot")
    ids = r.retrieve_ids(task, k=k)
    rep.retrieved_ids = ids
    rep.text = llm(rep, REPORT_RULES + f"\n\nThe library contains these papers:\n{PAPER_LIST}",
                   f"Evidence passages:\n\n{evidence_block(r, ids)}\n\nTask: {task}")
    rep.trace.append({"stage": "retrieve+write", "ids": ids})
    rep.seconds = time.time() - t0
    return rep


# --- system 2: the tool-use agent loop, asked for a report -------------------------------------
REACT_SYSTEM = f"""You are a research assistant working with a library of ML papers:
{PAPER_LIST}

Use search_papers to gather evidence (several searches, rephrased queries, one search per paper for comparisons), then call final_answer.
The final_answer 'answer' field must be the finished brief.
{REPORT_RULES}
If the task cannot be answered from the library, call final_answer with answerable=false and explain what is missing, but still report what the library does cover."""


def react_report(r: Retriever, task: str) -> Report:
    t0, rep = time.time(), Report("", "react")
    out = run_agent(r, task, max_steps=10, system=REACT_SYSTEM, max_tokens=6000)  # a report is long; 2500 truncated one in testing
    rep.text, rep.retrieved_ids = out.answer, out.retrieved_ids
    rep.input_tokens, rep.output_tokens, rep.llm_calls = out.input_tokens, out.output_tokens, out.steps
    rep.trace = [{"stage": "search", **s} for s in out.searches]
    rep.seconds = time.time() - t0
    return rep


# --- system 3: staged pipeline with a citation verifier -----------------------------------------
class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sub_questions: list[str]


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unsupported: list[int]  # indexes of the numbered sentences whose claim is NOT supported by the cited passages
    reasons: list[str]


PLAN_SYSTEM = f"""You plan research briefs over this paper library:
{PAPER_LIST}
Split the task into 3-6 focused sub-questions, each answerable from one paper or one topic. Name the paper (by title) in the sub-question when the task names it. Return JSON."""

WRITE_SYSTEM = REPORT_RULES + "\nThe evidence passages come from sub-question research; a sub-question may have found nothing, which means the library may not cover it."

VERIFY_SYSTEM = """You check citations. You get a numbered list of sentences, each with the passages it cites. A sentence is UNSUPPORTED if a factual claim in it is not stated by, or not clearly implied by, the cited passages, or if it cites no passage but makes a factual claim. Sentences that only organise the text, or that say the library does not cover something, are supported. Return the indexes of unsupported sentences with a short reason each."""


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?\]])\s+(?=[A-Z\-*\d#])|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 25 and not p.strip().startswith("#")]


def verify(r: Retriever, rep: Report, text: str):
    sents = split_sentences(text)
    blocks = []
    for n, s in enumerate(sents):
        ids = [i for i in cited_ids(s) if i < len(r.idx.chunks)]
        blocks.append(f"({n}) {s}\n   cited passages:\n" + (evidence_block(r, ids) if ids else "   (none)"))
    chk = llm(rep, VERIFY_SYSTEM, "\n\n".join(blocks), max_tokens=1200, schema=Check)
    return sents, [i for i in chk.unsupported if 0 <= i < len(sents)], chk.reasons


# The steps of the staged pipeline as separate functions: the hand-written orchestration below and the LangGraph version
# (research_graph.py) both call these, so they differ only in how the steps are wired together.
def step_plan(rep: Report, task: str) -> list[str]:
    sub_questions = llm(rep, PLAN_SYSTEM, task, max_tokens=600, schema=Plan).sub_questions
    rep.trace.append({"stage": "plan", "sub_questions": sub_questions})
    return sub_questions


def step_research(r: Retriever, rep: Report, sub_question: str) -> list[int]:
    out = run_agent(r, sub_question, max_steps=4)
    rep.add_counts(out.input_tokens, out.output_tokens, out.steps)
    rep.trace.append({"stage": "research", "sub_question": sub_question, "searches": [s["query"] for s in out.searches], "answerable": out.answerable})
    return out.retrieved_ids


def step_write(r: Retriever, rep: Report, task: str, pool: list[int]) -> str:
    return llm(rep, WRITE_SYSTEM, f"Evidence passages:\n\n{evidence_block(r, pool)}\n\nTask: {task}", max_tokens=REPORT_MAX_TOKENS)


def step_verify(r: Retriever, rep: Report, text: str, attempt: int = 0):
    sents, bad, _ = verify(r, rep, text)
    rep.trace.append({"stage": "verify", "attempt": attempt, "sentences": len(sents), "unsupported": len(bad)})
    return sents, bad


def step_repair(r: Retriever, rep: Report, task: str, pool: list[int], text: str, sents: list[str], bad: list[int], attempt: int = 0):
    """Find evidence for each flagged claim, then revise the brief. Returns (new text, extended pool)."""
    extra = []
    for i in bad:
        for c in r.retrieve_ids(re.sub(r"\[c\d+\]", "", sents[i]), k=3):
            if c not in pool and c not in extra:
                extra.append(c)
    pool = pool + extra
    flagged = "\n".join(f"- {sents[i]}" for i in bad)
    text = llm(rep, WRITE_SYSTEM,
               f"Evidence passages:\n\n{evidence_block(r, pool)}\n\nTask: {task}\n\nDraft brief:\n{text}\n\n"
               f"A checker found these sentences unsupported by their citations:\n{flagged}\n\n"
               "Revise the brief: fix each flagged claim using the evidence, or delete it / say the library does not cover it. Keep supported content. Output the full revised brief.", max_tokens=REPORT_MAX_TOKENS)
    rep.trace.append({"stage": "repair", "attempt": attempt, "extra_evidence": extra})
    return text, pool


def staged(r: Retriever, task: str, max_repairs: int = 2) -> Report:
    t0, rep = time.time(), Report("", "staged")
    pool: list[int] = []
    for sq in step_plan(rep, task):
        pool += [i for i in step_research(r, rep, sq) if i not in pool]
    text = step_write(r, rep, task, pool)
    for attempt in range(max_repairs + 1):
        sents, bad = step_verify(r, rep, text, attempt)
        if not bad or attempt == max_repairs:
            break
        text, pool = step_repair(r, rep, task, pool, text, sents, bad, attempt)
    rep.text, rep.retrieved_ids = text, pool
    rep.seconds = time.time() - t0
    return rep


def staged_noverify(r: Retriever, task: str) -> Report:
    """Ablation: the same plan -> research -> write, with no citation verification or repair."""
    t0, rep = time.time(), Report("", "staged_noverify")
    pool: list[int] = []
    for sq in step_plan(rep, task):
        pool += [i for i in step_research(r, rep, sq) if i not in pool]
    rep.text, rep.retrieved_ids = step_write(r, rep, task, pool), pool
    rep.seconds = time.time() - t0
    return rep


SYSTEMS = {"single_shot": single_shot, "react": react_report, "staged": staged}

if __name__ == "__main__":
    r = Retriever()
    task = json.load(open("eval/research_tasks.json"))["tasks"][12]["q"]
    for name, fn in SYSTEMS.items():
        rep = fn(r, task)
        print(f"\n=== {name}: {rep.llm_calls} calls, {rep.input_tokens} in / {rep.output_tokens} out, ${rep.cost:.4f}, {rep.seconds:.0f}s")
        print(rep.text[:900])
