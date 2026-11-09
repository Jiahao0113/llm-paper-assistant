"""The staged research pipeline as a LangGraph StateGraph.
Same step functions as research.staged(); what the framework changes is the wiring:
  - research runs once per sub-question IN PARALLEL (Send fan-out) instead of in a Python loop
  - the verify -> repair loop is a conditional edge
  - state is checkpointed, so the run can pause at the plan (human approval) and resume
"""
import time
import uuid
from typing import Annotated, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from .pipeline import Retriever
from .research import Report, step_plan, step_repair, step_research, step_verify, step_write


def merge_pool(a: list[int], b: list[int]) -> list[int]:
    return a + [x for x in b if x not in a]


class State(TypedDict):
    task: str
    sub_questions: list[str]
    pool: Annotated[list[int], merge_pool]  # parallel research branches each add their evidence
    text: str
    sents: list[str]
    bad: list[int]
    attempt: int


def build_graph(r: Retriever, rep: Report, max_repairs: int = 2, approve_plan: bool = False):
    def plan(state: State):
        return {"sub_questions": step_plan(rep, state["task"])}

    def approve(state: State):
        # Pauses the run. The caller resumes with the (possibly edited) list of sub-questions.
        return {"sub_questions": interrupt({"sub_questions": state["sub_questions"]})}

    def fan_out(state: State):
        return [Send("research", {"sub_question": sq}) for sq in state["sub_questions"]]

    def research(branch: dict):
        return {"pool": step_research(r, rep, branch["sub_question"])}

    def write(state: State):
        return {"text": step_write(r, rep, state["task"], state["pool"]), "attempt": 0}

    def verify(state: State):
        sents, bad = step_verify(r, rep, state["text"], state["attempt"])
        return {"sents": sents, "bad": bad}

    def repair(state: State):
        text, pool = step_repair(r, rep, state["task"], state["pool"], state["text"], state["sents"], state["bad"], state["attempt"])
        return {"text": text, "pool": pool, "attempt": state["attempt"] + 1}

    def route(state: State):
        return "repair" if state["bad"] and state["attempt"] < max_repairs else END

    g = StateGraph(State)
    for name, fn in [("plan", plan), ("research", research), ("write", write), ("verify", verify), ("repair", repair)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    if approve_plan:
        g.add_node("approve", approve)
        g.add_edge("plan", "approve")
        g.add_conditional_edges("approve", fan_out, ["research"])
    else:
        g.add_conditional_edges("plan", fan_out, ["research"])
    g.add_edge("research", "write")  # waits for all parallel branches
    g.add_edge("write", "verify")
    g.add_conditional_edges("verify", route, ["repair", END])
    g.add_edge("repair", "verify")
    return g.compile(checkpointer=MemorySaver())


def staged_graph(r: Retriever, task: str, max_repairs: int = 2, approve: Callable[[list[str]], list[str]] | None = None) -> Report:
    """Run the graph. If `approve` is given, the run pauses after planning, calls approve(sub_questions) and resumes with its result."""
    t0, rep = time.time(), Report("", "staged_graph")
    graph = build_graph(r, rep, max_repairs, approve_plan=approve is not None)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    out = graph.invoke({"task": task, "pool": [], "attempt": 0}, config)
    while out.get("__interrupt__"):
        proposed = out["__interrupt__"][0].value["sub_questions"]
        rep.trace.append({"stage": "human_approval", "proposed": proposed})
        out = graph.invoke(Command(resume=approve(proposed)), config)
    rep.text, rep.retrieved_ids = out["text"], out["pool"]
    rep.seconds = time.time() - t0
    return rep
