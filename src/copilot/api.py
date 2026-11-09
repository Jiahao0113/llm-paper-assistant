"""HTTP API. POST /ask answers a question with the agent; POST /research writes a cited brief with one of the research systems."""
import hmac
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from .config import load_secrets

load_secrets()  # must run before the modules below: they create the Anthropic client at import time

from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .agent import run_agent  # noqa: E402
from .pipeline import Retriever  # noqa: E402
from .research import cited_ids  # noqa: E402
from .systems import SYSTEMS  # noqa: E402

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever()  # load models and index once, not per request
    yield


app = FastAPI(title="LLM Paper Assistant", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)):
    """If APP_API_KEY is set (always, on AWS), requests must send it. A public URL that spends my model credits needs a gate."""
    expected = os.environ.get("APP_API_KEY")
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid x-api-key")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class ResearchRequest(BaseModel):
    task: str = Field(min_length=10, max_length=1500)
    # staged_r1 is the default: one repair round matched two in the evaluation at about 20% lower cost
    system: Literal["single_shot", "react", "staged_r1", "staged", "graph"] = "staged_r1"


class Source(BaseModel):
    id: int
    paper: str
    title: str
    section: str
    page: int
    text: str


class AskResponse(BaseModel):
    answerable: bool
    answer: str
    sources: list[Source]
    searches: list[dict]
    steps: int
    input_tokens: int
    hit_step_limit: bool


class ResearchResponse(BaseModel):
    system: str
    text: str
    sources: list[Source]
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    seconds: float
    trace: list[dict]


def sources_for(r: Retriever, ids: list[int]) -> list[Source]:
    return [Source(id=i, **{k: getattr(r.idx.chunks[i], k) for k in ("paper", "title", "section", "page", "text")}) for i in ids]


def emit_metrics(system: str, cost: float, seconds: float, calls: int, tokens_in: int, tokens_out: int) -> None:
    """One CloudWatch Embedded Metric Format line: CloudWatch turns it into metrics, no agent or SDK needed."""
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": "PaperAssistant", "Dimensions": [["System"]],
            "Metrics": [{"Name": "CostUSD", "Unit": "None"}, {"Name": "LatencySeconds", "Unit": "Seconds"}, {"Name": "LLMCalls", "Unit": "Count"},
                        {"Name": "InputTokens", "Unit": "Count"}, {"Name": "OutputTokens", "Unit": "Count"}]}]},
        "System": system, "CostUSD": cost, "LatencySeconds": seconds, "LLMCalls": calls, "InputTokens": tokens_in, "OutputTokens": tokens_out}), flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].idx.chunks)}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
async def ask(req: AskRequest):
    r = state["retriever"]
    try:
        out = await run_in_threadpool(run_agent, r, req.question)  # blocking API calls off the event loop
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}")
    # Only return citations that were really retrieved; the model can cite an id it never saw.
    cited = [i for i in dict.fromkeys(out.citations) if i in out.retrieved_ids]
    return AskResponse(answerable=out.answerable, answer=out.answer, sources=sources_for(r, cited), searches=out.searches,
                       steps=out.steps, input_tokens=out.input_tokens, hit_step_limit=out.hit_step_limit)


@app.post("/research", response_model=ResearchResponse, dependencies=[Depends(require_key)])
async def research(req: ResearchRequest):
    r = state["retriever"]
    try:
        rep = await run_in_threadpool(SYSTEMS[req.system], r, req.task)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}")
    ids = [i for i in cited_ids(rep.text) if i < len(r.idx.chunks) and i in rep.retrieved_ids]  # cited AND actually retrieved
    emit_metrics(req.system, rep.cost, rep.seconds, rep.llm_calls, rep.input_tokens, rep.output_tokens)
    return ResearchResponse(system=req.system, text=rep.text, sources=sources_for(r, ids), llm_calls=rep.llm_calls, input_tokens=rep.input_tokens,
                            output_tokens=rep.output_tokens, cost_usd=round(rep.cost, 5), seconds=round(rep.seconds, 1), trace=rep.trace)
