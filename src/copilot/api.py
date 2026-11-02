"""HTTP API: POST /ask runs the agent and returns the answer with the cited passages."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .agent import run_agent
from .pipeline import Retriever

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever()  # load models and index once, not per request
    yield


app = FastAPI(title="LLM Paper Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


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


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].idx.chunks)}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    r = state["retriever"]
    try:
        out = await run_in_threadpool(run_agent, r, req.question)  # blocking API calls off the event loop
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}")
    # Only return citations that were really retrieved; the model can cite an id it never saw.
    cited = [i for i in dict.fromkeys(out.citations) if i in out.retrieved_ids]
    return AskResponse(
        answerable=out.answerable, answer=out.answer,
        sources=[Source(id=i, **{k: getattr(r.idx.chunks[i], k) for k in ("paper", "title", "section", "page", "text")}) for i in cited],
        searches=out.searches, steps=out.steps, input_tokens=out.input_tokens, hit_step_limit=out.hit_step_limit,
    )
