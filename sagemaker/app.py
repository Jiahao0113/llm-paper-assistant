"""Embedding + reranking service that follows the SageMaker container contract:
GET /ping (ready?) and POST /invocations on port 8080. Two operations:
  {"op": "embed",  "texts": [...]}                 -> {"embeddings": [[...], ...]}   (L2-normalised MiniLM vectors)
  {"op": "rerank", "query": "...", "texts": [...]} -> {"scores": [...]}              (cross-encoder, higher = more relevant)"""
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, model_validator

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

models: dict = {}
lock = threading.Lock()  # concurrent torch inference hung in testing on the main project: one request at a time per container


def load_models() -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    models["embed"] = SentenceTransformer(EMBED_MODEL)
    models["rerank"] = CrossEncoder(RERANK_MODEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load in the background so the port opens immediately; /ping answers 503 until the models are ready.
    if not os.environ.get("SKIP_MODEL_LOAD"):
        threading.Thread(target=load_models, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


class Invocation(BaseModel):
    op: Literal["embed", "rerank"]
    texts: list[str] = Field(min_length=1, max_length=256)
    query: str | None = None

    @model_validator(mode="after")
    def rerank_needs_query(self):
        if self.op == "rerank" and not self.query:
            raise ValueError("rerank needs a query")
        return self


@app.get("/ping")
def ping():
    # SageMaker routes traffic here only after this returns 200, and replaces the instance if it keeps failing.
    return Response(status_code=200 if {"embed", "rerank"} <= models.keys() else 503)


@app.post("/invocations")
def invocations(req: Invocation):
    if not {"embed", "rerank"} <= models.keys():
        raise HTTPException(status_code=503, detail="models still loading")
    with lock:
        if req.op == "embed":
            return {"embeddings": models["embed"].encode(req.texts, normalize_embeddings=True).tolist()}
        scores = models["rerank"].predict([(req.query, t) for t in req.texts])
    return {"scores": [float(s) for s in scores]}
