"""Contract tests for the SageMaker encoder service and its client. No AWS, no model downloads: the models are replaced by fakes."""
import importlib.util
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.copilot.remote import RemoteModels

os.environ["SKIP_MODEL_LOAD"] = "1"
spec = importlib.util.spec_from_file_location("encoder_app", Path(__file__).resolve().parent.parent / "sagemaker" / "app.py")
encoder_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoder_app)


class FakeEmbed:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[len(t), 1.0, 0.0] for t in texts], dtype=np.float32)


class FakeRerank:
    def predict(self, pairs):
        return np.array([float(len(set(q.split()) & set(t.split()))) for q, t in pairs])


@pytest.fixture
def client():
    encoder_app.models.clear()
    return TestClient(encoder_app.app)


@pytest.fixture
def loaded(client):
    encoder_app.models.update(embed=FakeEmbed(), rerank=FakeRerank())
    return client


def test_ping_is_503_until_models_are_loaded(client):
    assert client.get("/ping").status_code == 503
    assert client.post("/invocations", json={"op": "embed", "texts": ["x"]}).status_code == 503
    encoder_app.models.update(embed=FakeEmbed(), rerank=FakeRerank())
    assert client.get("/ping").status_code == 200


def test_embed_and_rerank_contract(loaded):
    emb = loaded.post("/invocations", json={"op": "embed", "texts": ["ab", "abcd"]}).json()["embeddings"]
    assert emb == [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]]
    scores = loaded.post("/invocations", json={"op": "rerank", "query": "low rank", "texts": ["low rank adaptation", "attention"]}).json()["scores"]
    assert scores == [2.0, 0.0]


@pytest.mark.parametrize("body", [
    {"op": "rerank", "texts": ["a"]},                   # rerank without a query
    {"op": "embed", "texts": []},                       # empty batch
    {"op": "embed", "texts": ["a"] * 257},              # over the batch limit
    {"op": "nope", "texts": ["a"]},                     # unknown operation
])
def test_bad_requests_are_rejected(loaded, body):
    assert loaded.post("/invocations", json=body).status_code == 422


class FakeRuntime:
    """Stands in for boto3's sagemaker-runtime client by calling the app directly."""

    def __init__(self, http):
        self.http, self.calls = http, 0

    def invoke_endpoint(self, EndpointName, ContentType, Body):
        self.calls += 1
        r = self.http.post("/invocations", content=Body, headers={"content-type": ContentType})
        assert r.status_code == 200, r.text
        return {"Body": io.BytesIO(r.content)}


def test_remote_client_round_trip_and_batching(loaded):
    rt = FakeRuntime(loaded)
    remote = RemoteModels("any", client=rt, batch_size=2)
    one = remote.encode("hello")
    assert one.shape == (3,) and one[0] == 5
    many = remote.encode(["a", "bb", "ccc", "dddd", "eeeee"])
    assert many.shape == (5, 3) and many[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert rt.calls == 1 + 3  # 5 texts at batch size 2 -> 3 requests
    assert remote.rerank_scores("low rank", ["low rank adaptation", "attention"]) == [2.0, 0.0]
