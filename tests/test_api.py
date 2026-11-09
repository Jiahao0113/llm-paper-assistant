from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.copilot import api
from src.copilot.research import Report


class FakeIdx:
    chunks = [SimpleNamespace(paper="p", title="t", section="s", page=1, text=f"text {i}") for i in range(5)]


@pytest.fixture
def client(monkeypatch):
    api.state["retriever"] = SimpleNamespace(idx=FakeIdx())
    monkeypatch.setenv("APP_API_KEY", "secret")

    def fake_system(r, task):
        # cites c1 (retrieved), c3 (never retrieved: must not come back as a source) and c99 (does not exist)
        return Report(text="A claim [c1]. Another claim [c3][c99].", system="fake", retrieved_ids=[0, 1, 2], input_tokens=1000, output_tokens=100, llm_calls=2)

    monkeypatch.setitem(api.SYSTEMS, "staged_r1", fake_system)  # the default system
    return TestClient(api.app)  # no `with`: the lifespan (which loads the real models) is skipped


TASK = {"task": "Compare LoRA and QLoRA in detail please."}


def test_research_requires_the_api_key(client):
    assert client.post("/research", json=TASK).status_code == 401
    assert client.post("/research", json=TASK, headers={"x-api-key": "wrong"}).status_code == 401


def test_research_returns_only_cited_and_retrieved_sources(client):
    resp = client.post("/research", json=TASK, headers={"x-api-key": "secret"})
    assert resp.status_code == 200
    assert [s["id"] for s in resp.json()["sources"]] == [1]
    assert resp.json()["cost_usd"] > 0


def test_research_validates_input(client):
    assert client.post("/research", json={"task": "short"}, headers={"x-api-key": "secret"}).status_code == 422
    assert client.post("/research", json={**TASK, "system": "nope"}, headers={"x-api-key": "secret"}).status_code == 422


def test_upstream_failure_becomes_502_not_a_stack_trace(client, monkeypatch):
    def boom(r, task):
        raise RuntimeError("secret internals")

    monkeypatch.setitem(api.SYSTEMS, "staged_r1", boom)
    resp = client.post("/research", json=TASK, headers={"x-api-key": "secret"})
    assert resp.status_code == 502 and "secret internals" not in resp.text
