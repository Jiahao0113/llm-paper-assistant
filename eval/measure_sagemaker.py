"""After deploying the endpoint: do SageMaker encoders give the same retrieval results as the local models, and how fast are they?
Usage: SAGEMAKER_ENDPOINT=paper-assistant-encoders python -m eval.measure_sagemaker   (needs AWS credentials in the shell)"""
import json
import os
import statistics as st
import time

endpoint = os.environ.pop("SAGEMAKER_ENDPOINT")  # build the local retriever first, then switch the models to the endpoint

from src.copilot import rerank as rr  # noqa: E402
from src.copilot.pipeline import Retriever  # noqa: E402
from src.copilot.remote import RemoteModels  # noqa: E402

queries = [x["question"] for x in json.load(open("eval/retrieval_set.json"))][:20]
r = Retriever()
local = [r.retrieve_ids(q, k=5) for q in queries]

remote = RemoteModels(endpoint)
r.idx.model, rr._model = remote, remote
times, got = [], []
for q in queries:
    t = time.perf_counter()
    got.append(r.retrieve_ids(q, k=5))
    times.append(time.perf_counter() - t)
print(f"endpoint: {endpoint}")
print(f"identical top-5 as local models: {sum(a == b for a, b in zip(local, got))}/{len(queries)} queries")
print(f"first query (may include a cold start): {times[0]:.2f}s")
warm = sorted(times[1:])
print(f"remaining {len(warm)} queries: median {st.median(warm) * 1000:.0f} ms, max {warm[-1] * 1000:.0f} ms per retrieval (embed + rerank = 2 endpoint calls)")
