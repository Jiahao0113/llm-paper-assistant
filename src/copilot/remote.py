"""Client for the SageMaker encoder endpoint (sagemaker/app.py). Set SAGEMAKER_ENDPOINT to use it instead of local models."""
import json
import os

import numpy as np


class RemoteModels:
    """Same `encode` call as SentenceTransformer, plus `rerank_scores`, backed by a SageMaker endpoint."""

    def __init__(self, endpoint: str | None = None, client=None, batch_size: int = 64):
        self.endpoint = endpoint or os.environ["SAGEMAKER_ENDPOINT"]
        if client is None:
            import boto3  # only needed when the endpoint is used

            client = boto3.client("sagemaker-runtime")
        self.client, self.batch_size = client, batch_size

    def _call(self, payload: dict) -> dict:
        resp = self.client.invoke_endpoint(EndpointName=self.endpoint, ContentType="application/json", Body=json.dumps(payload))
        return json.loads(resp["Body"].read())

    def encode(self, texts, normalize_embeddings: bool = True, batch_size: int | None = None) -> np.ndarray:
        assert normalize_embeddings, "the endpoint always returns normalised vectors"
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        step = batch_size or self.batch_size
        out = [v for i in range(0, len(texts), step) for v in self._call({"op": "embed", "texts": texts[i:i + step]})["embeddings"]]
        arr = np.asarray(out, dtype=np.float32)
        return arr[0] if single else arr

    def rerank_scores(self, query: str, texts: list[str]) -> list[float]:
        return self._call({"op": "rerank", "query": query, "texts": texts})["scores"]


def endpoint_name() -> str | None:
    return os.environ.get("SAGEMAKER_ENDPOINT") or None
