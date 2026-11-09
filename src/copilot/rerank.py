"""Cross-encoder reranking: score (query, chunk) pairs jointly, then sort."""
from sentence_transformers import CrossEncoder

from .remote import RemoteModels, endpoint_name

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def rerank(query: str, chunk_ids: list[int], texts: list[str]) -> list[int]:
    """Return chunk_ids sorted best-first by cross-encoder score."""
    global _model
    if _model is None:
        endpoint = endpoint_name()  # SAGEMAKER_ENDPOINT set -> score through the SageMaker endpoint
        _model = RemoteModels(endpoint) if endpoint else CrossEncoder(RERANK_MODEL)
    if isinstance(_model, RemoteModels):
        scores = _model.rerank_scores(query, texts)
    else:
        scores = _model.predict([(query, t) for t in texts])  # one forward pass per pair
    return [cid for _, cid in sorted(zip(scores, chunk_ids), key=lambda x: -x[0])]
