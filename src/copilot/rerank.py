"""Cross-encoder reranking: score (query, chunk) pairs jointly, then sort."""
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def rerank(query: str, chunk_ids: list[int], texts: list[str]) -> list[int]:
    """Return chunk_ids sorted best-first by cross-encoder score."""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    scores = _model.predict([(query, t) for t in texts])  # one forward pass per pair
    return [cid for _, cid in sorted(zip(scores, chunk_ids), key=lambda x: -x[0])]
