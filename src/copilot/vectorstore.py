"""Dense search through a vector database (Chroma, HNSW index) instead of numpy brute force."""
import chromadb
import numpy as np

PATH = "chroma_db"  # gitignored
NAME = "chunks"


class ChromaDense:
    def __init__(self, path: str = PATH):
        self.client = chromadb.PersistentClient(path=path)

    def build(self, matrix: np.ndarray, chunks, batch: int = 1000):
        """Load the precomputed embeddings (no re-encoding). Document ids are the chunk indices."""
        try:
            self.client.delete_collection(NAME)
        except Exception:
            pass
        col = self.client.create_collection(NAME, metadata={"hnsw:space": "cosine"})
        for s in range(0, len(chunks), batch):
            e = min(s + batch, len(chunks))
            col.add(ids=[str(i) for i in range(s, e)], embeddings=matrix[s:e].tolist(),
                    metadatas=[{"paper": c.paper, "page": c.page} for c in chunks[s:e]])

    def search_ids(self, qvec: np.ndarray, n: int) -> list[int]:
        col = self.client.get_collection(NAME)
        res = col.query(query_embeddings=[qvec.tolist()], n_results=n)
        return [int(i) for i in res["ids"][0]]
