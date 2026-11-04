"""Exact brute force vs HNSW as the number of vectors grows.
Vectors = real MiniLM chunk embeddings plus noise, so they keep realistic cluster structure. (Pure random vectors in 384-D are
nearly equidistant and make ANN look terrible: HNSW recall@10 was 0.02 at 500k there.)"""
import time

import hnswlib
import numpy as np

DIM, K, NQ, NOISE = 384, 10, 200, 0.7
base = np.load("data/index/matrix.npy")
rng = np.random.default_rng(0)


def noisy(n):
    x = base[rng.integers(0, len(base), n)] + NOISE * rng.standard_normal((n, DIM), dtype=np.float32) / np.sqrt(DIM)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


print(f"{'N':>9s} {'exact ms':>9s} {'ef':>4s} {'hnsw ms':>8s} {'recall@10':>10s} {'build s':>8s}")
for n in (5_000, 50_000, 500_000):
    X, Q = noisy(n), noisy(NQ)
    t = time.perf_counter()
    truth = [set(np.argpartition(-(X @ q), K)[:K]) for q in Q]
    t_exact = (time.perf_counter() - t) / NQ * 1000
    idx = hnswlib.Index(space="cosine", dim=DIM)
    t = time.perf_counter()
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    idx.add_items(X)
    t_build = time.perf_counter() - t
    for ef in (20, 50, 200):
        idx.set_ef(ef)
        t = time.perf_counter()
        got = [set(idx.knn_query(q, k=K)[0][0]) for q in Q]
        t_hnsw = (time.perf_counter() - t) / NQ * 1000
        rec = np.mean([len(a & b) / K for a, b in zip(truth, got)])
        print(f"{n:9d} {t_exact:9.2f} {ef:4d} {t_hnsw:8.2f} {rec:10.3f} {t_build:8.1f}", flush=True)
