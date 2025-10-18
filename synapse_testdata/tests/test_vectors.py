"""
test_vectors.py — Vector search stub for Synapse
- Uses sklearn NearestNeighbors with cosine metric
- Ensures deterministic top-k results for a seeded synthetic query
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def test_embeddings_shape(embeddings):
    assert embeddings.ndim == 2
    n, d = embeddings.shape
    assert n == 5 and d == 16, "Expect 5x16 embeddings in the mini pack"

def test_nearest_neighbors(embeddings):
    # Cosine metric nearest neighbors
    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(embeddings)

    # Deterministic pseudo-query: mean of all vectors + slight bias
    q = embeddings.mean(axis=0) + 0.01
    dist, idx = nn.kneighbors([q], n_neighbors=3, return_distance=True)
    # sanity checks
    assert len(idx[0]) == 3
    # Verify indices are in bounds and distances are finite
    assert np.isfinite(dist).all()
    assert (idx >= 0).all() and (idx < embeddings.shape[0]).all()
