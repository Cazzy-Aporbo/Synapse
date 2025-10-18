"""
test_papers_rag.py — Tiny RAG pipeline smoke test
- Rank papers by cosine similarity to a query vector
- Build a short context and verify output schema
"""

import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def build_context(papers_df, indices):
    rows = papers_df.iloc[indices][["id","title","abstract","url"]]
    ctx = "\n\n".join(f"[{r.id}] {r.title}\n{r.abstract}\n{r.url}" for r in rows.itertuples())
    return ctx

def test_rag_pipeline(embeddings, papers_df, queries):
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(embeddings)

    # Use first query as a stub; in real code, you would embed the query
    # Here we emulate an embedding by a deterministic projection of token length
    q_len = len(queries[0]["query"])
    rng = np.random.default_rng(q_len)
    q = rng.normal(0, 1, size=embeddings.shape[1])

    dist, idx = nn.kneighbors([q], n_neighbors=3, return_distance=True)
    ctx = build_context(papers_df, idx[0])

    # "Mock LLM" reply (just a stitched string); verify schema
    answer = {
        "answer": "This is a stubbed synthesis based on the top-3 retrieved abstracts.",
        "citations": papers_df.iloc[idx[0]]["id"].tolist(),
        "context_preview": ctx[:240] + "...",
    }
    assert isinstance(answer["answer"], str)
    assert len(answer["citations"]) == 3
    assert all(isinstance(cid, str) for cid in answer["citations"])
