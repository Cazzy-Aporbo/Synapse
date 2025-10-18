"""
test_full_cycle.py — End-to-end mini integration test
Pipeline:
  query -> embed (deterministic stub) -> nearest neighbors -> build context
  -> mock agent trio -> weighted consensus -> structured final output
"""

import json, yaml, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors

def stub_embed(text: str, dim: int = 16) -> np.ndarray:
    # Deterministic pseudo-embedding based on text length + simple hash
    seed = len(text) + sum(ord(c) for c in text[:10])
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=dim)

def test_full_cycle(data_root, papers_df, embeddings, queries):
    # 1) Vector search
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(embeddings)
    q = stub_embed(queries[0]["query"], dim=embeddings.shape[1])
    dist, idx = nn.kneighbors([q], n_neighbors=3, return_distance=True)

    # 2) Build context
    ctx_rows = papers_df.iloc[idx[0]][["id","title","abstract"]].to_dict(orient="records")
    context = "\n\n".join(f"[{r['id']}] {r['title']}\n{r['abstract']}" for r in ctx_rows)

    # 3) Mock 3 agents
    agents = yaml.safe_load((data_root / "agents.yaml").read_text(encoding="utf-8"))["agents"]
    def mock_agent_response(role, ctx):
        # simple scoring: longer abstracts get slightly higher score; role temp adds jitter
        score = sum(len(r['abstract']) for r in ctx_rows) / 1000.0
        rng = np.random.default_rng(len(role["name"]) + int(role["temperature"]*100))
        score += rng.normal(0, 0.02)
        return {"text": f"{role['name']} summary of evidence...", "score": float(max(0.0, min(1.0, score)))}

    agent_out = [mock_agent_response(a, context) for a in agents]

    # 4) Weighted consensus
    weights = np.array([a["weight"] for a in agents], dtype=float); weights /= weights.sum()
    final_score = float(sum(w * r["score"] for w, r in zip(weights, agent_out)))
    final = {
        "question": queries[0]["query"],
        "citations": [r["id"] for r in ctx_rows],
        "consensus_score": final_score,
        "summary": "Integrated brief combining Researcher, Critic, and Synthesizer perspectives.",
    }

    # Assertions
    assert 0.0 <= final["consensus_score"] <= 1.0
    assert isinstance(final["summary"], str) and len(final["summary"]) > 10
    assert len(final["citations"]) == 3
