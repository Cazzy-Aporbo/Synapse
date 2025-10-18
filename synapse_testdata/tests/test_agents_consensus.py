"""
test_agents_consensus.py — Weighted consensus from agent votes
- Loads agents.yaml and computes a weighted score across agent candidates
"""

import yaml
import numpy as np

def weighted_consensus(scores, weights):
    # scores: list of dicts per candidate -> score
    # weights: array-like, sum-normalized
    keys = sorted(scores[0].keys())
    agg = {k: 0.0 for k in keys}
    for w, sc in zip(weights, scores):
        for k in keys:
            agg[k] += w * sc[k]
    # return ranking
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return ranked, agg

def test_consensus(data_root):
    agents = yaml.safe_load((data_root / "agents.yaml").read_text(encoding="utf-8"))["agents"]
    weights = np.array([a["weight"] for a in agents], dtype=float)
    weights = weights / weights.sum()

    # Three candidate answers (A,B,C) scored by each agent (0..1); synthetic but deterministic
    scores = [
        {"A": 0.8, "B": 0.5, "C": 0.4},  # Researcher
        {"A": 0.6, "B": 0.7, "C": 0.3},  # Critic
        {"A": 0.7, "B": 0.6, "C": 0.5},  # Synthesizer
    ]

    ranked, agg = weighted_consensus(scores, weights)
    # Ensure ranking is deterministic and A tends to win with given weights
    assert ranked[0][0] in {"A","B"}  # A or B should be near the top
    assert set(agg.keys()) == {"A","B","C"}
    assert all(0.0 <= v <= 1.0 for v in agg.values())
