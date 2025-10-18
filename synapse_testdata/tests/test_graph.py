"""
test_graph.py — Graph integrity checks for Synapse
- Loads nodes/edges from knowledge_graph.json
- Asserts node/edge counts, required attributes, and basic connectivity
"""

import pytest

def test_graph_integrity(knowledge_graph):
    nodes = knowledge_graph["nodes"]
    edges = knowledge_graph["edges"]

    assert isinstance(nodes, list) and isinstance(edges, list)
    assert len(nodes) >= 5, "Expect at least 5 nodes in the mini pack"
    assert len(edges) >= 4, "Expect some connectivity"

    # Nodes must have id, title, text, tags
    for n in nodes:
        assert all(k in n for k in ("id", "title", "text", "tags"))
        assert isinstance(n["tags"], list)

    # Edges must have source, target, weight in [0,1]
    for e in edges:
        assert all(k in e for k in ("source", "target", "weight"))
        assert 0.0 <= e["weight"] <= 1.0
