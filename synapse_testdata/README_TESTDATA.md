# Synapse Test Data (Mini Pack)

Use this folder for unit/integration tests without hitting real APIs.

## Files
- `knowledge_graph.json` — nodes (id,title,text,tags) + edges (source,target,weight).
- `papers.jsonl` — 5 small paper records.
- `embeddings.npy` — 5x16 float32 embeddings (P1..P5 order).
- `queries.jsonl` — 3 sample questions with top-k.
- `agents.yaml` — example multi-agent config (Researcher, Critic, Synthesizer).
- `citations.csv` — quick lookup for doi/url.
- `env.example` — stub secrets.

## Typical test flows
1. **Knowledge graph load**: parse nodes/edges, build NetworkX graph, assert node/edge counts.

2. **Vector search stub**: load `embeddings.npy`, compute cosine similarity for query vectors (use deterministic random for query seed), assert top-k ids.

3. **RAG pipeline**: map query -> top papers -> create short context -> mock LLM response -> assert output schema.

4. **Consensus**: create 3 fake agent answers with given weights in `agents.yaml`, compute weighted score, assert ranking stability.

5. **Citations**: ensure outputs include an id that maps to `citations.csv`.

## Example (Python)
```python
import json, numpy as np, pandas as pd
from pathlib import Path

root = Path("synapse_testdata")
nodes = json.loads((root/"knowledge_graph.json").read_text())["nodes"]
emb = np.load(root/"embeddings.npy")

assert len(nodes) == 5 and emb.shape == (5,16)
```

Tip: if FAISS isn't installed, use `sklearn.neighbors.NearestNeighbors(metric="cosine")` on `embeddings.npy` for tests.
