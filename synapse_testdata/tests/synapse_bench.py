#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synapse Advanced Benchmark Harness (offline, single-file)
--------------------------------------------------------
Features:
- Deterministic seeding & reproducibility
- Pluggable vector index (sklearn NearestNeighbors)
- Retrieval metrics: Precision@k, Recall@k, MRR, nDCG@k
- Lightweight RAG: retrieve top-k, build context, simulate 3-agent responses
- Consensus: weighted + Borda fallback (utility provided)
- Reports: JSONL per-query + CSV summary + pretty console print
- CLI options for k, seed, temperature, weights, metric, out dir

Usage:
  python synapse_bench.py --data ./synapse_testdata --k 3 --out ./bench_out

Assumes 'synapse_testdata' structure from the generated test pack.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ------------------------- Config Types -------------------------
@dataclass
class BenchConfig:
    data_root: Path
    k: int = 3
    seed: int = 42
    out_dir: Path = Path("./bench_out")
    temperature: float = 0.3
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)  # Researcher, Critic, Synthesizer
    index_metric: str = "cosine"  # "cosine" or "euclidean"
    queries_file: str = "queries.jsonl"
    papers_file: str = "papers.jsonl"
    embeddings_file: str = "embeddings.npy"
    agents_file: str = "agents.yaml"  # optional; not strictly required in this harness

    def normalize(self) -> None:
        w = np.array(self.weights, dtype=float)
        w = np.maximum(w, 1e-9)
        self.weights = tuple((w / w.sum()).tolist())  # type: ignore


# ------------------------- Index -------------------------
class VectorIndex:
    """Simple vector index over embeddings with cosine/euclidean distance."""

    def __init__(self, X: np.ndarray, metric: str = "cosine"):
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.X = X.astype(np.float32, copy=False)
        self.metric = metric
        self.nn = NearestNeighbors(metric=metric, n_neighbors=min(10, len(X)))
        self.nn.fit(self.X)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if q.ndim == 1:
            q = q[None, :]
        dist, idx = self.nn.kneighbors(q, n_neighbors=min(k, len(self.X)), return_distance=True)
        if self.metric == "cosine":
            sim = 1.0 - dist[0]
        else:
            sim = -dist[0]  # smaller dist => higher similarity
        return idx[0], sim


# ------------------------- Mock Components -------------------------
def stub_embed(text: str, dim: int, seed: int) -> np.ndarray:
    """Deterministic pseudo-embedding from text + seed (no external models)."""
    val = sum(ord(c) for c in text[:32]) + seed * 131
    rng = np.random.default_rng(val)
    return rng.normal(0, 1, size=dim).astype(np.float32)


def build_context(rows: pd.DataFrame) -> str:
    parts = []
    for r in rows.itertuples():
        url = getattr(r, "url", "")
        parts.append(f"[{r.id}] {r.title}\n{r.abstract}\n{url}".strip())
    return "\n\n---\n\n".join(parts)


def agent_score(name: str, ctx_rows: pd.DataFrame, temperature: float, seed: int) -> float:
    """Heuristic agent score: longer abstracts & diversity help; role adds bias; jitter by temperature."""
    base = float(sum(len(a) for a in ctx_rows["abstract"]) / 1000.0)
    diversity = float(ctx_rows["title"].str.len().std(ddof=0) / 100.0)
    role_bias = {"Researcher": 0.06, "Critic": 0.03, "Synthesizer": 0.04}.get(name, 0.04)
    rng = np.random.default_rng(len(name) * 97 + seed)
    noise = float(rng.normal(0, 0.06 * max(0.05, temperature)))
    score = base + diversity + role_bias + noise
    return float(max(0.0, min(1.0, score)))


def weighted_consensus(scores: List[float], weights: Tuple[float, float, float]) -> float:
    w = np.array(weights, dtype=float)
    s = np.array(scores, dtype=float)
    return float(np.dot(w / w.sum(), s))


def borda_consensus(ranks: List[List[int]], m: int) -> List[int]:
    """Borda count aggregation for agent rankings of m candidates (utility)."""
    scores = np.zeros(m)
    for r in ranks:
        for pos, idx in enumerate(r):
            scores[idx] += (m - pos - 1)
    return list(np.argsort(-scores))


# ------------------------- Retrieval Metrics -------------------------
def precision_at_k(gt: List[int], pred: List[int]) -> float:
    return float(len(set(gt) & set(pred[: len(pred)])) / max(1, len(pred)))


def recall_at_k(gt: List[int], pred: List[int]) -> float:
    return float(len(set(gt) & set(pred[: len(pred)])) / max(1, len(gt)))


def mrr_at_k(gt: List[int], pred: List[int]) -> float:
    for i, p in enumerate(pred):
        if p in gt:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(gt: List[int], pred: List[int]) -> float:
    def dcg(items):
        return sum((1.0 / math.log2(i + 2)) for i, x in enumerate(items) if x in gt)
    ideal = dcg(gt[: len(pred)])
    return float(dcg(pred) / max(1e-9, ideal))


# ------------------------- Benchmark Runner -------------------------
def run_benchmark(cfg: BenchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Loading data from %s", cfg.data_root)
    papers = [json.loads(l) for l in (cfg.data_root / cfg.papers_file).read_text(encoding="utf-8").splitlines()]
    papers_df = pd.DataFrame(papers)
    emb = np.load(cfg.data_root / cfg.embeddings_file)
    queries = [json.loads(l) for l in (cfg.data_root / cfg.queries_file).read_text(encoding="utf-8").splitlines()]

    # Build index
    index = VectorIndex(emb, metric=cfg.index_metric)

    # Weak GT via keyword overlap: tokens from query vs paper keywords
    kw = papers_df["keywords"].apply(lambda ks: set([k.lower() for k in ks]) if isinstance(ks, list) else set()).tolist()

    summary_rows = []
    detailed_rows = []
    for qi, q in enumerate(queries, 1):
        qtext = q["query"]
        qdim = emb.shape[1]
        qvec = stub_embed(qtext, dim=qdim, seed=cfg.seed + qi)
        idx, sim = index.search(qvec, k=cfg.k)

        # Ground-truth signal: overlap of tokens and paper keywords
        toks = set(t.strip("?,.!:;").lower() for t in qtext.split())
        gt = [i for i, kws in enumerate(kw) if len(toks & kws) > 0]

        prec = precision_at_k(gt, idx.tolist())
        rec = recall_at_k(gt, idx.tolist())
        mrr = mrr_at_k(gt, idx.tolist())
        ndcg = ndcg_at_k(gt, idx.tolist())

        ctx_rows = papers_df.iloc[idx][["id", "title", "abstract"]]
        context = build_context(ctx_rows)

        agents = ["Researcher", "Critic", "Synthesizer"]
        scores = [agent_score(n, ctx_rows, cfg.temperature, seed=cfg.seed + qi) for n in agents]
        consensus = weighted_consensus(scores, cfg.weights)

        detailed_rows.append({
            "qid": q["qid"],
            "query": qtext,
            "ret_indices": idx.tolist(),
            "ret_scores": sim.tolist(),
            "precision@k": prec,
            "recall@k": rec,
            "mrr@k": mrr,
            "ndcg@k": ndcg,
            "agent_scores": dict(zip(agents, scores)),
            "consensus_score": consensus,
            "citations": papers_df.iloc[idx]["id"].tolist(),
            "context_preview": context[:300] + ("..." if len(context) > 300 else ""),
        })

        summary_rows.append({
            "qid": q["qid"],
            "precision_at_k": prec,
            "recall_at_k": rec,
            "mrr_at_k": mrr,
            "ndcg_at_k": ndcg,
            "consensus": consensus,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_detailed = pd.DataFrame(detailed_rows)
    return df_summary, df_detailed


def main() -> None:
    ap = argparse.ArgumentParser(description="Synapse advanced benchmark harness (offline)")
    ap.add_argument("--data", required=True, help="Path to synapse_testdata folder")
    ap.add_argument("--k", type=int, default=3, help="Top-k for retrieval (default: 3)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--out", default="./bench_out", help="Output directory (default: ./bench_out)")
    ap.add_argument("--temperature", type=float, default=0.3, help="Agent jitter scale (default: 0.3)")
    ap.add_argument("--weights", type=float, nargs=3, default=(0.5, 0.3, 0.2), help="Weights for Researcher, Critic, Synthesizer")
    ap.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine", help="Distance metric for NN")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = BenchConfig(
        data_root=Path(args.data),
        k=int(args.k),
        seed=int(args.seed),
        out_dir=Path(args.out),
        temperature=float(args.temperature),
        weights=(args.weights[0], args.weights[1], args.weights[2]),
        index_metric=args.metric,
    )
    cfg.normalize()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary, detailed = run_benchmark(cfg)
    except Exception as e:
        logging.error("Benchmark failed: %s", e)
        sys.exit(1)

    # Save outputs
    summary_path = cfg.out_dir / "summary.csv"
    detailed_path = cfg.out_dir / "results.jsonl"
    summary.to_csv(summary_path, index=False)
    with open(detailed_path, "w", encoding="utf-8") as f:
        for rec in detailed.to_dict(orient="records"):
            f.write(json.dumps(rec) + "\n")

    # Pretty console print
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print("\n=== Retrieval & Consensus Summary ===")
        print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}\n       {detailed_path}")


if __name__ == "__main__":
    main()
