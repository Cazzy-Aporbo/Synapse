"""
Pytest configuration and shared fixtures for Synapse tests.
Looks for `synapse_testdata` next to the project root or in the working directory.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from pathlib import Path

@pytest.fixture(scope="session")
def data_root() -> Path:
    # try local folder first
    candidates = [
        Path("synapse_testdata"),
        Path("./tests/synapse_testdata"),
        Path(__file__).resolve().parent / "synapse_testdata",
        Path.cwd() / "synapse_testdata",
        Path("/mnt/data/synapse_testdata"),  # notebook runtime default
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate synapse_testdata folder. Place it at project root or tests/.")

@pytest.fixture(scope="session")
def papers_df(data_root: Path) -> pd.DataFrame:
    # load JSONL into DataFrame
    lines = (data_root / "papers.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return pd.DataFrame([json.loads(l) for l in lines])

@pytest.fixture(scope="session")
def embeddings(data_root: Path) -> np.ndarray:
    return np.load(data_root / "embeddings.npy")

@pytest.fixture(scope="session")
def knowledge_graph(data_root: Path) -> dict:
    return json.loads((data_root / "knowledge_graph.json").read_text(encoding="utf-8"))

@pytest.fixture(scope="session")
def queries(data_root: Path):
    lines = (data_root / "queries.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines]
