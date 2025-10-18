"""
test_citations.py — Citation mapping
- Ensures each citation id resolves to a DOI or URL
"""

import pandas as pd

def test_citation_map(data_root, papers_df):
    cit = pd.read_csv(data_root / "citations.csv")
    have = set(cit["id"].tolist())
    need = set(papers_df["id"].tolist())
    # All papers in this mini pack should have a mapping
    assert need.issubset(have), f"Missing citation for: {need - have}"

    # Spot check the columns exist
    assert {"id","doi","url"}.issubset(set(cit.columns))
    # Ensure at least one DOI and one URL present
    assert cit["doi"].notna().any() and cit["url"].notna().any()
