"""Offline tests for hybrid retrieval (BM25 + RRF) and the production rerank path."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.hybrid import bm25_scores, rrf_fuse, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("GRE-Message, Encapsulation!") == ["gre", "message", "encapsulation"]


def test_bm25_prefers_matching_doc():
    docs = ["gre message encapsulation method",
            "cooking recipes for pasta",
            "message forwarding using gre tunneling"]
    scores = bm25_scores("gre encapsulation", docs)
    assert scores[0] == max(scores)
    assert scores[1] == 0.0


def test_bm25_rare_term_outweighs_common():
    docs = ["alpha beta", "alpha gamma", "alpha delta", "epsilon beta"]
    scores = bm25_scores("epsilon", docs)
    assert np.argmax(scores) == 3


def test_bm25_empty_inputs():
    assert list(bm25_scores("", ["a doc"])) == [0.0]
    assert len(bm25_scores("query", [])) == 0


def test_rrf_consensus_wins():
    # doc0 is #1 in both rankings; doc1 and doc2 each win one
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([30.0, 10.0, 20.0])
    fused = rrf_fuse([a, b])
    assert np.argmax(fused) == 0


def test_rrf_is_scale_invariant():
    a = np.array([0.9, 0.5, 0.1])
    b = np.array([1.0, 3.0, 2.0])
    assert np.allclose(rrf_fuse([a, b]), rrf_fuse([a * 1000, b * 0.001]))


def test_rerank_hybrid_sparse_overturns_dense(monkeypatch):
    """Dense mildly prefers the wrong doc; BM25 strongly prefers the
    term-matching doc. Fusion must let the sparse signal win — and
    hybrid_score must be populated."""
    from patent_analyzer import semantic_search

    def fake_embed(texts):
        vecs = []
        for t in texts:
            if "Pasta" in t:
                vecs.append([1.0, 0.0])        # dense sim 1.0 (wrong doc)
            elif "GRE" in t:
                vecs.append([0.9, 0.436])      # dense sim 0.9
            elif "Bicycle" in t:
                vecs.append([0.5, 0.866])      # dense sim 0.5
            else:
                vecs.append([1.0, 0.0])        # the query itself
        return np.array(vecs)

    monkeypatch.setattr(semantic_search, "embed_texts", fake_embed)
    docs = [
        {"title": "Pasta cooking device", "abstract": "boiling water control"},
        {"title": "GRE message encapsulation", "abstract": "tunnel header attribute setting"},
        {"title": "Bicycle frame", "abstract": "lightweight alloy header"},
    ]
    ranked = semantic_search.rerank_hybrid("gre encapsulation tunnel header", docs, limit=3)
    assert ranked[0]["title"] == "GRE message encapsulation"
    assert "hybrid_score" in ranked[0] and "semantic_score" in ranked[0]


def test_rerank_hybrid_empty_docs():
    from patent_analyzer.semantic_search import rerank_hybrid
    assert rerank_hybrid("anything", []) == []
