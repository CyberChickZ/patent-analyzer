"""Offline tests for the production rerank path — no network, no model load.

Fake embedders stand in for Vertex/MiniLM so the tests exercise chunking,
max-pool ranking, and the fallback logic deterministically.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import encoders, semantic_search
from patent_analyzer.semantic_search import doc_to_chunks, rerank_docs


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.array([1.0, 0.0]) if "quantum" in t.lower() else np.array([0.0, 1.0])
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


# ── doc_to_chunks ──

def test_chunks_title_abstract_only():
    chunks = doc_to_chunks({"title": "T", "abstract": "some abstract text"})
    assert chunks == ["T some abstract text"]


def test_chunks_split_numbered_claims():
    claims = ("1. A method for encoding data, comprising a first step of things. "
              "2. The method of claim 1, wherein the encoding uses a codec module. "
              "3. An apparatus configured to perform the method of claim 1 entirely.")
    chunks = doc_to_chunks({"title": "T", "abstract": "A", "claims_text": claims})
    assert len(chunks) == 4
    assert chunks[1].startswith("1. A method")
    assert chunks[2].startswith("2. The method")


def test_chunks_window_fallback_for_unnumbered_text():
    blob = "word " * 400
    chunks = doc_to_chunks({"title": "T", "abstract": "A", "claims_text": blob})
    assert len(chunks) > 2
    assert all(len(c) <= 1000 for c in chunks[1:])


def test_chunks_capped():
    claims = " ".join(f"{i}. A distinct claim number {i} about widgets and gadgets."
                      for i in range(1, 40))
    chunks = doc_to_chunks({"title": "T", "abstract": "A", "claims_text": claims}, max_chunks=12)
    assert len(chunks) == 12


# ── rerank_docs: vertex path with max-pool ──

def test_claim_chunk_match_wins_via_maxpool(monkeypatch):
    monkeypatch.setattr(encoders, "embed_docs", lambda texts, model=None: fake_embed(texts))
    monkeypatch.setattr(encoders, "embed_queries", lambda texts, model=None: fake_embed(texts))
    docs = [
        {"title": "Plain doc", "abstract": "about classical widgets"},
        {"title": "Hidden gem", "abstract": "boring abstract",
         "claims_text": "1. A widget thing entirely mundane. 2. The widget of claim 1 using quantum tunneling."},
    ]
    ranked = rerank_docs("quantum widget", docs, limit=2)
    assert ranked[0]["title"] == "Hidden gem"
    assert ranked[0]["rerank_encoder"] == encoders.VERTEX_MODEL
    assert "hybrid_score" not in ranked[0]


# ── rerank_docs: fallback path ──

def test_fallback_to_minilm_with_fusion(monkeypatch):
    def boom(texts, model=None):
        raise RuntimeError("vertex down")
    monkeypatch.setattr(encoders, "embed_docs", boom)
    monkeypatch.setattr(semantic_search, "embed_texts", fake_embed)
    docs = [
        {"title": "Plain doc", "abstract": "about classical widgets"},
        {"title": "Quantum doc", "abstract": "quantum tunneling widgets"},
    ]
    ranked = rerank_docs("quantum widget", docs, limit=2)
    assert ranked[0]["title"] == "Quantum doc"
    assert ranked[0]["rerank_encoder"] == "minilm+bm25"
    assert "hybrid_score" in ranked[0]


def test_empty_documents():
    assert rerank_docs("anything", []) == []
