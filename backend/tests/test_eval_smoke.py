"""Smoke test for the FiNE-Patents retrieval eval.

Skipped automatically when the dataset isn't unpacked locally
(see backend/evals/README.md for setup). Keeps the benchmark harness
itself under test: loading, corpus building, BM25 scoring, metrics.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from retrieval_eval import DATA_DIR, build_corpus, eval_bm25, load_samples

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="FiNE-Patents dataset not unpacked (see evals/README.md)")


def test_loads_samples_with_required_fields():
    samples = load_samples(limit=10, split=None)
    assert len(samples) == 10
    for s in samples:
        assert s["query_claim"].strip()
        assert s["target"]


def test_bm25_beats_chance_on_small_slice():
    samples = load_samples(limit=30, split=None)
    corpus = build_corpus(samples)
    m = eval_bm25(samples, corpus)
    assert m["n"] == 30
    # Chance level for R@10 on a ~30-doc corpus is ~1/3; BM25 should be far above.
    assert m["recall@10"] >= 0.6
    assert 0.0 < m["mrr"] <= 1.0
