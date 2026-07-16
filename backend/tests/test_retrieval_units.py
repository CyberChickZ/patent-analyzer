"""Offline unit tests for the retrieval layer — no network, no LLM.

Covers the deterministic pieces: candidate dedupe/merge, legacy doc
conversion, pdf url resolution, abstract reconstruction, scoring math.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.fetch_abstracts import reconstruct_abstract
from patent_analyzer.recall.pool import (
    Candidate,
    candidates_to_legacy_docs,
    pool_and_dedupe,
    resolve_pdf_url,
)
from patent_analyzer.scorer import compute_css_ewss


# ── resolve_pdf_url — the field-name regression that silently killed all PDF downloads ──

def test_resolve_pdf_url_prefers_pdf_link():
    assert resolve_pdf_url({"pdf_link": "http://x/a.pdf", "url": "http://x"}) == "http://x/a.pdf"


def test_resolve_pdf_url_falls_back_to_url():
    assert resolve_pdf_url({"pdf_link": "", "url": "http://x"}) == "http://x"


def test_resolve_pdf_url_flattens_list():
    assert resolve_pdf_url({"pdf_link": ["http://x/a.pdf", "http://x/b.pdf"]}) == "http://x/a.pdf"


def test_resolve_pdf_url_empty_inputs():
    assert resolve_pdf_url({}) == ""
    assert resolve_pdf_url({"pdf_link": [], "url": ""}) == ""


def test_legacy_docs_expose_fields_download_loop_reads():
    docs = candidates_to_legacy_docs(
        [Candidate(title="t" * 25, pdf_link="http://x/a.pdf", url="http://x")])
    assert resolve_pdf_url(docs[0]) == "http://x/a.pdf"


# ── pool_and_dedupe ──

def _cand(**kw):
    defaults = dict(title="a reasonably long candidate title here", match_type="Paper")
    defaults.update(kw)
    return Candidate(**defaults)


def test_dedupe_by_doi_across_channels():
    pooled = pool_and_dedupe({
        "ch1": [_cand(doi="10.1/x", source_score=1.0)],
        "ch2": [_cand(doi="10.1/X", title="totally different long title of same work",
                      abstract="full abstract", source_score=0.5)],
    })
    assert len(pooled) == 1
    c = pooled[0]
    assert sorted(c.sources) == ["ch1", "ch2"]
    # field backfill from second record
    assert c.abstract == "full abstract"
    # consensus bonus: max(1.0, 0.5) * sqrt(2)
    assert abs(c.source_score - 2 ** 0.5) < 1e-9


def test_dedupe_by_patent_number_ignores_spacing():
    pooled = pool_and_dedupe({
        "serp": [_cand(pub_num="US 1234567 B2", match_type="Patent")],
        "bq": [_cand(pub_num="US1234567B2", match_type="Patent",
                     title="another long enough patent title variant")],
    })
    assert len(pooled) == 1


def test_short_title_without_ids_is_dropped():
    pooled = pool_and_dedupe({"ch": [Candidate(title="short")]})
    assert pooled == []


def test_distinct_works_stay_distinct():
    pooled = pool_and_dedupe({
        "ch": [_cand(doi="10.1/a"), _cand(doi="10.1/b",
               title="second long candidate title that differs")],
    })
    assert len(pooled) == 2


# ── OpenAlex inverted-index reconstruction ──

def test_reconstruct_abstract_orders_positions():
    inv = {"world": [1], "hello": [0], "again": [2]}
    assert reconstruct_abstract(inv) == "hello world again"


def test_reconstruct_abstract_empty():
    assert reconstruct_abstract({}) == ""


# ── scoring math (deterministic layer) ──

def test_css_ewss_all_matched():
    css, ewss, n = compute_css_ewss({"a": {"score": 2}, "b": {"score": 2}})
    assert css == 1.0 and ewss == 1.0 and n == 2


def test_css_counts_absent_in_denominator_ewss_does_not():
    results = {"a": {"score": 2}, "b": {"score": 0}}
    css, ewss, n = compute_css_ewss(results)
    assert css == 0.5      # 1 of 2 criteria
    assert ewss == 1.0     # absent excluded from EWSS denominator
    assert n == 1


def test_binary_match_fallback():
    css, _, _ = compute_css_ewss({"a": {"match": True}, "b": {"match": False}})
    assert css == 0.5
