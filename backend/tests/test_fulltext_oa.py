import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import fulltext as ft  # noqa: E402
from patent_analyzer.report_sections import fulltext_tier_html, fulltext_tier_md  # noqa: E402
from patent_analyzer.runtime_state import day_key  # noqa: E402


def test_day_key_is_a_utc_calendar_day():
    from datetime import datetime, timezone
    assert day_key(datetime(2026, 9, 18, 23, 59, tzinfo=timezone.utc)) == "2026-09-18"
    assert len(day_key()) == 10


@pytest.mark.parametrize("raw,want", [
    ("10.1038/nature14539", "10.1038/nature14539"),
    ("https://doi.org/10.1038/Nature14539", "10.1038/nature14539"),
    ("doi:10.1016/j.cell.2011.02.013", "10.1016/j.cell.2011.02.013"),
    ("Gatti et al., 10.1234/abcd.5678, J. Trauma", "10.1234/abcd.5678"),
    ("US-1234567-B2", ""),
    ("", ""),
    (None, ""),
])
def test_normalise_doi(raw, want):
    assert ft.normalise_doi(raw) == want


@pytest.mark.parametrize("vals,want", [
    (("1706.03762",), "1706.03762"),
    (("arXiv:2301.00234",), "2301.00234"),
    ((None, "https://arxiv.org/abs/1706.03762v5"), "1706.03762"),
    ((None, "https://arxiv.org/pdf/2301.00234.pdf"), "2301.00234"),
    (("cs.CV/0701001",), "cs.CV/0701001"),
    (("10.1038/nature14539",), ""),
])
def test_arxiv_id_of(vals, want):
    assert ft.arxiv_id_of(*vals) == want


def test_doi_slug_is_path_safe_and_stable():
    assert ft.doi_slug("10.1038/Nature14539") == "10.1038_nature14539"
    assert ft.doi_slug("https://doi.org/10.1038/nature14539") == ft.doi_slug("10.1038/nature14539")
    assert "/" not in ft.doi_slug("10.1145/3292500.3330701")


def test_unpaywall_falls_through_when_url_for_pdf_is_null():
    """Measured on 10.1038/nature14539: is_oa true, host_type repository,
    url_for_pdf null. Reading only url_for_pdf loses a real OA copy."""
    payload = {"is_oa": True,
               "best_oa_location": {"url_for_pdf": None, "url": "https://hal.science/hal-04206682",
                                    "host_type": "repository"},
               "oa_locations": []}
    assert ft.oa_url_from_unpaywall(payload) == ("https://hal.science/hal-04206682", "repository")


def test_unpaywall_prefers_a_pdf_in_any_location_over_a_landing_page():
    payload = {"is_oa": True,
               "best_oa_location": {"url_for_landing_page": "https://x.example/abs/1"},
               "oa_locations": [{"url_for_landing_page": "https://x.example/abs/1"},
                                {"url_for_pdf": "https://y.example/1.pdf", "host_type": "publisher"}]}
    assert ft.oa_url_from_unpaywall(payload) == ("https://y.example/1.pdf", "publisher")


def test_unpaywall_closed_and_missing_give_nothing():
    assert ft.oa_url_from_unpaywall({"is_oa": False, "best_oa_location": None}) == ("", "")
    assert ft.oa_url_from_unpaywall(None) == ("", "")


def test_resolve_prefers_arxiv_without_any_network_call(monkeypatch):
    async def _boom(doi):
        raise AssertionError("Unpaywall must not be called for an arXiv paper")
    monkeypatch.setattr(ft, "unpaywall", _boom)
    p = asyncio.run(ft.resolve({"pub_num": "1706.03762", "title": "Attention Is All You Need"}))
    assert p["fulltext_tier"] == "arxiv"
    assert p["fulltext_url"] == "https://arxiv.org/pdf/1706.03762"


def test_resolve_uses_the_channel_oa_field_before_unpaywall(monkeypatch):
    async def _boom(doi):
        raise AssertionError("an OA URL was already on the candidate")
    monkeypatch.setattr(ft, "unpaywall", _boom)
    p = asyncio.run(ft.resolve({"pub_num": "10.1145/3292500.3330701",
                                "pdf_link": "https://oa.example/paper.pdf", "title": "t"}))
    assert p["fulltext_tier"] == "oa"
    assert p["fulltext_url"] == "https://oa.example/paper.pdf"
    assert p["doi"] == "10.1145/3292500.3330701"


def test_resolve_falls_to_unpaywall_then_abstract_only(monkeypatch):
    calls = []

    async def _fake(doi):
        calls.append(doi)
        return ({"is_oa": True, "best_oa_location": {"url_for_pdf": "https://r.example/a.pdf",
                                                     "host_type": "repository"}}, None)
    monkeypatch.setattr(ft, "unpaywall", _fake)
    p = asyncio.run(ft.resolve({"pub_num": "10.1016/j.cell.2011.02.013", "title": "t"}))
    assert p["fulltext_tier"] == "oa" and calls == ["10.1016/j.cell.2011.02.013"]

    async def _closed(doi):
        return ({"is_oa": False}, None)
    monkeypatch.setattr(ft, "unpaywall", _closed)
    p = asyncio.run(ft.resolve({"pub_num": "10.1016/j.cell.2011.02.013", "title": "t",
                                "url": "https://www.cell.com/x"}))
    assert p["fulltext_tier"] == "abstract_only"
    assert p["landing_page"] == "https://www.cell.com/x"


def test_resolve_without_a_doi_never_asks_unpaywall(monkeypatch):
    async def _boom(doi):
        raise AssertionError("no DOI to ask about")
    monkeypatch.setattr(ft, "unpaywall", _boom)
    p = asyncio.run(ft.resolve({"pub_num": "US-1234567-B2", "title": "t"}))
    assert p["fulltext_tier"] == "abstract_only"
    assert p["landing_page"] == ""


def test_manifest_lists_every_paper_without_a_pdf_and_carries_the_policy():
    """Membership is decided on local_pdf, not on the tier: a paper can resolve
    to an open-access URL and still arrive with nothing (measured: 6 of the 17
    NPL gold resolved, 0 returned a PDF). A tier-only list would omit those."""
    docs = [{"title": "A", "local_pdf": "/tmp/a.pdf"},
            {"title": "B"},
            {"title": "C", "fulltext_download": "failed"}]
    patches = [{"fulltext_tier": "arxiv"},
               {"fulltext_tier": "abstract_only", "doi": "10.1/x",
                "landing_page": "https://p.example/x", "fulltext_detail": "Unpaywall: not open access"},
               {"fulltext_tier": "oa", "doi": "10.1/c", "landing_page": "https://p.example/c",
                "fulltext_detail": "Unpaywall (publisher)"}]
    rows = ft.manifest_rows(docs, patches)
    assert [r["title"] for r in rows] == ["B", "C"]
    assert "returned no readable PDF" in rows[1]["reason"]
    md = ft.manifest_markdown(rows)
    assert "10.1/x" in md and "https://p.example/x" in md
    # The reason the proxied tier does not exist travels with the list.
    assert "scripts, spiders, crawlers" in md
    assert "entire OSU community" in md


def test_read_counts_only_counts_documents_that_hold_a_pdf():
    docs = [{"local_pdf": "/tmp/a.pdf"}, {}, {"local_pdf": "/tmp/c.pdf"}]
    patches = [{"fulltext_tier": "arxiv"}, {"fulltext_tier": "oa"}, {"fulltext_tier": "oa"}]
    assert ft.read_counts(docs, patches) == {"arxiv": 1, "oa": 1, "abstract_only": 0}


def test_tier_counts_covers_every_tier():
    counts = ft.tier_counts([{"fulltext_tier": "arxiv"}, {"fulltext_tier": "arxiv"},
                             {"fulltext_tier": "abstract_only"}, {}])
    assert counts == {"arxiv": 2, "oa": 0, "abstract_only": 2}


def test_gcs_cache_is_silent_when_there_is_no_bucket(monkeypatch):
    def _boom(doi):
        raise RuntimeError("no credentials")
    monkeypatch.setattr(ft, "_blob", _boom)
    assert ft.cache_get("10.1/x") is None
    assert ft.cache_put("10.1/x", b"%PDF-1.4") is False
    assert ft.cache_get("") is None and ft.cache_put("10.1/x", b"") is False


def test_cache_uri_is_keyed_by_doi():
    assert ft.cache_uri("10.1038/nature14539").endswith("/fulltext/10.1038_nature14539.pdf")
    assert ft.cache_uri("") == ""


def test_report_section_is_silent_without_papers():
    assert fulltext_tier_html(None) == "" and fulltext_tier_md({}) == []
    assert fulltext_tier_html({"fulltext_oa": {"papers": 0}}) == ""


def test_report_section_states_the_tiers_and_the_policy():
    ss = {"fulltext_oa": {"papers": 4, "arxiv": 1, "oa": 1, "abstract_only": 2,
                          "read": {"arxiv": 1, "oa": 0, "abstract_only": 0},
                          "manifest": [{"doi": "10.1/x", "title": "A paper",
                                        "landing_page": "https://p.example/x",
                                        "reason": "Unpaywall: not open access"}]}}
    h, m = fulltext_tier_html(ss), "\n".join(fulltext_tier_md(ss))
    for blob in (h, m):
        assert "25%" in blob and "50%" in blob
        # resolved (2) and read (1) are different numbers and both are printed
        assert "1 of 4 (25%)" in blob
        assert "entire OSU community" in blob
        assert "library.oregonstate.edu/responsible-use-licensed-electronic-resources" in blob
        assert "10.1/x" in blob
