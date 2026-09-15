"""N7b: the open-access sources, and the ordering that decides which is asked first."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import fulltext as ft  # noqa: E402
from patent_analyzer import fulltext_sources as fs  # noqa: E402


# ── a PDF is %PDF, and nothing else ─────────────────────────────────────────

def test_only_the_magic_number_counts_as_a_pdf():
    """Measured 2026-09-19 on onlinelibrary.wiley.com: a Cloudflare page came
    back as HTTP 200 with Content-Type application/pdf. Trusting the header
    would have recorded a challenge page as a downloaded article."""
    assert fs.is_pdf(b"%PDF-1.7\nstuff")
    assert not fs.is_pdf(b"<!DOCTYPE html><title>Just a moment...</title>")
    assert not fs.is_pdf(b"")
    assert not fs.is_pdf(None)


# ── repository before publisher ─────────────────────────────────────────────

def test_repository_copies_are_offered_before_publisher_copies():
    payload = {"is_oa": True,
               "best_oa_location": {"url_for_pdf": "https://publisher.example/a.pdf",
                                    "host_type": "publisher"},
               "oa_locations": [{"url_for_pdf": "https://publisher.example/a.pdf",
                                 "host_type": "publisher"},
                                {"url_for_landing_page": "https://repo.example/a",
                                 "host_type": "repository"}]}
    rows = ft.oa_locations_ordered(payload)
    assert [r["host_type"] for r in rows] == ["repository", "publisher"]
    assert ft.oa_url_from_unpaywall(payload) == ("https://repo.example/a", "repository")


def test_a_pdf_still_beats_a_landing_page_inside_one_host_class():
    payload = {"is_oa": True,
               "best_oa_location": {"url_for_landing_page": "https://repo.example/abs/1",
                                    "host_type": "repository"},
               "oa_locations": [{"url_for_pdf": "https://repo.example/1.pdf",
                                 "url_for_landing_page": "https://repo.example/abs/1",
                                 "host_type": "repository"}]}
    assert ft.oa_url_from_unpaywall(payload)[0] == "https://repo.example/1.pdf"


def test_an_unknown_host_type_sorts_last_because_nothing_justifies_preferring_it():
    payload = {"is_oa": True,
               "best_oa_location": {"url_for_landing_page": "https://mystery.example/1"},
               "oa_locations": [{"url_for_pdf": "https://publisher.example/1.pdf",
                                 "host_type": "publisher"}]}
    assert ft.oa_url_from_unpaywall(payload)[0] == "https://publisher.example/1.pdf"


def test_a_repository_looking_url_is_ranked_as_one_even_without_the_field():
    assert ft.is_repository_url("https://arxiv.org/pdf/1706.03762")
    assert ft.is_repository_url("https://scholarworks.umass.edu/x/2109")
    assert ft.is_repository_url("https://www.osti.gov/servlets/purl/1076450")
    assert not ft.is_repository_url("https://onlinelibrary.wiley.com/doi/pdfdirect/10.1/x")


def test_the_plan_puts_the_apis_that_want_to_be_called_before_the_publisher():
    """The order is the finding, not a style choice: publisher hosts are where
    the bot management is, so they are asked last rather than first."""
    payload = {"is_oa": True,
               "best_oa_location": {"url_for_pdf": "https://publisher.example/a.pdf",
                                    "host_type": "publisher"},
               "oa_locations": [{"url_for_pdf": "https://repo.example/a.pdf",
                                 "host_type": "repository"}]}
    plan = ft.build_plan({"title": "t"}, "10.1/x", payload)
    srcs = [s["source"] for s in plan]
    assert srcs.index("repository") < srcs.index("bioc_pmc") < srcs.index("crossref_tdm")
    assert srcs.index("crossref_tdm") < srcs.index("publisher")
    assert srcs[-1] in ("elsevier_tdm", "wiley_tdm")


def test_a_preprint_doi_gets_a_direct_pdf_url_with_no_network_call():
    assert fs.preprint_pdf_url("10.1101/2020.04.06.20055475", "medRxiv preprint") == \
        "https://www.medrxiv.org/content/10.1101/2020.04.06.20055475v1.full.pdf"
    assert fs.preprint_pdf_url("10.1101/2020.09.16.299313", "") == \
        "https://www.biorxiv.org/content/10.1101/2020.09.16.299313v1.full.pdf"
    assert fs.preprint_pdf_url("10.1038/nature14539", "") == ""


# ── Crossref text and data mining links ─────────────────────────────────────

def test_only_text_mining_links_count_and_a_pdf_one_comes_first():
    """Measured on the six acceptance DOIs: Wiley publishes only a
    `similarity-checking` link, which is for plagiarism services and not for
    us. Treating it as a TDM link would fabricate a route that does not exist."""
    work = {"link": [
        {"intended-application": "similarity-checking", "URL": "https://x.example/sim"},
        {"intended-application": "text-mining", "URL": "https://x.example/tm.xml",
         "content-type": "text/xml"},
        {"intended-application": "text-mining", "URL": "https://x.example/tm.pdf",
         "content-type": "application/pdf"}]}
    links = fs.tdm_links(work)
    assert [L["url"] for L in links] == ["https://x.example/tm.pdf", "https://x.example/tm.xml"]
    assert fs.tdm_links({"link": [{"intended-application": "similarity-checking",
                                   "URL": "https://x.example/sim"}]}) == []
    assert fs.tdm_links(None) == []


# ── the landing page's own declaration ──────────────────────────────────────

@pytest.mark.parametrize("html,want", [
    ('<meta name="citation_pdf_url" content="https://p.example/a.pdf">',
     "https://p.example/a.pdf"),
    ("<meta content='https://p.example/b.pdf' name='citation_pdf_url'/>",
     "https://p.example/b.pdf"),
    ('<meta name="citation_title" content="Not the PDF">', ""),
])
def test_citation_pdf_url_is_read_in_either_attribute_order(html, want):
    assert fs.citation_pdf_url_in(html) == want


def test_a_relative_citation_pdf_url_is_resolved_against_the_landing_page():
    assert fs.citation_pdf_url_in('<meta name="citation_pdf_url" content="/x/a.pdf">',
                                  "https://p.example/article/1") == "https://p.example/x/a.pdf"


# ── BioC, and the 429 that must not be filed under "unavailable" ────────────

def test_bioc_text_keeps_the_passages_and_drops_the_markup():
    xml = (b'<collection><document><passage><infon key="type">title</infon>'
           b'<text>A title about things</text></passage>'
           b'<passage><text>Body with an &amp; entity</text></passage></document></collection>')
    out = fs.bioc_to_text(xml)
    assert "A title about things" in out and "Body with an & entity" in out
    assert "<" not in out


def test_bioc_reports_rate_limiting_as_its_own_reason(monkeypatch):
    """429 means come back later; "no PMCID" means this article is not there.
    Counting them together would understate the tier's real coverage."""
    async def _pmcid(doi):
        return "PMC1", ""

    async def _get(url, headers=None, identify=True):
        return fs.Resp(429, "text/html", b"slow down", url)
    monkeypatch.setattr(fs, "pmcid_for_doi", _pmcid)
    monkeypatch.setattr(fs, "get", _get)
    text, why = asyncio.run(fs.bioc_pmc_text("10.1/x"))
    assert text == "" and "429" in why and "not unavailable" in why


def test_the_credentialled_tiers_say_skipped_not_failed(monkeypatch):
    """No token is not the same answer as "the publisher refused". The wording
    matters because these two lines are the library TDM request."""
    monkeypatch.setattr(fs, "WILEY_TDM_TOKEN", "")
    monkeypatch.setattr(fs, "ELSEVIER_TDM_KEY", "")
    monkeypatch.setattr(fs, "CORE_API_KEY", "")
    assert "skipped" in asyncio.run(fs.wiley_tdm_pdf("10.1/x"))[1]
    assert "skipped" in asyncio.run(fs.elsevier_tdm_text("10.1/x"))[1]
    assert "skipped" in asyncio.run(fs.core_pdf_url("10.1/x"))[1]


# ── acquisition walks the plan, and tells the truth about what it got ───────

def test_acquire_stops_at_the_first_route_that_returns_real_bytes(monkeypatch):
    tried = []

    async def _get_pdf(url, headers=None):
        tried.append(url)
        if "repo" in url:
            return None, "HTTP 404"
        return b"%PDF-1.4 real", "12 bytes"
    monkeypatch.setattr(fs, "get_pdf", _get_pdf)

    async def _bioc(doi):
        return "", "no PMCID for this DOI"
    monkeypatch.setattr(fs, "bioc_pmc_text", _bioc)

    patch = {"doi": "10.1/x", "landing_page": "", "fulltext_plan": [
        {"source": "repository", "url": "https://repo.example/a.pdf", "tier": "oa", "detail": "r"},
        {"source": "bioc_pmc", "tier": "oa", "detail": "b"},
        {"source": "publisher", "url": "https://pub.example/a.pdf", "tier": "oa", "detail": "p"}]}
    got = asyncio.run(ft.acquire({}, patch))
    assert got["fulltext_download"] == "ok" and got["fulltext_source"] == "publisher"
    assert tried == ["https://repo.example/a.pdf", "https://pub.example/a.pdf"]
    assert [a["source"] for a in got["attempts"]] == ["repository", "bioc_pmc", "publisher"]


def test_acquire_reports_api_full_text_as_ok_text_not_as_a_download(monkeypatch):
    async def _bioc(doi):
        return "x" * 4000, ""
    monkeypatch.setattr(fs, "bioc_pmc_text", _bioc)
    patch = {"doi": "10.1/x", "fulltext_plan": [{"source": "bioc_pmc", "tier": "oa",
                                                 "detail": "NCBI BioC API"}]}
    got = asyncio.run(ft.acquire({}, patch))
    assert got["fulltext_download"] == "ok_text" and got["pdf"] is None
    assert len(got["text"]) == 4000


def test_acquire_with_nothing_to_try_says_no_url_not_failed():
    got = asyncio.run(ft.acquire({}, {"doi": "", "fulltext_plan": []}))
    assert got["fulltext_download"] == "no_url"


def test_resolution_is_not_a_claim_that_anything_was_read():
    """The whole reason this module reports two fields. A plan made only of
    speculative API routes is not a copy found, so the tier stays
    abstract_only; if one of those routes delivers, it shows up in
    fulltext_download instead."""
    plan_only = ft.build_plan({"title": "t"}, "10.1/x", {"is_oa": False})
    assert plan_only and not any(s.get("url") for s in plan_only)
