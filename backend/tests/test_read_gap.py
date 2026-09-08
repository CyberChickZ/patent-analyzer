"""Delivered vs deep-read must add up, and the difference must be on the page.

The report said "25 references evaluated" while 17-24 of those 25 had been read
from nothing and another 35 delivered references were never sent to a model at
all. Three numbers were being spoken of as one. These tests pin the arithmetic
(read + every itemised reason == delivered) and the fact that a shortfall is
rendered rather than inferred.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import graph.eval_subgraph as ev
from patent_analyzer.recall import bigquery_patents as bq
from patent_analyzer.report_sections import inject_html, inject_md, read_gap_html, read_gap_md


def test_read_gap_accounts_for_every_delivered_reference():
    delivered = [{"pub_num": f"US-{i}-A1"} for i in range(60)]
    fanned = ([{"source": "full_text"}] * 8
              + [{"source": "abstract"}] * 4
              + [{"source": "no_content", "no_content_reason": "no claims in amie_patents"}] * 12
              + [{"source": "pdf", "is_source_duplicate": True}])
    scoring = [r for r in fanned if not r.get("is_source_duplicate")]

    g = ev._read_gap(delivered, fanned, scoring)

    assert g["delivered"] == 60 and g["fanned_out"] == 25 and g["read"] == 12
    assert g["shortfall"] == 48
    # nothing hides in the difference: the reasons account for all of it
    assert sum(g["reasons"].values()) == g["shortfall"]
    assert g["reasons"][f"never sent to a model (EVAL_MAX_DOCS={ev.MAX_EVAL})"] == 35
    assert g["reasons"]["dropped as a duplicate of the source document"] == 1
    assert g["reasons"]["no claims in amie_patents"] == 12
    assert "Delivered 60 references, deep-read 12" in g["headline"]


def test_read_gap_is_quiet_only_when_everything_was_read():
    delivered = [{"pub_num": "US-1-A1"}, {"pub_num": "US-2-A1"}]
    fanned = [{"source": "full_text"}, {"source": "pdf"}]
    g = ev._read_gap(delivered, fanned, fanned)
    assert g["shortfall"] == 0 and g["reasons"] == {}
    assert "all of them" in g["headline"]
    assert "All 2 delivered references were deep-read." in read_gap_html(g)
    assert read_gap_html({}) == "" and read_gap_md(None) == []


def test_a_shortfall_is_rendered_in_red_with_its_reasons():
    g = ev._read_gap([{"pub_num": f"US-{i}-A1"} for i in range(30)],
                     [{"source": "no_content", "no_content_reason": "PDF unreadable"}],
                     [{"source": "no_content", "no_content_reason": "PDF unreadable"}])
    html = read_gap_html(g)
    assert "#dc2626" in html                       # the red border/heading, not a footnote
    assert "Delivered 30 references, deep-read 0" in html
    assert "unchecked" in html and "PDF unreadable" in html
    md = "\n".join(read_gap_md(g))
    assert "Delivered 30 references, deep-read 0" in md and "PDF unreadable" in md
    assert "<b>" not in md                         # the html emphasis does not leak into markdown


def test_the_shortfall_reaches_the_report_above_everything_else():
    mix = [{"source": "abstract"}, {"source": "no_content"}]
    g = ev._read_gap([{"pub_num": "US-1-A1"}] * 40, mix, mix)
    html = inject_html('<div class="sec-t">Invention Summary</div><div>x</div></div>\n</div>',
                       None, None, mix, None, read_gap=g)
    assert "Reading Coverage" in html
    # above the evidence mix, which is the weaker statement of the same problem
    assert html.index("Reading Coverage") < html.index("Evidence Coverage")
    md = inject_md("## Evaluation Criteria", None, None, mix, None, read_gap=g)
    assert md.index("## Reading Coverage") < md.index("## Evaluation Criteria")


def test_bigquery_claims_beat_a_missing_pdf_and_no_content_says_why(monkeypatch):
    seen = {}

    async def fake_text(summary, checklist, text, title, match_type, doc_mode=""):
        seen["mode"], seen["text"] = doc_mode, text
        return {"title": title, "checklist_results": {"c": {"score": 2}},
                "source": "full_text" if doc_mode == "full_text" else "abstract"}

    monkeypatch.setattr("app.llm.evaluate_single_document_text", fake_text, raising=False)
    doc = {"pub_num": "US-9075557-B2", "match_type": "Patent", "title": "T",
           "abstract": "a" * 200, "claims_text": "1. A widget " + "w" * 300}
    out = asyncio.run(ev.eval_single_doc({"summary": "s", "checklist": [], "doc": doc,
                                          "source_pdf_path": None, "source_title": ""}))
    r = out["eval_results"][0]
    assert r["source"] == "full_text" and seen["mode"] == "full_text"
    assert "[CLAIMS]" in seen["text"] and r["text_chars"] == len(seen["text"])

    empty = {"pub_num": "EP-1608112-A1", "match_type": "Patent", "title": "T", "abstract": ""}
    out = asyncio.run(ev.eval_single_doc({"summary": "s", "checklist": [], "doc": empty,
                                          "source_pdf_path": None, "source_title": ""}))
    r = out["eval_results"][0]
    assert r["source"] == "no_content"
    assert "amie_patents" in r["no_content_reason"] and "no PDF" in r["no_content_reason"]


def test_hydrate_full_text_chunks_and_only_asks_for_what_is_missing(monkeypatch):
    calls = []

    async def fake_fetch(pubs, with_claims=True):
        calls.append(list(pubs))
        return {bq._canon_pub(p): {"title": "T", "abstract": "A", "claims_text": f"claims of {p}"}
                for p in pubs}

    monkeypatch.setattr(bq, "fetch_by_pub_nums", fake_fetch)
    docs = ([{"pub_num": f"US{9000000 + i}B2", "match_type": "Patent"} for i in range(700)]
            + [{"pub_num": "US9075557B2", "match_type": "Patent", "claims_text": "already here"}]
            + [{"pub_num": "10.1/abc", "match_type": "Paper", "title": "a paper"}])
    stats = asyncio.run(bq.hydrate_full_text(docs, chunk=300))

    assert stats["asked"] == 700 and stats["chunks"] == 3 and stats["with_claims"] == 700
    assert [len(c) for c in calls] == [300, 300, 100]      # never near the 30 GiB single-query ceiling
    assert all(len(c) <= 300 for c in calls)
    assert docs[700]["claims_text"] == "already here"      # a doc that had claims is not re-fetched
    assert not docs[701].get("claims_text")                # papers are not patents
    assert docs[0]["abstract"] == "A" and docs[0]["title"] == "T"


def test_hydrate_full_text_survives_a_failed_chunk(monkeypatch):
    async def boom(pubs, with_claims=True):
        raise RuntimeError("bytesBilledLimitExceeded")

    monkeypatch.setattr(bq, "fetch_by_pub_nums", boom)
    docs = [{"pub_num": "US9075557B2", "match_type": "Patent"}]
    stats = asyncio.run(bq.hydrate_full_text(docs))
    assert stats["with_claims"] == 0 and stats["errors"]
    assert "bytesBilledLimitExceeded" in stats["errors"][0]


def test_an_unreadable_pdf_is_not_counted_as_a_full_pdf_read(monkeypatch, tmp_path):
    """evaluate_single_document returns an empty checklist_results when the call
    raised or the reply would not parse. `source` was stamped "pdf" regardless,
    so those rows read as full-PDF evidence in the mix."""
    pdf = tmp_path / "prior_art_000.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    async def fake_pdf_eval(*a, **kw):
        return {"title": "T", "match_type": "Patent", "error": "500 Internal", "checklist_results": {}}

    monkeypatch.setattr("app.llm.evaluate_single_document", fake_pdf_eval, raising=False)
    monkeypatch.setattr("patent_analyzer.quote_verify.pdf_text", lambda p: "", raising=False)

    out = asyncio.run(ev.eval_single_doc({
        "summary": "s", "checklist": [], "source_pdf_path": None, "source_title": "",
        "doc": {"pub_num": "US-1-A1", "match_type": "Patent", "title": "T", "local_pdf": str(pdf)}}))
    r = out["eval_results"][0]
    assert r["source"] == "no_content"
    assert "returned nothing" in r["no_content_reason"] and "500 Internal" in r["no_content_reason"]

    g = ev._read_gap([{"pub_num": "US-1-A1"}], [r], [r])
    assert g["read"] == 0 and g["shortfall"] == 1


def test_reissue_design_and_plant_numbers_are_spelled_back_correctly(monkeypatch):
    """fetch_by_pub_nums carried local copies of _bq_form/_canon_pub whose number
    pattern was a bare \\d+, so US-RE41525-E and friends were passed through
    unchanged, hashed to the wrong bucket and never found. 125/300 hits became
    300/300 on publications drawn from amie_patents.claims itself."""
    asked = {}

    async def fake_fetch(pubs, with_claims=True):
        asked["pubs"] = list(pubs)
        return {}

    monkeypatch.setattr(bq, "fetch_by_pub_nums", fake_fetch)
    asyncio.run(bq.hydrate_full_text(
        [{"pub_num": p, "match_type": "Patent"} for p in
         ("US-RE41525-E", "US-D712191-S", "US-PP20104-P2", "US9075557B2", "US20120194631A1")]))
    assert [bq._bq_form(p) for p in asked["pubs"]] == [
        "US-RE41525-E", "US-D712191-S", "US-PP20104-P2", "US-9075557-B2", "US-2012194631-A1"]
