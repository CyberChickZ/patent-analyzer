"""The evidence re-run: only the uploaded references are read again, and the
determination is recomputed rather than carried over."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

PDF = b"%PDF-1.4\n% a tiny but genuine-looking header\n"

CHECKLIST = [{"id": "c1", "criterion": "A distributed controller", "weight": 0.5},
             {"id": "c2", "criterion": "A phase-synchronised clock", "weight": 0.5}]


# A quote long enough for quote_verify to locate, and present verbatim in the
# stubbed PDF text below. `adjudicate.element_covered` counts an element only
# when a quote was found in the document, so a made-up fragment would make the
# determination depend on the quote matcher's threshold rather than on the
# evidence these tests are about.
QUOTE = "The controller is distributed and the clock is phase-synchronised."


def _hit(crit, score=2):
    return {crit: {"score": score, "match": score >= 2, "analysis": "…",
                   "evidence_quotes": [QUOTE], "verified_quotes": [QUOTE]}}


def _results(tmp_path):
    return {
        "job_id": "jr",
        "source_title": "Source",
        "phase1": {"summary": "S", "status_determination": "Present", "input_mode": "academic_paper"},
        "phase2": {"checklist": CHECKLIST},
        "search": {"summary": {"total_papers": 2, "fulltext_oa": {"manifest": []}}},
        "extraction": {"candidate_inventions": [{"id": "inv1", "concept": "c", "elements": []}]},
        "eval_stats": {"read_gap": {"delivered": 9}},
        "adjudication": {"label": "ALLOW", "basis": "none",
                         "label_text": "No blocking reference", "n_elements": 2},
        "draft_claims": {},
        "evaluation": {"scoring_report": [
            {"pub_num": "10.1/paywalled", "title": "Paywalled paper", "match_type": "Paper",
             "source": "abstract", "similarity_score": 0.3, "checklist_results": _hit(CHECKLIST[0]["criterion"], 1),
             "fulltext_tier": "abstract_only", "fulltext_detail": "Unpaywall: not open access"},
            {"pub_num": "10.1/open", "title": "Open paper", "match_type": "Paper",
             "source": "pdf", "similarity_score": 0.5, "checklist_results": _hit(CHECKLIST[0]["criterion"], 2)},
        ]},
    }


@pytest.fixture()
def job(tmp_path, monkeypatch):
    import app.main as m
    from patent_analyzer import fulltext as ft

    (tmp_path / "results.json").write_text(json.dumps(_results(tmp_path)))
    (tmp_path / "uploaded_10.1-paywalled.pdf").write_bytes(PDF)
    j = {"id": "jr", "status": "queued", "output_dir": str(tmp_path), "input_path": str(tmp_path / "src.md"),
         "fulltext_uploads": {"10.1-paywalled": {
             "ref_id": "10.1-paywalled", "filename": "p.pdf", "bytes": len(PDF),
             "path": str(tmp_path / "uploaded_10.1-paywalled.pdf"), "doi": "10.1/paywalled",
             "gcs_uri": "", "uploaded_at": "2026-09-18T00:00:00Z", "reread": False}},
         "_rerun_evidence": ["10.1-paywalled"]}
    monkeypatch.setattr(m, "_save_job", lambda x: None)
    monkeypatch.setattr(ft, "cache_put", lambda doi, data: False)
    return j


# What quote_verify reads out of the uploaded PDF. Pinned rather than left to
# PyMuPDF on a 40-byte fake: `adjudicate` only counts an element covered when a
# quote was located in the document, so whether these tests see a §102 or an
# ALLOW would otherwise depend on what the extractor happens to return for a
# file that is not really a PDF.
PDF_TEXT = f"Introduction. {QUOTE} The rest of the paper follows."


def _stub_read(monkeypatch, covered):
    """The deep read of the uploaded PDF, without Vertex."""
    import app.llm as llm
    import patent_analyzer.quote_verify as qv

    monkeypatch.setattr(qv, "pdf_text", lambda path: PDF_TEXT)

    async def fake(summary, checklist, pdf, title, match_type, **kw):
        assert Path(pdf).read_bytes() == PDF        # the uploaded file, not a download
        cr = {}
        for c in checklist:
            cr.update(_hit(c["criterion"], 2 if c["criterion"] in covered else 0))
        return {"title": title, "match_type": match_type, "checklist_results": cr,
                "anticipation_assessment": "…", "key_teachings": "…"}

    monkeypatch.setattr(llm, "evaluate_single_document", fake)
    calls: list[str] = []
    orig = llm.evaluate_single_document_text

    async def counted(*a, **kw):
        calls.append("text")
        return await orig(*a, **kw)

    monkeypatch.setattr(llm, "evaluate_single_document_text", counted)
    return calls


def test_only_the_uploaded_reference_is_read_again(job, monkeypatch):
    from app.fulltext_rerun import rerun_evidence

    text_calls = _stub_read(monkeypatch, {CHECKLIST[0]["criterion"]})
    asyncio.run(rerun_evidence(job))

    out = json.loads((Path(job["output_dir"]) / "results.json").read_text())
    rows = {r["pub_num"]: r for r in out["evaluation"]["scoring_report"]}
    assert rows["10.1/paywalled"]["source"] == "pdf"          # re-read from the upload
    assert rows["10.1/open"]["source"] == "pdf"               # untouched
    assert rows["10.1/open"]["checklist_results"] == _hit(CHECKLIST[0]["criterion"], 2)
    assert text_calls == []                                   # nothing else went to a model
    assert job["fulltext_uploads"]["10.1-paywalled"]["reread"] is True
    assert job["status"] == "completed"


def test_the_determination_is_recomputed_from_the_new_evidence(job, monkeypatch):
    from app.fulltext_rerun import rerun_evidence

    # The upload covers both elements, which is a single anticipating reference.
    _stub_read(monkeypatch, {c["criterion"] for c in CHECKLIST})
    asyncio.run(rerun_evidence(job))

    out = json.loads((Path(job["output_dir"]) / "results.json").read_text())
    assert out["adjudication"]["label"] == "102"
    hist = job["fulltext_reruns"][-1]
    assert hist["read"] == ["10.1-paywalled"] and hist["failed"] == []
    assert hist["label_before"] == "ALLOW" and hist["label_after"] == "102" and hist["changed"] is True
    assert hist["determination_before"] == "No blocking reference"


def test_the_delivered_count_survives_the_rerun(job, monkeypatch):
    """`read_gap` is rebuilt by reduce_eval, which counts what Phase 3 delivered.
    That pool is not in this process, so the original number has to be carried
    across or the report would silently claim the run delivered 2 references."""
    from app.fulltext_rerun import rerun_evidence

    _stub_read(monkeypatch, {CHECKLIST[0]["criterion"]})
    asyncio.run(rerun_evidence(job))

    out = json.loads((Path(job["output_dir"]) / "results.json").read_text())
    assert out["eval_stats"]["read_gap"]["delivered"] == 9
    assert out["eval_stats"]["read_gap"]["read"] == 2


def test_an_upload_that_reads_as_nothing_is_reported_as_failed(job, monkeypatch):
    import app.llm as llm
    import patent_analyzer.quote_verify as qv

    from app.fulltext_rerun import rerun_evidence

    async def empty(summary, checklist, pdf, title, match_type, **kw):
        return {"title": title, "match_type": match_type, "checklist_results": {}, "error": "unparseable"}

    monkeypatch.setattr(llm, "evaluate_single_document", empty)
    monkeypatch.setattr(qv, "pdf_text", lambda path: PDF_TEXT)
    asyncio.run(rerun_evidence(job))

    hist = job["fulltext_reruns"][-1]
    assert hist["failed"] == ["10.1-paywalled"] and hist["read"] == [] and hist["changed"] is False
    assert any("produced no" in e["message"] for e in job["events"])


def test_a_job_with_no_results_json_fails_loudly(tmp_path, monkeypatch):
    import app.main as m

    from app.fulltext_rerun import rerun_evidence

    monkeypatch.setattr(m, "_save_job", lambda x: None)
    j = {"id": "jr2", "output_dir": str(tmp_path), "_rerun_evidence": ["x"], "fulltext_uploads": {}}
    asyncio.run(rerun_evidence(j))
    assert j["status"] == "error" and "results.json" in j["error"]
