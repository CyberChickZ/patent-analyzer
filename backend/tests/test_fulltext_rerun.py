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


def _hit(crit, score=2):
    # `verified_quotes` is what adjudicate.element_covered counts; a criterion
    # with a quote nobody could find in the document is not coverage.
    return {crit: {"score": score, "match": score >= 2, "analysis": "…",
                   "evidence_quotes": ["verbatim"], "verified_quotes": ["verbatim"]}}


def _results(tmp_path):
    return {
        "job_id": "jr",
        "source_title": "Source",
        "phase1": {"summary": "S", "status_determination": "Present", "input_mode": "academic_paper"},
        "phase2": {"checklist": CHECKLIST},
        "search": {"summary": {"total_papers": 2, "fulltext_oa": {"manifest": []}}},
        "extraction": {"candidate_inventions": [{"id": "inv1", "concept": "c", "elements": []}]},
        "eval_stats": {"read_gap": {"delivered": 9}},
        "adjudication": {"label": "allow", "label_text": "No blocking reference", "n_elements": 2},
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


def _stub_read(monkeypatch, covered):
    """The deep read of the uploaded PDF, without Vertex."""
    import app.llm as llm

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
    assert hist["determination_before"] == "No blocking reference"
    assert hist["determination_after"] != hist["determination_before"]


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

    from app.fulltext_rerun import rerun_evidence

    async def empty(summary, checklist, pdf, title, match_type, **kw):
        return {"title": title, "match_type": match_type, "checklist_results": {}, "error": "unparseable"}

    monkeypatch.setattr(llm, "evaluate_single_document", empty)
    asyncio.run(rerun_evidence(job))

    hist = job["fulltext_reruns"][-1]
    assert hist["failed"] == ["10.1-paywalled"] and hist["read"] == []
    assert any("produced no" in e["message"] for e in job["events"])


def test_a_job_with_no_results_json_fails_loudly(tmp_path, monkeypatch):
    import app.main as m

    from app.fulltext_rerun import rerun_evidence

    monkeypatch.setattr(m, "_save_job", lambda x: None)
    j = {"id": "jr2", "output_dir": str(tmp_path), "_rerun_evidence": ["x"], "fulltext_uploads": {}}
    asyncio.run(rerun_evidence(j))
    assert j["status"] == "error" and "results.json" in j["error"]
