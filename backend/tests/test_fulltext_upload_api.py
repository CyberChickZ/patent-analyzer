"""The missing-full-text endpoints: the list, the upload slot, the re-run trigger."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tests.test_fulltext_gap import _results  # noqa: E402

PDF = b"%PDF-1.4\n% a tiny but genuine-looking header\n"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.main as m
    from patent_analyzer import fulltext as ft

    job = {"id": "jg", "status": "completed", "output_dir": str(tmp_path), "input_path": str(tmp_path / "src.pdf")}
    (tmp_path / "results.json").write_text(json.dumps(_results([
        {"pub_num": "10.1/a", "title": "A", "source": "abstract", "similarity_score": 0.5,
         "fulltext_tier": "abstract_only", "fulltext_detail": "Unpaywall: not open access"},
        {"pub_num": "10.1/b", "title": "B", "source": "pdf", "similarity_score": 0.9},
    ])))
    monkeypatch.setattr(m, "_save_job", lambda j: None)
    monkeypatch.setattr(m, "_enqueue_job", lambda jid: job.__setitem__("_enqueued", True))
    monkeypatch.setattr(ft, "cache_put", lambda doi, data: False)
    m.jobs["jg"] = job
    yield TestClient(m.app), job
    m.jobs.pop("jg", None)


def test_the_endpoint_lists_the_gap_and_carries_the_policy_note(client):
    c, _ = client
    r = c.get("/api/jobs/jg/fulltext-gaps").json()
    assert [x["ref_id"] for x in r["rows"]] == ["10.1-a"]
    assert r["summary"]["missing"] == 1 and r["summary"]["full_text"] == 1
    assert "suspend access for the entire OSU community" in r["policy_note"]


def test_upload_attaches_a_pdf_to_one_reference_and_leaves_it_pending(client):
    c, job = client
    r = c.post("/api/jobs/jg/fulltext/10.1-a/upload", files={"file": ("a.pdf", PDF, "application/pdf")})
    assert r.status_code == 200 and r.json()["pending_reread"] == 1
    up = job["fulltext_uploads"]["10.1-a"]
    assert up["reread"] is False and Path(up["path"]).read_bytes() == PDF
    assert c.get("/api/jobs/jg/fulltext-gaps").json()["rows"][0]["upload"]["filename"] == "a.pdf"


def test_a_file_that_is_not_a_pdf_is_refused_here_rather_than_inside_the_model_call(client):
    c, _ = client
    r = c.post("/api/jobs/jg/fulltext/10.1-a/upload", files={"file": ("a.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400 and "%PDF" in r.json()["detail"]


def test_uploading_against_a_reference_that_was_read_is_refused(client):
    c, _ = client
    r = c.post("/api/jobs/jg/fulltext/10.1-b/upload", files={"file": ("b.pdf", PDF, "application/pdf")})
    assert r.status_code == 404


def test_rerun_needs_something_uploaded_and_then_queues_the_job(client):
    c, job = client
    assert c.post("/api/jobs/jg/rerun-evidence", json={}).status_code == 400
    c.post("/api/jobs/jg/fulltext/10.1-a/upload", files={"file": ("a.pdf", PDF, "application/pdf")})
    r = c.post("/api/jobs/jg/rerun-evidence", json={})
    assert r.status_code == 200 and r.json()["refs"] == ["10.1-a"]
    assert job["_rerun_evidence"] == ["10.1-a"] and job["status"] == "queued" and job["_enqueued"]


def test_a_running_job_does_not_take_a_rerun(client):
    c, job = client
    c.post("/api/jobs/jg/fulltext/10.1-a/upload", files={"file": ("a.pdf", PDF, "application/pdf")})
    job["status"] = "running"
    assert c.post("/api/jobs/jg/rerun-evidence", json={}).status_code == 409


def test_a_staged_upload_can_be_dropped_but_a_read_one_cannot(client):
    c, job = client
    c.post("/api/jobs/jg/fulltext/10.1-a/upload", files={"file": ("a.pdf", PDF, "application/pdf")})
    job["fulltext_uploads"]["10.1-a"]["reread"] = True
    assert c.delete("/api/jobs/jg/fulltext/10.1-a").status_code == 409
    job["fulltext_uploads"]["10.1-a"]["reread"] = False
    assert c.delete("/api/jobs/jg/fulltext/10.1-a").status_code == 200
    assert job["fulltext_uploads"] == {}
