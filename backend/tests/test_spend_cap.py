import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import spend


@pytest.fixture
def local(monkeypatch, tmp_path):
    """Never the shared bucket. A test that writes a fake amount into the day
    object makes the production panel lie — which happened once, on the day
    this was written, and had to be deleted from the bucket by hand."""
    monkeypatch.setenv("SPEND_STORE", "local")
    monkeypatch.setenv("SPEND_LOCAL_DIR", str(tmp_path))
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", "30")
    return tmp_path


def test_a_job_reports_its_total_and_only_the_difference_is_added(local):
    assert spend.record_job_total("j1", 5.0)["usd"] == 5.0
    assert spend.record_job_total("j1", 7.5)["usd"] == 7.5, "the second call is a total, not a delta"
    assert spend.record_job_total("j1", 7.5)["usd"] == 7.5, "reporting the same phase twice is free"
    assert spend.record_job_total("j2", 2.5)["usd"] == 10.0
    assert spend.today()["by_job"] == {"j1": 7.5, "j2": 2.5}


def test_under_the_cap_there_is_no_refusal(local):
    spend.record_job_total("j1", 29.99)
    assert spend.refusal() is None


def test_the_refusal_says_how_much_what_the_cap_is_and_when_it_lifts(local):
    spend.record_job_total("j1", 30.0)
    r = spend.refusal()
    assert r is not None
    assert r["spent_usd"] == 30.0 and r["cap_usd"] == 30.0
    for must in ("$30.00", "daily ceiling", "already running are left alone",
                 "DAILY_SPEND_CAP_USD", "not a bill"):
        assert must in r["detail"], must
    assert r["resets_at"].endswith("+00:00") and 0 <= r["resets_in_hours"] <= 24


def test_the_cap_is_read_per_call_so_an_eval_run_can_raise_it(local, monkeypatch):
    spend.record_job_total("j1", 40.0)
    assert spend.refusal() is not None
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", "100")
    assert spend.refusal() is None


def test_a_new_job_is_refused_with_429_and_a_running_one_is_not_touched(local, monkeypatch):
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.setenv("AUTH_DISABLED", "1")
    import app.main as m
    monkeypatch.delitem(m.app.dependency_overrides, m.require_auth, raising=False)
    c = TestClient(m.app)

    spend.record_job_total("earlier", 31.0)
    r = c.post("/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 429, r.status_code
    body = r.json()["detail"]
    assert "$31.00" in body and "daily ceiling" in body
    assert r.headers.get("Retry-After", "").isdigit()
    # the gate is on submission only: a job already running is untouched
    m.jobs["running1"] = {"id": "running1", "status": "running", "phase": "phase3",
                          "filename": "f.pdf", "submitted_by": "dev@oregonstate.edu"}
    try:
        assert c.post("/analyze", files={"file": ("y.txt", b"hi", "text/plain")}).status_code == 429
        assert m.jobs["running1"]["status"] == "running"
    finally:
        m.jobs.pop("running1", None)
