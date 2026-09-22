import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    import app.main as m
    monkeypatch.setattr(m, "_save_job", lambda j: None)
    monkeypatch.setattr(m, "_enqueue_job", lambda jid: None)
    monkeypatch.setattr(m, "OUTPUT_BASE", tmp_path)
    monkeypatch.setattr(m, "_get_gcs", lambda: _FakeGCS())
    # setitem, not plain assignment: dependency_overrides lives on the module-level
    # app object, so an override written here outlives this test and every later
    # test runs signed in. test_route_auth caught exactly that.
    monkeypatch.setitem(m.app.dependency_overrides, m.require_auth, lambda: {"email": "t@x"})
    return TestClient(m.app), m


class _FakeGCS:
    def bucket(self, name):
        return self

    def blob(self, key):
        return self

    def download_to_filename(self, path):
        Path(path).write_text("Core Idea\nx\n")


def test_analyze_form_input_mode_is_validated(monkeypatch, tmp_path):
    c, m = _client(monkeypatch, tmp_path)
    r = c.post("/analyze", files={"file": ("d.txt", b"Core Idea\nx\n")}, data={"input_mode": "disclosure"})
    assert r.status_code == 200 and m.jobs[r.json()["job_id"]]["input_mode"] == "disclosure"
    r = c.post("/analyze", files={"file": ("d.txt", b"x")}, data={"input_mode": "Manuscript "})
    assert m.jobs[r.json()["job_id"]]["input_mode"] == "manuscript"
    r = c.post("/analyze", files={"file": ("d.txt", b"x")}, data={"input_mode": "weird"})
    assert m.jobs[r.json()["job_id"]]["input_mode"] == ""
    r = c.post("/analyze", files={"file": ("d.txt", b"x")})
    assert m.jobs[r.json()["job_id"]]["input_mode"] == ""


def test_analyze_gcs_json_input_mode(monkeypatch, tmp_path):
    c, m = _client(monkeypatch, tmp_path)
    r = c.post("/analyze-gcs", json={"gcs_uri": "gs://b/k.txt", "filename": "k.txt", "input_mode": "patent_draft"})
    assert r.status_code == 200 and m.jobs[r.json()["job_id"]]["input_mode"] == "patent_draft"
    r = c.post("/analyze-gcs", json={"gcs_uri": "gs://b/k.txt", "filename": "k.txt", "input_mode": "auto"})
    assert m.jobs[r.json()["job_id"]]["input_mode"] == ""
