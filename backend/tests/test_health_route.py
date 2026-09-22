import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


def _client():
    import app.main as m
    return TestClient(m.app), m


def test_health_reports_gcs_failures():
    c, m = _client()
    m._gcs_failures.clear()
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["gcs"] == {}
    try:
        m._gcs_failed("upload report.html", RuntimeError("403 no bucket access"))
        body = c.get("/health").json()
        assert body["gcs"] == {"upload report.html": 1}
        assert "403 no bucket access" in body["gcs_last_error"]["upload report.html"]
    finally:
        m._gcs_failures.clear()


def test_healthz_is_gone_because_cloud_run_never_let_it_answer():
    c, _ = _client()
    assert c.get("/healthz").status_code == 404
