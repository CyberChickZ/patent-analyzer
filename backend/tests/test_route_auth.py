"""Which routes answer without a token, and which refuse.

The split is a product decision, not an implementation detail: a report is
shared by sending somebody its URL, while the quota panel and a job's spend are
about the deployment. It was measured wrong before it was decided — an
unauthenticated /api/quota returned the whole quota JSON on the live service
(cloud_smoke.md §4) — so it gets a test that fails loudly if it drifts back.

AUTH_DISABLED is read per request, and test_e2e_full sets it at import time for
the whole process, so every test here sets it explicitly instead of trusting
whatever ran first.
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

PRIVATE = ["/api/quota", "/api/jobs/nosuchjob/usage", "/api/jobs/nosuchjob/funnel"]
PUBLIC_WITH_THE_LINK = ["/status/nosuchjob", "/report/nosuchjob", "/results/nosuchjob"]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    import app.main as m
    return TestClient(m.app)


@pytest.mark.parametrize("path", PRIVATE)
def test_deployment_wide_routes_need_a_token(client, path):
    r = client.get(path)
    assert r.status_code == 401, f"{path} answered {r.status_code} with no token"
    assert "token" in r.json()["detail"].lower()


@pytest.mark.parametrize("path", PUBLIC_WITH_THE_LINK)
def test_a_shared_job_url_still_answers_without_one(client, path):
    """404 because the job does not exist — the point is that it is not 401."""
    r = client.get(path)
    assert r.status_code == 404, f"{path} answered {r.status_code}; sharing a report URL is broken"


@pytest.mark.parametrize("path", PRIVATE[1:])
def test_auth_disabled_still_opens_them_for_local_development(monkeypatch, path):
    monkeypatch.setenv("AUTH_DISABLED", "1")
    import app.main as m
    r = TestClient(m.app).get(path)
    assert r.status_code == 404, f"{path} answered {r.status_code} under AUTH_DISABLED"


def test_auth_disabled_opens_the_quota_panel_too(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "1")
    import app.main as m
    from patent_analyzer import quota

    async def fake_snapshot():
        return {"sources": []}
    monkeypatch.setattr(quota, "snapshot", fake_snapshot)
    r = TestClient(m.app).get("/api/quota")
    assert r.status_code == 200 and r.json() == {"sources": []}


def test_the_switch_is_read_per_request_not_at_import():
    """If AUTH_DISABLED were captured at import, the two cases above could not
    both hold in one process — which is exactly how a stale switch hides."""
    import app.auth as a
    src = Path(a.__file__).read_text()
    assert 'os.getenv("AUTH_DISABLED"' in src.split("async def require_auth")[1]
