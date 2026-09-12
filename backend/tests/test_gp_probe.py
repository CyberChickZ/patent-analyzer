import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


def _client(monkeypatch, statuses):
    import app.diagnostics as d
    calls = []

    async def fake_one(client, i):
        calls.append(i)
        s = statuses[i % len(statuses)]
        return {"i": i, "status": s, "sorry": s == 429, "bytes": 10, "ms": 1}

    monkeypatch.setattr(d, "_one", fake_one)
    import app.main as m
    return TestClient(m.app), calls


def test_probe_reports_every_status_not_just_the_first(monkeypatch):
    c, calls = _client(monkeypatch, [200, 200, 503, 503])
    r = c.get("/admin/gp-probe?n=4&gap=0").json()
    assert calls == [0, 1, 2, 3]                    # no early exit on the first 503
    assert r["ok"] == 2 and r["blocked"] == 2 and r["block_rate"] == 0.5
    assert r["first_rate_limited_index"] == 2
    assert r["by_status"] == {"200": 2, "503": 2}


def test_probe_caps_n_and_gap(monkeypatch):
    c, calls = _client(monkeypatch, [200])
    r = c.get("/admin/gp-probe?n=9999&gap=0").json()
    assert r["n"] == 200 and len(calls) == 200


def test_probe_urls_are_distinct_so_the_cache_cannot_answer():
    import app.diagnostics as d
    assert len({d._probe_url(i) for i in range(50)}) == 50
