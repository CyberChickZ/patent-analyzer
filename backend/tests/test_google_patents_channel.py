import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import cache
from patent_analyzer.recall import google_patents as gp


@pytest.fixture(autouse=True)
def _kv(tmp_path):
    cache.reset_for_tests(tmp_path / "kv.sqlite")
    yield
    cache.reset_for_tests(None)


class _Resp:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def _payload(n, total):
    return {"results": {"total_num_results": total, "cluster": [{"result": [
        {"patent": {"publication_number": f"US{i}A1", "title": f"T{i}", "priority_date": "2010-01-01"}}
        for i in range(n)]}]}}


def test_search_caches_and_exposes_total(monkeypatch):
    calls = []

    async def fake_get(url, timeout=30):
        calls.append(url)
        return _Resp(_payload(3, 12345))
    monkeypatch.setattr(gp, "_get", fake_get)
    c1, e1 = asyncio.run(gp.search("gaze estimation video", num=10, before="priority:20120101"))
    c2, e2 = asyncio.run(gp.search("gaze  estimation video", num=10, before="priority:20120101"))
    assert e1 is None and e2 is None and len(c1) == 3 and len(c2) == 3
    assert len(calls) == 1  # second call served from cache (whitespace-normalised key)
    assert gp.last_total["gaze estimation video"] == 12345
    assert c1[0].raw["google_patents"]["total"] == 12345


def test_search_reports_block_via_shared_breaker(monkeypatch):
    async def fake_get(url, timeout=30):
        return None
    monkeypatch.setattr(gp, "_get", fake_get)
    gp._breaker.trip("test")
    assert gp.is_blocked()
    c, e = asyncio.run(gp.search("anything", num=10))
    assert c == [] and "blocked" in e
