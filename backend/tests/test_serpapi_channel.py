import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import cache
from patent_analyzer.recall import serpapi as sp


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    cache.reset_for_tests(tmp_path / "kv.sqlite")
    monkeypatch.setenv("SERPAPI_KEYS", "keyA,keyB")
    monkeypatch.setattr(sp, "FREE_TIER_PER_KEY", 2)
    yield
    cache.reset_for_tests(None)


def _ok(n, total=777):
    return [{"title": f"T{i}", "pub_num": f"US{i}B2", "match_type": "Patent", "total": total} for i in range(n)], None


def test_rotates_to_next_key_on_exhaustion(monkeypatch):
    used = []

    def fake(engine, query, key, log, pages, num, extra=None):
        used.append(key)
        return ([], "HTTP 401 (invalid SerpAPI key or quota exhausted)") if key == "keyA" else _ok(2)
    monkeypatch.setattr(sp, "_sync_search", fake)
    c, e = asyncio.run(sp.search_patents("q1"))
    assert e is None and len(c) == 2 and used == ["keyA", "keyB"]
    assert sp.last_total["q1"] == 777
    # keyA is marked spent; next query goes straight to keyB
    used.clear()
    c, e = asyncio.run(sp.search_patents("q2"))
    assert used == ["keyB"] and e is None


def test_per_key_monthly_cap(monkeypatch):
    monkeypatch.setattr(sp, "_sync_search", lambda *a: _ok(1))
    for q in ("a", "b", "c", "d"):
        assert asyncio.run(sp.search_patents(q))[1] is None
    c, e = asyncio.run(sp.search_patents("e"))
    assert c == [] and "exhausted" in e
    assert [s["used"] for s in sp.quota_status()] == [2, 2]


def test_cache_costs_no_credit(monkeypatch):
    calls = []
    monkeypatch.setattr(sp, "_sync_search", lambda *a: (calls.append(1), _ok(1))[1])
    asyncio.run(sp.search_patents("same  query"))
    asyncio.run(sp.search_patents("same query"))
    assert len(calls) == 1 and sp.quota_status()[0]["used"] == 1


def test_non_quota_error_releases_credit(monkeypatch):
    monkeypatch.setattr(sp, "_sync_search", lambda *a: ([], "network failure after 4 retries"))
    c, e = asyncio.run(sp.search_patents("x"))
    assert c == [] and "network" in e and sp.quota_status()[0]["used"] == 0


def test_before_is_forwarded_and_keyed_separately(monkeypatch):
    seen = []
    monkeypatch.setattr(sp, "_sync_search", lambda *a: (seen.append(a[6]), _ok(1))[1])
    asyncio.run(sp.search_patents("q", before="priority:20110202"))
    asyncio.run(sp.search_patents("q"))
    asyncio.run(sp.search_patents("q", before="priority:20110202"))
    assert seen == [{"before": "priority:20110202"}, None]


def test_searcher_puts_before_in_url(monkeypatch):
    from patent_analyzer import searcher
    urls = []

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"organic_results": []}'
    monkeypatch.setattr(searcher.urllib.request, "urlopen", lambda req, timeout=0, context=None: (urls.append(req.full_url), _Resp())[1])
    searcher.serpapi_search("google_patents", "q", "k", extra={"before": "priority:20110202"})
    assert "before=priority%3A20110202" in urls[0]


def test_no_results_answer_is_billed_and_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(sp, "_sync_search", lambda *a: (calls.append(1), ([], "SerpAPI error: Google Patents hasn't returned any results for this query."))[1])
    c, e = asyncio.run(sp.search_patents("nothing"))
    assert c == [] and e is None and sp.last_total["nothing"] == 0
    asyncio.run(sp.search_patents("nothing"))
    assert len(calls) == 1 and sp.quota_status()[0]["used"] == 1


def test_sync_account_overwrites_counter(monkeypatch):
    import io, json, urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=0: io.BytesIO(json.dumps({"this_month_usage": 129, "plan_searches_left": 121}).encode()))
    out = sp.sync_account()
    assert [o["used"] for o in out] == [129, 129]
    assert [s["used"] for s in sp.quota_status()] == [129, 129]


def test_scholar_flag_mixes_paper_results(monkeypatch):
    seen = []

    def fake(engine, query, key, log, pages, num, extra=None):
        seen.append(extra)
        return [{"title": "P", "pub_num": "US1B2", "match_type": "Patent", "total": 5},
                {"title": "A paper", "pub_num": "", "match_type": "Paper", "url": "https://x", "total": 5}], None
    monkeypatch.setattr(sp, "_sync_search", fake)
    c, e = asyncio.run(sp.search_patents("q", before="priority:20110101", scholar=True))
    assert seen == [{"before": "priority:20110101", "scholar": "true"}]
    assert [x.match_type for x in c] == ["Patent", "Paper"]


def test_searcher_maps_scholar_items_to_papers():
    from patent_analyzer import searcher
    import io, json, urllib.request
    payload = {"search_information": {"total_results": 2}, "organic_results": [
        {"title": "Pat", "publication_number": "US1B2", "patent_link": "https://p"},
        {"title": "Paper", "is_scholar": True, "scholar_link": "https://s", "publication_date": "2009-01-01"}]}
    real = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=0, context=None: io.BytesIO(json.dumps(payload).encode())
    try:
        res, err = searcher.serpapi_search("google_patents", "q", "k", extra={"scholar": "true"})
    finally:
        urllib.request.urlopen = real
    assert err is None and [r["match_type"] for r in res] == ["Patent", "Paper"]
    assert res[1]["url"] == "https://s" and res[1]["year"] == "2009"
