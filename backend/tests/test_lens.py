import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import cache
from patent_analyzer.recall import lens


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    cache.reset_for_tests(tmp_path / "kv.sqlite")
    monkeypatch.setenv("LENS_API_TOKEN", "test-token")
    lens.call_log.clear()
    yield
    cache.reset_for_tests(None)


class _Resp:
    def __init__(self, payload, status=200, headers=None):
        self._p, self.status_code, self.headers = payload, status, headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._p


class _Client:
    """Fake httpx.AsyncClient: records bodies, answers from a queue."""
    posts: list[tuple[str, dict, dict]] = []
    queue: list[_Resp] = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        _Client.posts.append((url, json, headers))
        return _Client.queue.pop(0) if _Client.queue else _Resp({"total": 0, "data": []})


@pytest.fixture
def client(monkeypatch):
    _Client.posts, _Client.queue = [], []
    monkeypatch.setattr(lens.httpx, "AsyncClient", _Client)
    return _Client


_SCHOLARLY = {"total": 2, "results": 2, "data": [
    {"lens_id": "036-136-701-555-455", "title": "Embodied social proxy",
     "external_ids": [{"type": "doi", "value": "10.1145/1753326.1753482"}, {"type": "magid", "value": "2171005050"}],
     "patent_citations": [{"lens_id": "166-918-267-737-124"}, {"lens_id": "174-042-583-643-177"}],
     "patent_citations_count": 2},
    {"lens_id": "051-020-197-268-076", "title": "Now, i have a body",
     "external_ids": [{"type": "magid", "value": "2000916439"}],
     "patent_citations": [{"lens_id": "095-817-746-680-78X"}], "patent_citations_count": 1},
]}

_PATENT = {"total": 1, "results": 1, "data": [
    {"lens_id": "166-918-267-737-124", "jurisdiction": "US", "doc_number": "8520052", "kind": "B2",
     "date_published": "2013-08-27", "publication_type": "GRANTED_PATENT",
     "biblio": {"invention_title": [{"text": "Functionality for indicating direction of attention", "lang": "en"}],
                "classifications_cpc": {"classifications": [{"symbol": "H04N7/15"}, {"symbol": "G06V40/176"}, {"symbol": "H04N7/15"}]}},
     "families": {"simple_family": {"members": [
         {"document_id": {"jurisdiction": "US", "doc_number": "8520052", "kind": "B2", "date": "2013-08-27"}, "lens_id": "166-918-267-737-124"},
         {"document_id": {"jurisdiction": "US", "doc_number": "20120194631", "kind": "A1", "date": "2012-08-02"}, "lens_id": "133-754-808-407-131"}],
         "size": 2}}},
]}


def test_scholarly_by_ids_query_shape_and_keys(client):
    client.queue.append(_Resp(_SCHOLARLY))
    out, err = asyncio.run(lens.scholarly_by_ids(["https://doi.org/10.1145/1753326.1753482"], ["W2000916439", "W2171005050"]))
    assert err is None
    url, body, headers = client.posts[0]
    assert url.endswith("/scholarly/search")
    assert headers["Authorization"] == "Bearer test-token"
    should = body["query"]["bool"]["should"]
    assert {"terms": {"ids.doi": ["10.1145/1753326.1753482"]}} in should
    assert {"terms": {"ids.openalex": ["W2000916439", "W2171005050"]}} in should
    assert body["size"] == lens.MAX_RECORDS["scholarly"] and "patent_citations" in body["include"]
    assert out["10.1145/1753326.1753482"]["patent_citations"] == ["166-918-267-737-124", "174-042-583-643-177"]
    assert out["W2171005050"]["lens_id"] == "036-136-701-555-455"      # matched through magid echo
    assert out["W2000916439"]["patent_citations_count"] == 1
    assert lens.call_log[-1]["endpoint"] == "scholarly" and lens.call_log[-1]["returned"] == 2
    assert "test-token" not in json.dumps(lens.call_log)


def test_scholarly_batches_by_max_records(client):
    ids = [f"W{i}" for i in range(lens.MAX_RECORDS["scholarly"] + 1)]
    client.queue += [_Resp({"total": 0, "data": []}), _Resp({"total": 0, "data": []})]
    asyncio.run(lens.scholarly_by_ids([], ids))
    assert len(client.posts) == 2
    assert len(client.posts[0][1]["query"]["terms"]["ids.openalex"]) == lens.MAX_RECORDS["scholarly"]


def test_patents_by_lens_ids_normalises(client):
    client.queue.append(_Resp(_PATENT))
    cands, err = asyncio.run(lens.patents_by_lens_ids(["166-918-267-737-124"]))
    assert err is None and len(cands) == 1
    c = cands[0]
    assert c.pub_num == "US8520052B2" and c.match_type == "Patent" and c.year == "2013"
    assert c.title == "Functionality for indicating direction of attention"
    assert c.sources == ["lens_bridge"]
    assert c.raw["lens"]["cpc"] == ["H04N7/15", "G06V40/176"]
    assert c.raw["lens"]["family"] == ["US20120194631A1", "US8520052B2"]
    assert c.raw["lens"]["family_key"] == "133-754-808-407-131"
    assert client.posts[0][1]["query"] == {"terms": {"lens_id": ["166-918-267-737-124"]}}


def test_search_patents_body_shape(client):
    client.queue.append(_Resp(_PATENT))
    cands, err = asyncio.run(lens.search_patents(["telepresence system", "turntable"], cpc="H04N7/15", before="20110202", size=50))
    assert err is None and cands[0].sources == ["lens_search"] and cands[0].raw["lens"]["total"] == 1
    body = client.posts[0][1]
    assert body["size"] == 50
    inner = body["query"]["bool"]["must"][0]["bool"]
    assert inner["minimum_should_match"] == 1
    assert {"match_phrase": {"title": "telepresence system"}} in inner["should"]
    assert {"match": {"claim": "turntable"}} in inner["should"]
    filt = body["query"]["bool"]["filter"]
    assert {"query_string": {"query": "class_cpc.symbol:H04N7\\/15*"}} in filt
    assert {"range": {"date_published": {"lt": "2011-02-02"}}} in filt


def test_cache_hit_skips_request(client):
    client.queue.append(_Resp(_PATENT))
    c1, _ = asyncio.run(lens.search_patents(["turntable"], cpc="A63F", before="20110202"))
    c2, _ = asyncio.run(lens.search_patents(["turntable"], cpc="A63F", before="20110202"))
    assert len(client.posts) == 1 and len(c1) == len(c2) == 1
    assert lens.call_log[-1]["cached"] is True and lens.call_log[-2]["cached"] is False


def test_rate_limit_and_missing_token(client, monkeypatch):
    client.queue.append(_Resp({"message": "too many"}, status=429,
                              headers={"x-rate-limit-remaining-request-per-minute": "0",
                                       "x-rate-limit-retry-after-seconds": "42"}))
    cands, err = asyncio.run(lens.search_patents(["turntable"]))
    assert cands == [] and "429" in err and "retry-after=42s" in err
    assert lens.call_log[-1]["http_status"] == 429 and "err" in lens.call_log[-1]
    monkeypatch.delenv("LENS_API_TOKEN")
    cands, err = asyncio.run(lens.search_patents(["other"]))
    assert cands == [] and "LENS_API_TOKEN" in err and len(client.posts) == 1
