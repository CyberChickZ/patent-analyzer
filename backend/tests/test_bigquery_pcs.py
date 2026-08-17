import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.recall import bigquery_patents as bq


class _FakeJob:
    def __init__(self, rows, dry):
        self._rows, self.total_bytes_processed = rows, 0.5 * 2 ** 30 if dry else 0
    def result(self):
        return iter(self._rows)


class _FakeClient:
    """Answers the bucket query with [7]*n, records the lookup SQL/params and
    returns canned rows for the point query."""
    def __init__(self, rows):
        self.rows, self.calls = rows, []
    def query(self, sql, job_config=None):
        params = {p.name: p.values for p in (job_config.query_parameters or [])} if job_config else {}
        if "FARM_FINGERPRINT" in sql and "amie_patents" not in sql:
            return _FakeJob([SimpleNamespace(b=[7] * len(params["keys"]))], dry=False)
        self.calls.append((sql, params, bool(job_config and job_config.dry_run)))
        return _FakeJob(self.rows, dry=job_config.dry_run)


def _install(monkeypatch, rows):
    from google.cloud import bigquery
    fake = _FakeClient(rows)
    monkeypatch.setattr(bigquery, "Client", lambda project=None: fake)
    return fake


def test_canon_oa_id():
    assert bq._canon_oa_id("https://openalex.org/W2936032879") == "W2936032879"
    assert bq._canon_oa_id("w2936032879") == "W2936032879"
    assert bq._canon_oa_id("2936032879") == "W2936032879"
    assert bq._canon_oa_id("") == ""


def test_fetch_citing_patents_bucketed_lookup(monkeypatch):
    fake = _install(monkeypatch, [
        SimpleNamespace(oa_id="W3066", patent_pub="US-10494607-B2", reftype="app", confscore=10,
                        wherefound="frontonly", grant_year=2019, family_id="f1"),
        SimpleNamespace(oa_id="W3066", patent_pub="US-7311905-B2", reftype="exm", confscore=8,
                        wherefound="both", grant_year=None, family_id=None),
    ])
    out = asyncio.run(bq.fetch_citing_patents(["https://openalex.org/W3066", "3066", "W99"]))
    assert list(out) == ["W3066"]
    assert out["W3066"][0] == {"patent_pub": "US10494607B2", "reftype": "app", "confscore": 10,
                               "wherefound": "frontonly", "grant_year": 2019, "family_id": "f1"}
    assert out["W3066"][1]["patent_pub"] == "US7311905B2" and out["W3066"][1]["grant_year"] is None
    sql, params, _ = fake.calls[0]
    assert "amie_patents.pcs_oa`" in sql and "bucket IN UNNEST(@buckets)" in sql and "oa_id IN UNNEST(@keys)" in sql
    assert params["keys"] == ["W3066", "W99"] and params["buckets"] == [7, 7]
    assert [c[2] for c in fake.calls] == [True, False]


def test_fetch_citing_patents_empty(monkeypatch):
    fake = _install(monkeypatch, [])
    assert asyncio.run(bq.fetch_citing_patents(["", None])) == {}
    assert fake.calls == []


def test_fetch_cited_papers_bucketed_lookup(monkeypatch):
    fake = _install(monkeypatch, [
        SimpleNamespace(patent_pub="US-10494607-B2", oa_id="W3066", reftype="app", confscore=10, wherefound="frontonly"),
        SimpleNamespace(patent_pub="US-10494607-B2", oa_id="W12687", reftype="exm", confscore=7, wherefound="both"),
        SimpleNamespace(patent_pub="US-RE39456-E", oa_id="W5", reftype="unk", confscore=4, wherefound="bodyonly"),
    ])
    out = asyncio.run(bq.fetch_cited_papers(["US10494607B2", "us-10,494,607-b2", "US-RE39456-E", "EP1285093A2"]))
    assert set(out) == {"US10494607B2", "USRE39456E"}
    assert [e["oa_id"] for e in out["US10494607B2"]] == ["W3066", "W12687"]
    assert out["USRE39456E"] == [{"oa_id": "W5", "reftype": "unk", "confscore": 4, "wherefound": "bodyonly"}]
    sql, params, _ = fake.calls[0]
    assert "amie_patents.pcs_oa_by_patent`" in sql and "patent_pub IN UNNEST(@keys)" in sql
    assert params["keys"] == ["EP-1285093-A2", "US-10494607-B2", "US-RE39456-E"]


def test_bigquery_module_exports_pcs_lookups():
    for name in ("fetch_citing_patents", "fetch_cited_papers"):
        assert callable(getattr(bq, name)), name


def test_bq_form_reissue_design_plant():
    assert bq._bq_form("USRE39456E") == "US-RE39456-E"
    assert bq._bq_form("USD612345S") == "US-D612345-S"
    assert bq._bq_form("USPP12345P2") == "US-PP12345-P2"
    assert bq._bq_form("US10494607B2") == "US-10494607-B2"
