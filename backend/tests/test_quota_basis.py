import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import quota


@pytest.fixture(autouse=True)
def _local_spend(monkeypatch, tmp_path):
    """snapshot() reads the day's spend object; without this it reads the real
    one out of the shared bucket, which is a network call and shared state."""
    monkeypatch.setenv("SPEND_STORE", "local")
    monkeypatch.setenv("SPEND_LOCAL_DIR", str(tmp_path / "spend"))


def test_every_row_says_where_its_number_came_from(monkeypatch):
    """Harry asked whether the panel's figures had been checked against each
    provider. They had not — every one was this process's own tally, and a
    tally that has never been reconciled looks exactly like one that has
    (2026-09-20). So the field is mandatory, and its values are a closed set."""
    monkeypatch.setattr(quota, "_bq_billed_mib_this_month", lambda: (None, "no creds"))
    snap = asyncio.run(quota.snapshot())
    allowed = {quota.BASIS_ACCOUNT_API, quota.BASIS_RESPONSE_HEADER,
               quota.BASIS_INFORMATION_SCHEMA, quota.BASIS_OURS, quota.BASIS_NONE}
    assert snap["sources"], "no sources at all"
    for r in snap["sources"]:
        assert r["basis"] in allowed, (r["name"], r["basis"])
        if r["used"] is None and r["cap"] is None:
            assert r["basis"] == quota.BASIS_NONE, r["name"]


def test_bigquery_prefers_the_project_wide_figure_and_falls_back_loudly(monkeypatch):
    monkeypatch.setattr(quota, "_bq_billed_mib_this_month", lambda: (4242, ""))
    row = quota._bigquery()[0]
    assert row["used"] == 4242 and row["basis"] == quota.BASIS_INFORMATION_SCHEMA
    assert "whole project" in row["note"]

    monkeypatch.setattr(quota, "_bq_billed_mib_this_month", lambda: (None, "Forbidden: no access"))
    row = quota._bigquery()[0]
    assert row["basis"] == quota.BASIS_OURS
    assert "Forbidden: no access" in row["note"], "a fallback that does not say why is the bug again"


def test_serpapi_marks_the_rows_local_when_the_account_api_will_not_answer(monkeypatch):
    from patent_analyzer.recall import serpapi as sp
    monkeypatch.setenv("SERPAPI_KEYS", "k1,k2")
    monkeypatch.setattr(sp, "sync_account", lambda: [{"key": "aaaa", "error": "HTTP 401"}])

    class NoCache:
        def get(self, *a, **k):
            return None

        def put(self, *a, **k):
            pass
    monkeypatch.setattr(quota, "kv", lambda: NoCache(), raising=False)
    import patent_analyzer.cache as cache
    monkeypatch.setattr(cache, "kv", lambda: NoCache())

    rows = asyncio.run(quota._serpapi())
    assert rows and all(r["basis"] == quota.BASIS_OURS for r in rows)
    assert "HTTP 401" in rows[0]["note"]


def test_uspto_is_our_own_count_because_the_api_gives_nothing_to_check(monkeypatch):
    """Measured: a live call to api.uspto.gov on 2026-09-20 came back with no
    rate-limit or remaining header, only AWS API-Gateway trace headers. The
    count is ours, but it is in GCS and shared — never "local"."""
    rows = quota._uspto_odp()
    assert rows and all(r["basis"] == quota.BASIS_OURS for r in rows)
    assert "no rate-limit or remaining header" in rows[0]["note"]
