"""The quota panel: one shape for every source, and no invented numbers."""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import cache, quota
from patent_analyzer.recall import serpapi as sp


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    cache.reset_for_tests(tmp_path / "kv.sqlite")
    monkeypatch.setenv("SERPAPI_KEYS", "keyA,keyB")
    monkeypatch.delenv("LENS_API_TOKEN", raising=False)
    monkeypatch.delenv("USPTO_ODP_API_KEY", raising=False)
    monkeypatch.setattr(sp, "FREE_TIER_PER_KEY", 2)
    yield
    cache.reset_for_tests(None)


def _by_name(snap, name):
    return next(s for s in snap["sources"] if s["name"] == name)


@pytest.fixture(autouse=True)
def _no_live_providers(monkeypatch, tmp_path):
    """`snapshot()` now asks two providers for their own figures: SerpAPI's
    /account and BigQuery's INFORMATION_SCHEMA. Both are network calls, and on
    a developer machine with credentials they quietly succeed — so the suite
    would pass here and report different numbers anywhere else, while the
    SerpAPI probe spent 15 s per fake key timing out. Tests read the local
    tally; the reconciled paths have their own tests with the call stubbed."""
    monkeypatch.setattr(quota, "_bq_billed_mib_this_month", lambda: (None, "not asked in tests"))
    monkeypatch.setattr(sp, "sync_account", lambda: [{"key": "test", "error": "not asked in tests"}])
    monkeypatch.setenv("SPEND_STORE", "local")
    monkeypatch.setenv("SPEND_LOCAL_DIR", str(tmp_path / "spend"))


def test_a_month_resets_on_the_first_and_an_iso_week_on_monday():
    wed = datetime(2026, 9, 16, 13, 0, tzinfo=timezone.utc)
    assert quota._next_month(wed) == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert quota._next_week(wed) == datetime(2026, 9, 21, tzinfo=timezone.utc)   # the Monday
    dec = datetime(2026, 12, 31, 23, 0, tzinfo=timezone.utc)
    assert quota._next_month(dec) == datetime(2027, 1, 1, tzinfo=timezone.utc)


def test_a_spent_key_is_exhausted_and_named_at_the_top():
    snap = asyncio.run(quota.snapshot())
    keys = [s for s in snap["sources"] if s["source"] == "serpapi"]
    assert len(keys) == 2 and all(s["remaining"] == 2 for s in keys)

    # spend one key right out
    sp._quota("keyA").exhaust()
    snap = asyncio.run(quota.snapshot())
    spent = [s for s in snap["sources"] if s["source"] == "serpapi" and s["exhausted"]]
    assert len(spent) == 1 and spent[0]["remaining"] == 0
    assert spent[0]["name"] in snap["exhausted"]
    assert spent[0]["resets_at"].startswith(("2026-10", "2026-11", "2026-12", "202"))


def test_a_source_with_no_counter_reports_its_rate_and_no_number():
    row = _by_name(asyncio.run(quota.snapshot()), "Semantic Scholar")
    assert row["used"] is None and row["cap"] is None and row["remaining"] is None
    assert row["exhausted"] is False and row["limits"]
    assert row["period"] == "none"


def test_a_dated_trial_counts_down_in_days_and_expires():
    future = quota._row("lens", "Lens", used=0, cap=1000, expires_on="2999-01-01")
    assert future["expires_in_days"] > 300_000 and future["exhausted"] is False
    past = quota._row("lens", "Lens", used=0, cap=1000, expires_on="2020-01-01")
    assert past["expires_in_days"] < 0 and past["exhausted"] is True, "an expired plan is gone"
    assert quota._row("lens", "Lens", expires_on="not-a-date")["error"]


def test_lens_falls_back_to_the_local_count_when_the_provider_will_not_say():
    rows = [s for s in asyncio.run(quota.snapshot())["sources"] if s["source"] == "lens"]
    assert len(rows) == 2
    for r in rows:
        assert "LENS_API_TOKEN not set" in r["error"]
        assert r["cap"] == 1000 and r["used"] == 0          # the local counter, not a guess
        assert r["expires_in_days"] is not None


def test_one_broken_source_does_not_take_the_panel_down(monkeypatch):
    def boom():
        raise RuntimeError("provider on fire")
    monkeypatch.setattr(quota, "_serpapi", boom)
    snap = asyncio.run(quota.snapshot())
    assert "provider on fire" in _by_name(snap, "serpapi")["error"]
    assert _by_name(snap, "Semantic Scholar")["limits"], "the other sources still report"


def test_unknown_usage_shapes_are_parsed_or_left_alone():
    payload = {"subscription": {"plan": "trial",
                               "limits": [{"resource": "REQUEST", "period": "MONTH", "limit": 1000, "used": 137},
                                          {"resource": "RECORD", "period": "MONTH", "limit": 100000, "remaining": 90000}]}}
    found = {u["resource"]: u for u in quota._walk_usage(payload)}
    assert found["REQUEST"]["used"] == 137 and found["REQUEST"]["cap"] == 1000
    assert found["RECORD"]["used"] == 10000                 # derived from remaining
    assert quota._walk_usage({"something": "else"}) == []


def test_the_monthly_row_wins_over_the_per_minute_one():
    """What Lens actually returns, verbatim (2026-09-18). The per-minute entry
    sits in the same flat list and reads 10/10; taking it would report a
    thousand-request allowance as ten."""
    live = [{"remaining": 676, "allowed": 1000, "frequency": "1 MONTH", "type": "REQUEST",
             "resetDate": "2026-10-18T17:53:06.903Z", "maxRecordsPerRequest": 100},
            {"remaining": 71493, "allowed": 100000, "frequency": "1 MONTH", "type": "RECORD",
             "resetDate": "2026-10-18T17:53:06.903Z", "maxRecordsPerRequest": 100},
            {"remaining": 10, "allowed": 10, "frequency": "1 MINUTE", "type": "REQUEST",
             "maxRecordsPerRequest": 100}]
    monthly = [u for u in quota._walk_usage(live)
               if "month" in u["period"] and u["resource"].lower().startswith("request")]
    assert monthly and monthly[0]["cap"] == 1000 and monthly[0]["used"] == 324
    assert monthly[0]["resets_at"].startswith("2026-10-18")


def test_lens_takes_the_providers_figures_and_its_anniversary_reset(monkeypatch):
    from patent_analyzer.recall import lens

    async def fake(endpoint):
        return ([{"remaining": 676, "allowed": 1000, "frequency": "1 MONTH", "type": "REQUEST",
                  "resetDate": "2026-10-18T17:53:06.903Z"}], None)
    monkeypatch.setattr(lens, "live_usage", fake)
    rows = [s for s in asyncio.run(quota.snapshot())["sources"] if s["source"] == "lens"]
    assert all(r["cap"] == 1000 and r["used"] == 324 for r in rows)
    assert all(r["resets_at"].startswith("2026-10-18") for r in rows), \
        "Lens counts a month from the subscription date, not the 1st"
    assert all("from the provider" in r["note"] for r in rows)


def test_bigquery_reports_what_is_left_of_the_free_tibibyte():
    from patent_analyzer import metering
    metering.count_bq(64 * 2 ** 20)                          # 64 MiB
    row = _by_name(asyncio.run(quota.snapshot()), "BigQuery free tier")
    assert row["cap"] == quota.BQ_FREE_MIB_PER_MONTH
    # the local tally, because the fixture denies it INFORMATION_SCHEMA
    assert row["used"] == 64 and row["remaining"] == quota.BQ_FREE_MIB_PER_MONTH - 64
    assert row["unit"] == "MiB scanned" and row["basis"] == quota.BASIS_LOCAL
