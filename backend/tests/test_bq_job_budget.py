import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import metering
from patent_analyzer.recall import bigquery_patents as bq


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Per-job counters, so each test starts at zero without touching the
    module's globals for the rest of the suite."""
    monkeypatch.setattr(metering, "bq", {"queries": 0, "bytes_billed": 0.0})
    monkeypatch.setattr(metering, "incidents", [])
    monkeypatch.setattr(metering, "count_bq", _count)


def _count(bytes_billed):
    metering.bq["queries"] += 1
    metering.bq["bytes_billed"] += float(bytes_billed or 0)


def test_the_budget_is_read_per_call(monkeypatch):
    assert bq.job_budget() == (200.0, 200)
    monkeypatch.setenv("BQ_MAX_GIB_PER_JOB", "12.5")
    monkeypatch.setenv("BQ_MAX_QUERIES_PER_JOB", "7")
    assert bq.job_budget() == (12.5, 7)
    monkeypatch.setenv("BQ_MAX_GIB_PER_JOB", "not a number")
    assert bq.job_budget()[0] == 200.0, "a bad value falls back, it does not crash the job"


def test_a_job_that_has_spent_its_bytes_is_refused(monkeypatch):
    monkeypatch.setenv("BQ_MAX_GIB_PER_JOB", "10")
    metering.count_bq(9 * 2 ** 30)
    bq._check_job_budget("a lookup")          # still under
    metering.count_bq(2 * 2 ** 30)
    with pytest.raises(bq.BQJobBudgetSpent) as e:
        bq._check_job_budget("a lookup")
    assert "11.0 GiB of 10 GiB" in str(e.value) and "BQ_MAX_GIB_PER_JOB" in str(e.value)


def test_a_job_that_has_spent_its_query_count_is_refused(monkeypatch):
    monkeypatch.setenv("BQ_MAX_QUERIES_PER_JOB", "3")
    for _ in range(3):
        metering.count_bq(1024)
    with pytest.raises(bq.BQJobBudgetSpent):
        bq._check_job_budget("a lookup")


def test_running_out_is_recorded_as_an_incident_not_swallowed(monkeypatch):
    """An empty result and a refused one look identical to the caller unless
    one of them is written down. Every channel here already treats an
    exception as 'produced nothing', so the refusal raises — and the ledger
    gets the reason."""
    monkeypatch.setenv("BQ_MAX_QUERIES_PER_JOB", "1")
    metering.count_bq(1024)
    with pytest.raises(bq.BQJobBudgetSpent):
        bq._check_job_budget("a lookup")
    assert any("budget for this job is spent" in (i.get("detail") or i.get("message") or "")
               for i in metering.incidents), metering.incidents


def test_the_panel_can_read_what_this_job_spent(monkeypatch):
    monkeypatch.setenv("BQ_MAX_GIB_PER_JOB", "50")
    metering.count_bq(int(2.5 * 2 ** 30))
    sp = bq.job_spend()
    assert sp["gib"] == 2.5 and sp["queries"] == 1 and sp["cap_gib"] == 50 and sp["over"] is False
