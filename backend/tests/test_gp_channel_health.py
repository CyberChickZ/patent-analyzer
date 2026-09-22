"""Google Patents direct has no row in channel_specs, so until this existed the
report could not distinguish "Google refused us" from "Google had nothing"."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from nodes.search import google_patents_health


def _stats(calls, blocked, flat=False):
    if flat:
        return {"rounds": [], "gp_calls": calls, "gp_blocked": blocked}
    return {"rounds": [{"gp_calls": calls, "gp_blocked": blocked}]}


def test_on_cloud_every_request_refused_reads_retired_cloud(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "patent-analyzer")
    h = google_patents_health(_stats(2, 2))
    assert h["channel"] == "google_patents" and h["status"] == "retired(cloud)"
    assert h["attempted"] == 2 and h["refused"] == 2 and h["n"] == 0
    assert "unavailable on Cloud Run" in h["detail"] and "0 of 100" in h["detail"]


def test_on_cloud_a_job_that_got_through_is_not_called_retired(monkeypatch):
    """Job 99a35c00: 3 of 5 direct queries answered from Cloud Run, three hours
    after the 0/100 measurement. The label follows the job, not the folklore."""
    monkeypatch.setenv("K_SERVICE", "patent-analyzer")
    h = google_patents_health(_stats(5, 2))
    assert h["status"] == "limited" and h["n"] == 3
    assert "comes and goes" in h["detail"]


def test_off_cloud_the_row_says_nothing_about_cloud_run(monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    assert google_patents_health(_stats(4, 0))["status"] == "ok"
    h = google_patents_health(_stats(4, 1))
    assert h["status"] == "limited" and "Cloud Run" not in h["detail"]


def test_a_blocked_laptop_is_not_a_retired_cloud_channel(monkeypatch):
    """Off Cloud Run, every request refused is still a refusal — but it says
    nothing about the service's egress, and it must never read as "ok"."""
    monkeypatch.delenv("K_SERVICE", raising=False)
    h = google_patents_health(_stats(2, 2))
    assert h["status"] == "blocked" and h["n"] == 0
    assert "Cloud Run" not in h["detail"]
    assert "answered" not in h["detail"]


def test_a_run_that_never_asked_is_empty_not_refused(monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    h = google_patents_health(_stats(0, 0))
    assert h["status"] == "empty" and h["attempted"] == 0


def test_a_truncated_loop_carries_the_counters_flat(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "patent-analyzer")
    h = google_patents_health(_stats(3, 3, flat=True))
    assert h["status"] == "retired(cloud)" and h["attempted"] == 3


def test_counters_sum_across_rounds(monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    h = google_patents_health({"rounds": [{"gp_calls": 2, "gp_blocked": 1},
                                          {"gp_calls": 3, "gp_blocked": 0}]})
    assert h["attempted"] == 5 and h["refused"] == 1 and h["n"] == 4
