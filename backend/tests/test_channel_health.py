"""A recall channel that hangs, crashes or gets rate-limited must not take the
job with it, and the degradation has to reach the report — a narrower search
changes what "no blocking reference" is worth."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import nodes.search as search_mod
from patent_analyzer.report_sections import channel_health_html, channel_health_md, inject_html, inject_md


def _timed():
    """The per-channel wrapper from search_node, rebuilt here with the same body."""
    import time as _time

    async def t(name, fn):
        t0 = _time.monotonic()
        try:
            return await asyncio.wait_for(fn(), timeout=search_mod._channel_timeout(name)), _time.monotonic() - t0
        except asyncio.TimeoutError:
            return asyncio.TimeoutError(f"no result within {search_mod._channel_timeout(name):.0f}s"), _time.monotonic() - t0
        except Exception as exc:
            return exc, _time.monotonic() - t0
    return t


def test_channel_timeout_is_bounded_and_not_fatal(monkeypatch):
    monkeypatch.setenv("SEARCH_TIMEOUT_HANGER", "0.05")
    timed = _timed()

    async def hangs():
        await asyncio.sleep(30)

    async def fine():
        return ["a", "b"], []

    async def main():
        return await asyncio.gather(timed("hanger", hangs), timed("fine", fine))

    (r1, s1), (r2, _) = asyncio.run(main())
    assert isinstance(r1, asyncio.TimeoutError) and s1 < 5      # dropped, not waited on
    assert r2 == (["a", "b"], [])                                # the other channel still returns


def test_channel_timeout_overrides_and_env():
    assert search_mod._channel_timeout("agentic_loop") == 2400.0
    assert search_mod._channel_timeout("arxiv") == search_mod._CHANNEL_TIMEOUT_DEFAULT


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("SEARCH_CHANNEL_TIMEOUT_S", "7")
    assert search_mod._channel_timeout("arxiv") == 7.0
    monkeypatch.setenv("SEARCH_TIMEOUT_ARXIV", "3")
    assert search_mod._channel_timeout("arxiv") == 3.0


HEALTH = {"channel_health": [
    {"channel": "arxiv", "status": "ok", "n": 40, "seconds": 2.1, "detail": "", "errors": []},
    {"channel": "bigquery_patents", "status": "timeout", "n": 0, "seconds": 900.0,
     "detail": "no result within 900s", "errors": []},
    {"channel": "serpapi_patents", "status": "limited", "n": 3, "seconds": 12.0,
     "detail": "serpapi budget exhausted", "errors": ["serpapi budget exhausted"]},
]}


def test_report_names_the_degraded_channels():
    h = channel_health_html(HEALTH)
    assert "Search Coverage Warnings" in h
    assert "bigquery_patents" in h and "timed out" in h
    assert "serpapi_patents" in h and "rate-limited" in h
    assert "arxiv" in h                       # listed as healthy, not as a warning
    assert "narrower sample" in h

    md = channel_health_md(HEALTH)
    assert any("bigquery_patents" in line for line in md)
    assert any("serpapi_patents" in line for line in md)


def test_clean_run_says_nothing():
    clean = {"channel_health": [{"channel": "arxiv", "status": "ok", "n": 40}]}
    assert channel_health_html(clean) == "" and channel_health_md(clean) == []
    assert channel_health_html({}) == "" and channel_health_md(None) == []


def test_warnings_reach_the_injected_report():
    html = inject_html('<div class="sec-t">Invention Summary</div><div>x</div>\n</div>', None, HEALTH, None, None)
    assert "Search Coverage Warnings" in html and "bigquery_patents" in html
    md = inject_md("# R\n\n## Evaluation Criteria\n", None, HEALTH, None, None)
    assert "## Search Coverage Warnings" in md and "bigquery_patents" in md


# ── evidence coverage ──
#
# A reference evaluated with source "no_content" was read from nothing, yet it
# occupies a row in the evaluation and counts as "checked" behind a determination
# of "no blocking reference". Measured on the M2 e2e runs: 17/25 and 24/25.

from patent_analyzer.report_sections import evidence_coverage_html, evidence_coverage_md  # noqa: E402

MIX = [{"source": "pdf"}] * 6 + [{"source": "abstract"}] * 2 + [{"source": "no_content"}] * 17


def test_evidence_coverage_counts_and_warns():
    h = evidence_coverage_html(MIX)
    assert "17 of 25 references were evaluated with no text at all" in h
    assert "68%" in h and "unchecked reference, not a cleared one" in h
    assert "full PDF" in h and "nothing to read" in h
    md = evidence_coverage_md(MIX)
    assert md[0] == "## Evidence Coverage"
    assert any("17 of 25" in line for line in md)


def test_evidence_coverage_silent_when_everything_was_read():
    assert evidence_coverage_html([{"source": "pdf"}, {"source": "full_text"}]) == ""
    assert evidence_coverage_md([{"source": "pdf"}]) == []
    assert evidence_coverage_html([]) == "" and evidence_coverage_md(None) == []


def test_evidence_coverage_reaches_the_report():
    html = inject_html('<div class="sec-t">Invention Summary</div><div>x</div>\n</div>', None, None, MIX, None)
    assert "Evidence Coverage" in html and "17 of 25" in html
    md = inject_md("# R\n\n## Evaluation Criteria\n", None, None, MIX, None)
    assert "## Evidence Coverage" in md
