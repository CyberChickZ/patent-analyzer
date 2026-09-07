"""The per-job ledger: incidents, embeddings, and the three questions."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patent_analyzer import metering


@pytest.fixture(autouse=True)
def fresh():
    from app.llm import usage
    usage.clear()
    metering._job_id = None
    metering.start_run("ledger-test")
    yield
    usage.clear()


def _spend(model, prompt, output, thought=0, calls=1, seconds=1.0):
    from app.llm import usage
    m = usage.setdefault(model, {"calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                                 "thought_tokens": 0, "errors_429": 0, "seconds": 0.0})
    m["calls"] += calls
    m["prompt_tokens"] += prompt
    m["output_tokens"] += output
    m["thought_tokens"] += thought
    m["seconds"] += seconds


def test_incidents_land_in_the_phase_that_raised_them():
    metering.incident("gemini-3.8-flash", metering.RETRY, "429 from Vertex")
    _spend("gemini-3.8-flash", 1000, 100)
    idca = metering.mark("idca")

    metering.incident("serpapi", metering.EXHAUSTED, "key ab12cd34 monthly quota exhausted")
    metering.incident("bigquery", metering.DEGRADED, "fell back to the cheaper SQL")
    metering.count("serpapi:google_patents", 3)
    search = metering.mark("search")

    assert idca["retries"] == 1 and idca["failures"] == 0
    assert search["failures"] == 1 and search["degradations"] == 1 and search["retries"] == 0
    assert [i["kind"] for i in search["incidents"]] == [metering.EXHAUSTED, metering.DEGRADED]


def test_embeddings_are_counted_and_costed():
    metering.count_embed("text-embedding-005", texts=100, chars=400_000)
    d = metering.mark("search")["embedding"]["text-embedding-005"]
    assert d["requests"] == 1 and d["texts"] == 100
    assert d["est_tokens"] == 100_000                      # 4 chars/token
    assert d["cost_usd"] == pytest.approx(100_000 / 1e6 * metering.EMBED_USD_PER_MTOK, abs=1e-6)


def test_billable_characters_beat_the_estimate_when_the_api_reports_them():
    metering.count_embed("text-embedding-005", texts=2, chars=4_000, billable_chars=1_000)
    d = metering.mark("search")["embedding"]["text-embedding-005"]
    assert d["chars"] == 1_000 and d["est_tokens"] == 250


def test_ledger_answers_the_three_questions():
    _spend("gemini-3.8-flash", 100_000, 10_000)
    metering.mark("idca")
    _spend("gemini-2.5-pro", 1_000_000, 200_000, thought=100_000)
    metering.count("serpapi:google_patents", 2)
    metering.incident("gemini-2.5-pro", metering.RETRY, "429")
    metering.mark("search")

    led = metering.ledger()

    # 1. what did it cost
    assert led["totals"]["cost_usd"] == pytest.approx(0.1125 + 1.25 + 3.0, abs=1e-3)
    # 2. which step was the most expensive
    assert led["most_expensive"]["phase"] == "search"
    assert led["most_expensive"]["share_of_total"] > 0.9
    assert "gemini-2.5-pro" in led["most_expensive"]["driver"]
    # 3. which call failed or degraded
    assert led["totals"]["retries"] == 1
    assert led["incidents"] == [{"phase": "search", "source": "gemini-2.5-pro",
                                 "kind": "retry", "detail": "429"}]
    pro = next(r for r in led["rows"] if r["name"] == "gemini-2.5-pro")
    assert pro["retries"] == 1 and pro["phase"] == "search"


def test_rows_cover_every_kind_and_sort_by_cost():
    _spend("gemini-3.8-flash", 1_000_000, 0)
    metering.count("openalex", 4)
    metering.count_bq(2 ** 40)
    metering.count_embed("text-embedding-005", texts=10, chars=40_000)
    metering.mark("search")

    rows = metering.ledger()["rows"]
    assert {r["kind"] for r in rows} == {"model", "external", "bigquery", "embedding"}
    assert [r["cost_usd"] for r in rows] == sorted((r["cost_usd"] for r in rows), reverse=True)
    assert all(r["phase"] == "search" for r in rows)


def test_an_incident_on_a_channel_attaches_to_its_prefixed_row():
    metering.count("serpapi:google_scholar", 1)
    metering.incident("serpapi", metering.FAILED, "HTTP 500")
    metering.mark("search")
    row = next(r for r in metering.ledger()["rows"] if r["name"] == "serpapi:google_scholar")
    assert row["failures"] == 1


def test_a_free_job_has_no_most_expensive_step():
    metering.count("arxiv", 1)
    metering.mark("search")
    assert metering.ledger()["most_expensive"] is None


def test_a_spent_serpapi_key_reaches_the_ledger(tmp_path, monkeypatch):
    """The wiring, not the mechanism: a real channel hitting a real cap has to
    end up as a row the reader can see, not only as a returned error string."""
    import asyncio

    from patent_analyzer import cache
    from patent_analyzer.recall import serpapi as sp
    cache.reset_for_tests(tmp_path / "kv.sqlite")
    monkeypatch.setenv("SERPAPI_KEYS", "keyA")
    monkeypatch.setattr(sp, "FREE_TIER_PER_KEY", 1)
    monkeypatch.setattr(sp, "_sync_search", lambda *a: (
        [{"title": "T", "pub_num": "US1B2", "match_type": "Patent", "total": 1}], None))
    try:
        assert asyncio.run(sp.search_patents("q1"))[1] is None
        assert "exhausted" in asyncio.run(sp.search_patents("q2"))[1]
    finally:
        cache.reset_for_tests(None)

    m = metering.mark("search")
    assert m["failures"] == 1 and m["incidents"][0]["kind"] == metering.EXHAUSTED
    row = next(r for r in metering.ledger()["rows"] if r["name"].startswith("serpapi"))
    assert row["failures"] == 1


def test_a_new_job_clears_the_incidents_of_the_last_one():
    metering.incident("serpapi", metering.FAILED, "HTTP 500")
    metering.mark("search")
    metering.start_run("some-other-job")
    assert metering.incidents == [] and metering.embed == {}


# ── the price schedule ────────────────────────────────────────────────────────

def test_a_price_is_a_schedule_not_a_number():
    """gemini-3.8-flash is the global model and its introductory rate doubles on
    2027-01-01. A table with one tuple per model is silently 2x wrong that day."""
    assert metering.price_of("gemini-3.8-flash", "2026-12-31") == (0.75, 3.75)
    assert metering.price_of("gemini-3.8-flash", "2027-01-01") == (1.50, 7.50)
    assert metering.price_of("gemini-3.8-flash", "2030-06-15") == (1.50, 7.50)
    assert metering.price_of("gemini-2.5-pro", "2027-06-01") == (1.25, 10.00), \
        "2.5-pro has no scheduled change"
    assert metering.price_of("gemini-99-imaginary") is None


def test_a_run_is_costed_at_the_rate_in_force_on_its_day():
    same = dict(prompt_tokens=1_000_000, output_tokens=1_000_000, thought_tokens=0)
    before = metering.cost_usd("gemini-3.8-flash", on="2026-12-31", **same)
    after = metering.cost_usd("gemini-3.8-flash", on="2027-01-01", **same)
    assert before == 0.75 + 3.75
    assert after == 1.50 + 7.50 == before * 2


def test_the_schedule_is_announced_before_it_bites():
    changes = metering.upcoming_price_changes(after="2026-09-18")
    by_model = {c["model"]: c for c in changes}
    c = by_model["gemini-3.8-flash"]
    assert c["effective_from"] == "2027-01-01" and c["multiple"] == 2.0
    assert c["from_output"] == 3.75 and c["output_usd_per_mtok"] == 7.50
    assert metering.upcoming_price_changes(after="2027-01-01") == [], \
        "nothing is 'upcoming' once it has taken effect"
    assert metering.PRICE_REVIEW_DATE == "2027-01-01"


def test_the_caveat_names_only_the_models_the_run_actually_used():
    _spend("gemini-3.8-flash", 1000, 100)
    metering.mark("idca")
    caveats = metering.ledger()["caveats"]
    assert any("gemini-3.8-flash goes from" in c for c in caveats)
    assert not any("gemini-3.7-flash" in c for c in caveats), \
        "a change to a model nobody called is reference, not a caveat on this bill"
    # the full schedule is still carried, for anyone planning rather than reading a bill
    assert len(metering.ledger()["prices"]["upcoming_changes"]) == 3
