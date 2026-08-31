"""Per-phase accounting: time, LLM calls/tokens, external calls, estimated cost.

The dollar figures are checked by hand against the price table so a typo in
PRICES cannot pass silently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm
from patent_analyzer import metering
from patent_analyzer.report_sections import cost_html, cost_md


def _reset(monkeypatch):
    monkeypatch.setattr(llm, "usage", {})
    metering._job_id = None
    metering.start_run("job-under-test")


def _spend(model, calls=1, pin=0, pout=0, pthought=0, secs=0.0):
    m = llm.usage.setdefault(model, {"calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                                     "thought_tokens": 0, "errors_429": 0, "seconds": 0.0})
    m["calls"] += calls
    m["prompt_tokens"] += pin
    m["output_tokens"] += pout
    m["thought_tokens"] += pthought
    m["seconds"] += secs


def test_cost_matches_the_published_prices():
    # gemini-3.8-flash: $0.75/M in, $3.75/M out; thought tokens bill as output
    assert metering.cost_usd("gemini-3.8-flash", 1_000_000, 0, 0) == 0.75
    assert metering.cost_usd("gemini-3.8-flash", 0, 500_000, 500_000) == 3.75
    # gemini-2.5-pro: $1.25/M in, $10/M out
    assert metering.cost_usd("gemini-2.5-pro", 1_000_000, 100_000, 0) == 1.25 + 1.0
    # gemini-3.1-flash-lite: $0.25/M in, $1.50/M out
    assert metering.cost_usd("gemini-3.1-flash-lite", 2_000_000, 1_000_000, 0) == 0.5 + 1.5
    # an unknown model is counted, not costed
    assert metering.cost_usd("gemini-99-imaginary", 1_000_000, 1_000_000, 0) == 0.0


def test_phase_deltas_are_per_phase(monkeypatch):
    _reset(monkeypatch)
    _spend("gemini-2.5-pro", calls=2, pin=100_000, pout=10_000, secs=3.0)
    metering.count("serpapi:google_patents", 2)
    idca = metering.mark("idca")

    _spend("gemini-3.8-flash", calls=5, pin=1_000_000, pout=200_000, pthought=50_000)
    metering.count("openalex", 7)
    metering.count_bq(2 ** 40)                      # exactly 1 TiB
    search = metering.mark("search")

    assert idca["llm_calls"] == 2 and idca["external_calls"] == 2
    assert idca["llm"]["gemini-2.5-pro"]["prompt_tokens"] == 100_000
    assert idca["cost_usd"] == round(0.125 + 0.1, 4)

    # the second phase must NOT re-count the first
    assert search["llm_calls"] == 5
    assert "gemini-2.5-pro" not in search["llm"]
    assert search["external"] == {"openalex": 7}
    assert search["bigquery"] == {"queries": 1, "gib_billed": 1024.0, "cost_usd": 6.25}
    assert search["cost_usd"] == round(0.75 + (200_000 + 50_000) * 3.75 / 1e6 + 6.25, 4)

    t = metering.totals()
    assert t["llm_calls"] == 7 and t["external_calls"] == 9
    assert t["cost_usd"] == round(idca["cost_usd"] + search["cost_usd"], 4)
    assert t["llm"]["gemini-2.5-pro"]["calls"] == 2 and t["llm"]["gemini-3.8-flash"]["calls"] == 5


def test_mark_is_idempotent_so_a_resumed_gate_cannot_clobber(monkeypatch):
    """A gate that pauses re-executes its whole body on resume; the second
    mark must return the phase as it was, not the ~0 measured across the pause."""
    _reset(monkeypatch)
    _spend("gemini-3.8-flash", calls=4, pin=100_000)
    first = metering.mark("extract")
    assert first["llm_calls"] == 4
    again = metering.mark("extract")
    assert again == first
    assert metering.phases()["extract"]["llm_calls"] == 4


def test_start_run_keeps_the_same_job_and_clears_a_new_one(monkeypatch):
    _reset(monkeypatch)
    _spend("gemini-3.8-flash", calls=1, pin=1000)
    metering.mark("idca")
    metering.start_run("job-under-test")            # a resume
    assert "idca" in metering.phases()
    metering.start_run("a-different-job")
    assert metering.phases() == {} and metering.external == {}


def test_report_block_shape(monkeypatch):
    _reset(monkeypatch)
    _spend("gemini-3.8-flash", calls=3, pin=10_000, pout=1_000)
    metering.mark("idca")
    r = metering.report()
    assert r["job_id"] == "job-under-test"
    assert r["prices_usd_per_mtok"]["gemini-2.5-pro"] == {"input": 1.25, "output": 10.0}
    assert r["bigquery_usd_per_tib"] == 6.25
    assert "not a bill" in r["note"].lower()
    assert r["phases"]["idca"]["llm_calls"] == 3 and r["totals"]["llm_calls"] == 3


COST = {"note": "Estimated from list prices; thought tokens billed as output. Not a bill.",
        "bigquery_usd_per_tib": 6.25,
        "phases": {"idca": {"seconds": 12.0, "llm_calls": 2, "external_calls": 0, "cost_usd": 0.03,
                            "llm": {"gemini-2.5-pro": {"calls": 2, "prompt_tokens": 20000, "output_tokens": 1000,
                                                       "thought_tokens": 500, "errors_429": 0, "cost_usd": 0.03}},
                            "external": {}, "bigquery": {"queries": 0, "gib_billed": 0, "cost_usd": 0}},
                   "search": {"seconds": 240.0, "llm_calls": 9, "external_calls": 31, "cost_usd": 6.4,
                              "llm": {"gemini-3.8-flash": {"calls": 9, "prompt_tokens": 500000, "output_tokens": 30000,
                                                           "thought_tokens": 9000, "errors_429": 2, "cost_usd": 0.15}},
                              "external": {"openalex": 24, "serpapi:google_patents": 7},
                              "bigquery": {"queries": 1, "gib_billed": 1024.0, "cost_usd": 6.25}}},
        "totals": {"seconds": 252.0, "llm_calls": 11, "external_calls": 31, "cost_usd": 6.43,
                   "llm": {"gemini-2.5-pro": {"calls": 2, "prompt_tokens": 20000, "output_tokens": 1000,
                                              "thought_tokens": 500, "errors_429": 0, "cost_usd": 0.03},
                           "gemini-3.8-flash": {"calls": 9, "prompt_tokens": 500000, "output_tokens": 30000,
                                                "thought_tokens": 9000, "errors_429": 2, "cost_usd": 0.15}},
                   "external": {"openalex": 24, "serpapi:google_patents": 7},
                   "bigquery": {"queries": 1, "gib_billed": 1024.0, "cost_usd": 6.25}}}


def test_report_renders_the_cost_block():
    h = cost_html(COST)
    assert "Run Cost" in h and "not a bill" in h.lower()
    assert "3 · Prior-art recall" in h and "$6.4300" in h        # total
    assert "4.0m" in h                                            # 240 s formatted
    assert "2&times; 429" in h                                    # 429s surfaced
    assert "openalex</code> 24" in h
    assert "BigQuery: 1 queries, 1024.0 GiB billed, $6.2500" in h

    md = cost_md(COST)
    assert md[0] == "## Run Cost"
    assert any("**$6.4300**" in line for line in md)


def test_no_metrics_renders_nothing():
    assert cost_html({}) == "" and cost_html(None) == ""
    assert cost_md({"phases": {}}) == []


# ── the gate writes it into graph state ──

def test_gate_emits_phase_metrics_and_the_reducer_protects_it(monkeypatch):
    import asyncio

    from graph.gates import make_gate
    from state import _merge_nonempty

    _reset(monkeypatch)
    _spend("gemini-3.8-flash", calls=2, pin=5000)
    patch = asyncio.run(make_gate("idca")({"pause_after": []}))
    assert patch["phase_metrics"]["idca"]["llm_calls"] == 2

    # a gate re-running after a resume must not blank a recorded phase
    merged = _merge_nonempty(patch["phase_metrics"], {"idca": {}})
    assert merged["idca"]["llm_calls"] == 2
    merged = _merge_nonempty(patch["phase_metrics"], {"search": {"llm_calls": 3}})
    assert merged["search"]["llm_calls"] == 3 and merged["idca"]["llm_calls"] == 2
