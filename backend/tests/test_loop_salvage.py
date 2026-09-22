"""A deadline is a reason to stop looking, not a reason to throw away what was
found. Job e7f847bf (2026-09-19) cancelled the recall loop after 8 Lens queries
and 46.4 GiB of BigQuery and shipped a delivery with 0 patents in it."""

import asyncio
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import nodes.search as search_mod
from patent_analyzer.agentic.loop import _register_partial, run_loop, run_wide


def test_run_loop_and_run_wide_still_take_a_partial_sink():
    assert "partial" in inspect.signature(run_loop).parameters
    assert "partial" in inspect.signature(run_wide).parameters


def test_registered_containers_are_live_not_copies():
    sink: dict = {}
    pool: dict = {}
    _register_partial(sink, pool=pool, mode="wide_good")
    pool["US1A"] = "cand"                      # the run keeps filling it afterwards
    assert sink["pool"] is pool and sink["pool"]["US1A"] == "cand"
    _register_partial(None, pool=pool)          # no sink is not an error


def _partial(n_pool=3, n_log=7):
    class B:
        gp_calls, serp_calls = 2, 5
    return {"pool": {f"US{i}": f"cand{i}" for i in range(n_pool)},
            "log": [{}] * n_log, "budget": B(), "mode": "wide_good",
            "cands": [{"id": "inv1", "level": "core",
                       "elements": [{"id": "inv1.e0", "text": "A", "facets": {"thing": ["x"]}},
                                    {"id": "inv1.e1", "text": "B"}]}]}


def test_salvage_keeps_what_the_loop_had_found():
    cands, stats = search_mod.salvage_timed_out_loop(_partial(), 2400.0)
    assert len(cands) == 3 and stats["salvaged"] == 3
    assert stats["queries_issued"] == 7
    assert stats["gp_calls"] == 2 and stats["serpapi_calls"] == 5


def test_salvage_says_out_loud_that_the_search_was_cut_short():
    _, stats = search_mod.salvage_timed_out_loop(_partial(), 2400.0)
    assert stats["truncated"] is True and stats["truncated_after_s"] == 2400.0


def test_salvage_does_not_rewrite_the_mode_and_strand_the_claims_judge():
    # delivery branches on mode and on elements being non-empty; a salvage that
    # lost either would route the job around the judge without saying so
    _, stats = search_mod.salvage_timed_out_loop(_partial(), 2400.0)
    assert stats["mode"] == "wide_good"
    assert [e["id"] for e in stats["elements"]] == ["inv1.e0", "inv1.e1"]
    assert stats["candidates"] == [{"id": "inv1", "level": "core", "n_elements": 2}]


def test_salvage_of_an_empty_run_is_empty_but_still_marked():
    cands, stats = search_mod.salvage_timed_out_loop({}, 60.0)
    assert cands == [] and stats["salvaged"] == 0 and stats["truncated"] is True


def test_a_cancelled_run_leaves_its_findings_in_the_caller_s_sink():
    """End to end on the mechanism: cancel mid-run, read the sink."""
    sink: dict = {}

    async def fake_run(partial):
        pool: dict = {}
        _register_partial(partial, pool=pool, mode="wide_good")
        for i in range(100):
            pool[f"US{i}"] = i
            await asyncio.sleep(0.01)

    async def main():
        try:
            await asyncio.wait_for(fake_run(sink), timeout=0.05)
        except asyncio.TimeoutError:
            return search_mod.salvage_timed_out_loop(sink, 0.05)

    cands, stats = asyncio.run(main())
    assert len(cands) > 0                      # the whole point: not zero
    assert stats["truncated"] is True
