import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic import moves as M
from patent_analyzer.recall.pool import Candidate


def _c(n, src):
    return Candidate(pub_num=n, match_type="Patent", sources=[src])


def test_round_budget_takes_from_every_move_in_turn():
    a = M.MoveResult("P1_citations", 1, [_c(f"US{i}A1", "P1") for i in range(10)])
    b = M.MoveResult("P5_cpc_enum", 1, [_c(f"EP{i}A1", "P5") for i in range(10)])
    got = M.round_budget([a, b], cap=6)
    assert [c.pub_num for c in got] == ["US0A1", "EP0A1", "US1A1", "EP1A1", "US2A1", "EP2A1"]


def test_round_budget_drops_duplicates_across_moves():
    a = M.MoveResult("P1_citations", 1, [_c("US1A1", "P1"), _c("US2A1", "P1")])
    b = M.MoveResult("P4_similar", 1, [_c("US1A1", "P4"), _c("US3A1", "P4")])
    assert [c.pub_num for c in M.round_budget([a, b], cap=10)] == ["US1A1", "US2A1", "US3A1"]


def test_stop_conditions():
    assert M.done({"e1": 3, "e2": 3}, 5, 1) == f"every element covered by >={M.COVER_TARGET} GOOD"
    assert M.done({"e1": 3, "e2": 0}, 5, 1) is None
    assert M.done({"e1": 0}, 0, 2) == "a round added no GOOD"
    assert M.done({"e1": 0}, 9, 0) is None                       # round 0 may add nothing yet
    assert M.done({"e1": 0}, 9, M.MAX_ROUNDS) == f"{M.MAX_ROUNDS} rounds"


def test_a_move_with_no_seeds_is_free():
    r = asyncio.run(M.p1_citations([], set()))
    assert r.candidates == [] and r.calls == 0 and r.error is None
    r = asyncio.run(M.p4_similar([], set()))
    assert r.candidates == [] and r.calls == 0
    assert asyncio.run(M.p5_cpc_enum([], ["marker"], None, set())).calls == 0
    assert asyncio.run(M.p6_same_party([], None, set())).calls == 0


def test_move_result_row_is_the_accounting_line():
    r = M.MoveResult("P5_cpc_enum", 2, [_c("US1A1", "P5")], calls=3, gib=1.25, seconds=4.06, note="G05D1:162 ")
    assert r.row() == {"move": "P5_cpc_enum", "round": 2, "brought": 1, "calls": 3, "gib": 1.25,
                       "seconds": 4.1, "error": None, "note": "G05D1:162 "}
