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
    # coverage no longer stops the loop before MIN_ROUNDS — that is what let m1a's first paper
    # stop after round 1 with reach 0/5, on one-element GOOD it had counted as coverage
    assert M.done({"e1": 3, "e2": 3}, 5, 1, 0) is None
    assert M.done({"e1": 3, "e2": 3}, 5, M.MIN_ROUNDS, 0) == f"every element covered by >={M.COVER_TARGET} strong GOOD"
    assert M.done({"e1": 3, "e2": 0}, 5, M.MIN_ROUNDS, 0) is None
    assert M.done({"e1": 0}, 0, 2, 2) is None                    # under MIN_ROUNDS, keep walking
    assert M.done({"e1": 0}, 0, M.MIN_ROUNDS, M.DRY_ROUNDS) == f"{M.DRY_ROUNDS} consecutive rounds with no new strong GOOD"
    assert M.done({"e1": 0}, 0, M.MIN_ROUNDS, 1) is None         # one dry round is not two
    assert M.done({"e1": 0}, 9, M.MAX_ROUNDS, 0) == f"{M.MAX_ROUNDS} rounds"


def test_p5_pages_on_from_where_the_last_round_stopped():
    """m1a round 1: P5 made 4 calls and brought back 0, because it re-read the
    pages round 0 had already put in `known`."""
    seen = []

    async def fake_enum(terms, cpc, before=None, max_records=3000, start=0):
        seen.append((cpc, start))
        base = start // 100
        return [_c(f"US{cpc}{base}{i}A1", "odp") for i in range(100)], 5000, None
    import patent_analyzer.recall.uspto_odp as odp
    orig, odp.enumerate_group = odp.enumerate_group, fake_enum
    try:
        offsets = {}
        r1 = asyncio.run(M.p5_cpc_enum(["H04N7"], ["camera"], None, set(), cap=100, offsets=offsets))
        r2 = asyncio.run(M.p5_cpc_enum(["H04N7"], ["camera"], None, set(), cap=100, offsets=offsets))
    finally:
        odp.enumerate_group = orig
    assert seen == [("H04N7", 0), ("H04N7", 100)]
    assert len(r1.candidates) == 100 and len(r2.candidates) == 100
    assert not {c.pub_num for c in r1.candidates} & {c.pub_num for c in r2.candidates}


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
                       "seconds": 4.1, "error": None, "note": "G05D1:162 ", "select": None}
