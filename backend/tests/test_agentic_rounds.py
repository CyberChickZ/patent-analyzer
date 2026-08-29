import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic import good as G
from patent_analyzer.agentic import moves as M
from patent_analyzer.agentic import rounds as R
from patent_analyzer.recall.pool import Candidate

ELS = [{"id": "e1", "text": "a camera"}, {"id": "e2", "text": "a marker"}]


def test_groups_of_reads_the_good_documents_own_cpc():
    docs = [{"raw": {"cpc": ["H04N   7/15", "G06T   7/00"]}}, {"raw": {"cpc": ["H04N   7/14"]}}]
    assert R._groups_of(docs)[0] == "H04N7"


def test_seed_meta_keeps_only_documents_with_a_party():
    docs = [{"raw": {"inventor": "Yulun Wang"}}, {"raw": {}}, {"raw": {"applicant": "InTouch"}}]
    assert R._seed_meta(docs) == [{"inventor": "Yulun Wang", "applicant": None},
                                  {"inventor": None, "applicant": "InTouch"}]


def test_round0_only_when_the_stop_rule_fires(monkeypatch):
    # every element covered by COVER_TARGET GOOD in round 0 -> one round, no later moves
    async def round0():
        cands = [Candidate(pub_num=f"US{i}B2", match_type="Patent") for i in range(6)]
        return cands, [M.MoveResult("S2_predicted_cpc", 0, cands)]

    async def fake_claims(cands):
        return {c.pub_num: f"1. A camera and a marker ({c.pub_num})." for c in cands}, 1

    async def fake_judge(elements, docs, claims, call=None):
        for d in docs:
            d["good_touches"] = {"e1": 1, "e2": 1}
            d["good"] = True
        return {"judged": len(docs), "calls": 1, "good": len(docs), "with_claims": len(docs)}
    monkeypatch.setattr(R, "_claims_for", fake_claims)
    monkeypatch.setattr(G, "judge", fake_judge)

    async def boom(*a, **k):
        raise AssertionError("no later round expected")
    monkeypatch.setattr(R, "_later_round", boom)
    out = asyncio.run(R.run_rounds(ELS, ["H04N7"], ["camera"], "20110202", round0))
    assert out["rounds"] == 1 and out["stop"].startswith("every element covered")
    assert len(out["good"]) == 6 and out["coverage"] == {"e1": 6, "e2": 6}
    assert [r for r in out["rows"] if r["move"] == "_round"][0]["new_good"] == 6


def test_a_round_that_adds_no_good_stops_the_loop(monkeypatch):
    async def round0():
        return [], [M.MoveResult("S2_predicted_cpc", 0, [])]

    async def fake_claims(cands):
        return {}, 0

    async def fake_judge(elements, docs, claims, call=None):
        return {"judged": 0, "calls": 0, "good": 0, "with_claims": 0}
    calls = {"n": 0}

    async def fake_later(*a, **k):
        calls["n"] += 1
        return [M.MoveResult("P1_citations", 1, [])]
    monkeypatch.setattr(R, "_claims_for", fake_claims)
    monkeypatch.setattr(G, "judge", fake_judge)
    monkeypatch.setattr(R, "_later_round", fake_later)
    out = asyncio.run(R.run_rounds(ELS, [], [], None, round0))
    assert out["stop"] == "a round added no GOOD" and out["rounds"] == 2 and calls["n"] == 1
