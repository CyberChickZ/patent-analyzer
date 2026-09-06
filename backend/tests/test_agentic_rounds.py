import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic import good as G
from patent_analyzer.agentic import moves as M
from patent_analyzer.agentic import rounds as R
from patent_analyzer.recall.pool import Candidate

ELS = [{"id": "e1", "text": "a camera", "facets": {"thing": ["stereo camera"]}},
       {"id": "e2", "text": "a marker", "facets": {"thing": ["fiducial marker"]}}]


def _quiet(monkeypatch):
    monkeypatch.setattr(R, "_score_cos", lambda els, docs: None)


def test_groups_of_reads_the_good_documents_own_cpc():
    docs = [{"raw": {"cpc": ["H04N   7/15", "G06T   7/00"]}}, {"raw": {"cpc": ["H04N   7/14"]}}]
    assert R._groups_of(docs)[0] == "H04N7"


def test_seed_meta_asks_odp_because_no_table_of_ours_has_a_party(monkeypatch):
    """The m1a bug: this read raw.inventor / raw.applicant, which BigQuery
    candidates never carry, so P6 was never called once in eight papers."""
    asked = {}

    async def fake_find(pubs, limit=100):
        asked["pubs"] = pubs
        return {"US1A1": {"applicationMetaData": {"firstInventorName": "Yulun Wang"}},
                "US2A1": {"applicationMetaData": {}}}
    import patent_analyzer.recall.uspto_odp as odp
    monkeypatch.setattr(odp, "find_many_by_publication", fake_find)
    got = asyncio.run(R._seed_meta([{"pub_num": "US1A1", "raw": {}}, {"pub_num": "US2A1", "raw": {}}]))
    assert asked["pubs"] == ["US1A1", "US2A1"]
    assert got == [{"inventor": "Yulun Wang", "applicant": None}]


def test_terms_for_follows_the_uncovered_elements():
    assert R._terms_for(ELS, ["e2"]) == ["fiducial", "marker"]
    assert R._terms_for(ELS, []) == ["stereo", "camera", "fiducial", "marker"]   # nothing named -> all


def _harness(monkeypatch, judge_fn, n_round0=6):
    _quiet(monkeypatch)

    async def round0():
        cands = [Candidate(pub_num=f"US{i}B2", match_type="Patent") for i in range(n_round0)]
        return cands, [M.MoveResult("S2_predicted_cpc", 0, cands)]

    async def fake_claims(cands):
        return {c.pub_num: f"1. A camera and a marker ({c.pub_num})." for c in cands}, 1
    monkeypatch.setattr(R, "_claims_for", fake_claims)
    monkeypatch.setattr(G, "judge", judge_fn)
    return round0


def test_round_1_expands_from_what_was_read_not_only_from_good(monkeypatch):
    """The core m1a fix: the gold arrives one hop from a query hit, and that
    hit is usually not GOOD itself, so round 1 must seed from the whole judged
    set ranked by (touches, cosine)."""
    seen = {}

    async def judge_nothing(elements, docs, claims, call=None):
        for d in docs:
            d["good"], d["good_touches"] = False, {}
        return {"judged": len(docs), "calls": 1, "good": 0, "with_claims": len(docs)}
    round0 = _harness(monkeypatch, judge_nothing)

    async def fake_wide(judged, pool, before, known, round_no):
        seen["seeds"] = [d["pub_num"] for d in judged]
        seen["pool"] = [c.pub_num for c in pool]
        return [M.MoveResult("W1_citations", round_no, [])]

    async def fake_later(*a, **k):
        seen["later"] = seen.get("later", 0) + 1
        return [M.MoveResult("P1_citations", 2, [])]
    monkeypatch.setattr(R, "_wide_round", fake_wide)
    monkeypatch.setattr(R, "_later_round", fake_later)
    out = asyncio.run(R.run_rounds(ELS, ["H04N7"], ["camera"], "20110202", round0))
    assert seen["seeds"] == [f"US{i}B2" for i in range(6)]        # none of them GOOD, all still seeds
    assert seen["pool"] == [f"US{i}B2" for i in range(6)]        # and the whole pool is offered, read or not
    assert out["rounds"] == 3 and out["stop"].endswith("no new strong GOOD")


def test_one_element_good_does_not_cover_and_does_not_stop_the_loop(monkeypatch):
    """m1a round 0 judged 17% GOOD on one element each, which met 'every
    element covered by >=3 GOOD' immediately. Reach was 0/5."""
    async def judge_weak(elements, docs, claims, call=None):
        for d in docs:
            d["good"], d["good_touches"] = True, {"e1": 1}
        return {"judged": len(docs), "calls": 1, "good": len(docs), "with_claims": len(docs)}
    round0 = _harness(monkeypatch, judge_weak)

    async def empty(*a, **k):
        return [M.MoveResult("W1_citations", 1, [])]
    monkeypatch.setattr(R, "_wide_round", empty)
    monkeypatch.setattr(R, "_later_round", empty)
    out = asyncio.run(R.run_rounds(ELS, [], [], None, round0))
    assert out["coverage"] == {"e1": 0, "e2": 0} and out["strong"] == 0
    assert out["rounds"] >= M.MIN_ROUNDS
    assert out["uncovered"] == ["e1", "e2"]


def test_strong_good_covers_and_the_round_row_carries_the_uncovered_list(monkeypatch):
    async def judge_strong(elements, docs, claims, call=None):
        for d in docs:
            d["good"], d["good_touches"] = True, {"e1": 1, "e2": 4}
        return {"judged": len(docs), "calls": 1, "good": len(docs), "with_claims": len(docs)}
    round0 = _harness(monkeypatch, judge_strong)

    async def empty(*a, **k):
        return [M.MoveResult("W1_citations", 1, [])]
    monkeypatch.setattr(R, "_wide_round", empty)
    monkeypatch.setattr(R, "_later_round", empty)
    out = asyncio.run(R.run_rounds(ELS, [], [], None, round0))
    assert out["coverage"] == {"e1": 6, "e2": 6} and out["strong"] == 6
    assert out["rounds"] == M.MIN_ROUNDS and out["stop"].startswith("every element covered")
    row0 = [r for r in out["rows"] if r["move"] == "_round"][0]
    assert row0["new_strong"] == 6 and row0["uncovered"] == [] and row0["dry_streak"] == 0


def test_later_rounds_aim_their_moves_at_the_uncovered_elements(monkeypatch):
    got = {}

    async def fake_p5(groups, terms, before, known, cap=0, round_no=0, name="P5_cpc_enum", offsets=None):
        got["groups"], got["terms"] = groups, terms
        return M.MoveResult(name, round_no, [])

    async def nothing(*a, **k):
        return M.MoveResult("x", 2, [])

    async def no_meta(docs, cap=20):
        return []
    monkeypatch.setattr(M, "p5_cpc_enum", fake_p5)
    monkeypatch.setattr(M, "p1_citations", nothing)
    monkeypatch.setattr(M, "p4_similar", nothing)
    monkeypatch.setattr(M, "p3_cited_papers", nothing)
    monkeypatch.setattr(M, "p6_same_party", nothing)
    monkeypatch.setattr(R, "_seed_meta", no_meta)
    good = [{"pub_num": "US1A1", "good": True, "good_touches": {"e2": 3},
             "raw": {"cpc": ["G01S  17/00"]}},
            {"pub_num": "US2A1", "good": True, "good_touches": {"e1": 1, "e3": 2},
             "raw": {"cpc": ["H04N   7/15"]}}]
    res = asyncio.run(R._later_round(good, ELS, ["Z99Z9"], ["camera"], None, set(), 2, ["e2"], {}))
    assert got["terms"] == ["fiducial", "marker"]                 # the uncovered element's own words
    assert got["groups"][0] == "G01S17"                           # the GOOD doc that touches it, first
    assert all("aimed at 1 uncovered: e2" in r.note for r in res)


def test_select_for_reading_ranks_inside_the_must_read_set_too(monkeypatch):
    """m1a/m1b read the first 300 of 2,140 in channel order and dropped the
    rest from the pool entirely. m1c kept the pool but 2,227 of 3,229
    candidates were "must-read", so must[:300] was an arbitrary cut again —
    and it dropped US20090147070A1, a gold family that arrived via lens_search.
    Each tier is cosine-ranked inside itself now."""
    def fake_embed(elements, rows, topk=1, cap=1, **kw):
        for r in rows:
            r["prune_cos"] = 0.95 if r["title"] == "gold" else (0.2 if r["title"].startswith("l") else 0.4)
        return list(range(len(rows))), {}
    import patent_analyzer.agentic.prune as P
    monkeypatch.setattr(P, "stage1_embed", fake_embed)
    cands = [Candidate(pub_num=f"Q{i}", title=f"q{i}", match_type="Patent",
                       sources=["google_patents"], raw={"loop": {"rank": i}}) for i in range(5)]
    cands += [Candidate(pub_num=f"L{i}", title=f"l{i}", match_type="Patent",
                        sources=["lens_search"]) for i in range(800)]
    cands.append(Candidate(pub_num="GOLD", title="gold", match_type="Patent", sources=["lens_search"]))
    picked, info = R.select_for_reading([{"id": "e1", "text": "a camera"}], cands, 20)
    got = [c.pub_num for c in picked]
    assert all(f"Q{i}" in got for i in range(5))      # every query's top ten is reserved
    assert "GOLD" in got                              # and the best of 801 Lens hits beats the rest
    assert info["in"] == 806 and info["read"] == 20
    assert info["tiers"] == [5, 801, 0] and info["read_per_tier"] == [5, 15, 0]
