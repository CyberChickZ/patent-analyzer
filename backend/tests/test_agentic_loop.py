import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import cache
from patent_analyzer.agentic import loop as L
from patent_analyzer.recall.pool import Candidate


@pytest.fixture(autouse=True)
def _kv(tmp_path, monkeypatch):
    cache.reset_for_tests(tmp_path / "kv.sqlite")
    monkeypatch.setattr(L, "LOOP_MODE", "elements")
    yield
    cache.reset_for_tests(None)


def _cand(pub, title):
    return Candidate(title=title, pub_num=pub, match_type="Patent", raw={"google_patents": {}})


def test_loop_rounds_and_budget(monkeypatch):
    state = {"summary": "s", "checklist": [
        {"id": "e1", "criterion": "gaze estimation for video conferencing"},
        {"id": "e2", "criterion": "quantum tunneling widget"}]}

    async def fake_facets(els, summary):
        return {"e1": {"thing": ["gaze"], "place": ["conferenc"], "apparatus": ["camera"]},
                "e2": {"thing": ["quantum"], "place": ["widget"], "apparatus": []}}
    monkeypatch.setattr("app.llm.facet_elements", fake_facets)

    calls = []

    async def fake_gp(query, num=20, page=0, before=None):
        calls.append(query)
        L.gp.last_total[query] = 50
        if "gaze" in query:
            return [_cand("US1", "Horizontal gaze estimation for video conferencing")], None
        return [], None
    monkeypatch.setattr(L.gp, "search", fake_gp)
    monkeypatch.setattr(L.gp, "is_blocked", lambda: False)

    async def fake_expand(seeds, known, max_cited=200, before=None):
        return [_cand("US7", "cited art")], {"cpc_subclasses": {"H04N": 1}}
    monkeypatch.setattr(L, "expand", fake_expand)

    events = []
    serp_left = {"n": 8}
    cands, stats = asyncio.run(L.run_loop(state, lambda: serp_left["n"], lambda: False,
                                          lambda k, m, p=None: events.append((k, p))))
    pubs = {c.pub_num for c in cands}
    assert {"US1", "US7"} <= pubs
    rounds = stats["rounds"]
    assert 1 <= len(rounds) <= 3
    assert rounds[0]["covered"] == ["e1"] and "e2" in rounds[0]["uncovered"]
    assert rounds[0]["cpc_hint"] == "H04N"
    # e2 never covered: rounds 2 (CL=) and 3 (CPC=) both tried
    assert any("CL=(" in q["query"] for q in rounds[1]["queries"]) if len(rounds) > 1 else True
    assert all("AB=(" not in q["query"] for q in rounds[0]["queries"])
    assert all("CPC=" not in q["query"] for q in rounds[2]["queries"]) if len(rounds) > 2 else True   # subclass hint → no clause (needs a main group)
    assert all(k == "round_done" for k, _ in events)


def test_expand_drops_seeds_and_cited_on_or_after_cutoff(monkeypatch):
    from patent_analyzer.agentic import expand as E

    async def fake_meta(pubs, with_claims=False):
        rows = {"US1B2": {"priority_date": "2010-05-01", "family_id": "f1", "cpc_codes": ["H04N7/15"]},
                "US2B2": {"priority_date": "2011-02-02", "family_id": "f2", "cpc_codes": ["G06F3/00"]},
                "US9A": {"priority_date": "2005-01-01", "family_id": "f9", "title": "old art"},
                "US8A": {"priority_date": "2011-06-01", "family_id": "f8", "title": "later art"}}
        return {p: rows[p] for p in pubs if p in rows}

    async def fake_cits(pubs):
        assert "US2B2" not in pubs
        return {"US1B2": {"cits": [{"cited": "US9A", "category": "SEA"}, {"cited": "US8A", "category": "APP"}]}}
    monkeypatch.setattr(E, "fetch_by_pub_nums", fake_meta)
    monkeypatch.setattr(E, "fetch_citations", fake_cits)
    out, info = asyncio.run(E.expand(["US1B2", "US2B2"], set(), before="20110202"))
    assert info["seeds_after_cutoff"] == ["US2B2"]
    assert info["cpc_subclasses"] == {"H04N": 1}
    assert [c.pub_num for c in out] == ["US9A"]


def test_loop_removes_post_cutoff_seeds_from_pool(monkeypatch):
    state = {"summary": "s", "date_cutoff": "20110202",
             "checklist": [{"id": "e1", "criterion": "gaze estimation for video conferencing"}]}

    async def fake_facets(els, summary):
        return {"e1": {"thing": ["gaze"], "place": ["conferenc"], "apparatus": []}}
    monkeypatch.setattr("app.llm.facet_elements", fake_facets)

    async def fake_gp(query, num=20, page=0, before=None):
        assert before == "priority:20110202"
        L.gp.last_total[query] = 2
        return [_cand("US1", "Horizontal gaze estimation for video conferencing"), _cand("US2", "later gaze patent")], None
    monkeypatch.setattr(L.gp, "search", fake_gp)
    monkeypatch.setattr(L.gp, "is_blocked", lambda: False)

    async def fake_expand(seeds, known, max_cited=200, before=None):
        assert before == "20110202"
        return [], {"cpc_subclasses": {}, "seeds_after_cutoff": ["US2"]}
    monkeypatch.setattr(L, "expand", fake_expand)
    cands, stats = asyncio.run(L.run_loop(state, lambda: 0, lambda: False, lambda k, m, p=None: None))
    assert {c.pub_num for c in cands} == {"US1"}
    assert stats["rounds"][0]["seeds_after_cutoff"] == ["US2"] and stats["rounds"][0]["seed_pubs"] == ["US1"]


def test_mode_walk_never_repeats_a_query(monkeypatch):
    state = {"summary": "s", "checklist": [{"id": "e1", "criterion": "x"}]}

    async def fake_facets(els, summary):
        return {"e1": {"thing": ["videoconference"], "place": ["hub"], "apparatus": []}}
    monkeypatch.setattr("app.llm.facet_elements", fake_facets)
    seen = []

    async def fake_gp(query, num=20, page=0, before=None):
        seen.append(query)
        L.gp.last_total[query] = 0 if "hub" in query else 90_000
        return ([], None) if "hub" in query else ([_cand(f"US{len(seen)}", "t")] * 20, None)
    monkeypatch.setattr(L.gp, "search", fake_gp)
    monkeypatch.setattr(L.gp, "is_blocked", lambda: False)

    async def fake_expand(seeds, known, max_cited=200, before=None):
        return [], {"cpc_subclasses": {}}
    monkeypatch.setattr(L, "expand", fake_expand)
    _, stats = asyncio.run(L.run_loop(state, lambda: 0, lambda: False, lambda k, m, p=None: None))
    r1 = [q["query"] for q in stats["rounds"][0]["queries"]]
    assert len(r1) == len(set(r1))


def test_expand_light_uses_narrow_lookup_beyond_the_head(monkeypatch):
    from patent_analyzer.agentic import expand as E
    monkeypatch.setattr(E, "FULL_META_HEAD", 2)
    heavy, light = [], []

    async def fake_meta(pubs, with_claims=False):
        heavy.append(list(pubs))
        return {p: {"priority_date": "2000-01-01", "family_id": f"f{p}", "title": f"t{p}", "abstract": "abs", "cpc_codes": []} for p in pubs}

    async def fake_light(pubs):
        light.append(list(pubs))
        return {p: {"priority_date": "2000-01-01", "family_id": f"f{p}", "title": f"t{p}"} for p in pubs}

    async def fake_cits(pubs):
        return {"S1": {"cits": [{"cited": f"C{i}", "category": "SEA" if i < 2 else "APP"} for i in range(5)]}}
    async def fake_fwd(pubs):
        return {"S1": [{"publication_number": "C4", "category": "PRS"}, {"publication_number": "C9", "category": "PRS"}]}
    monkeypatch.setattr(E, "fetch_by_pub_nums", fake_meta)
    monkeypatch.setattr(E, "fetch_meta_light", fake_light)
    monkeypatch.setattr(E, "fetch_citations", fake_cits)
    monkeypatch.setattr(E, "fetch_cited_by", fake_fwd)
    out, info = asyncio.run(E.expand(["S1"], set(), max_cited=2000, before="20110101", light=True, forward=True))
    assert info["forward_total"] == 2 and "C9" in info["cited_by_seed"]["S1"]
    # ordering: C4 is cited (1) + citing (1) = 2 like the SEA ones → head is C0/C1 (SEA=2, first seen) then C4
    assert light[0] == ["S1"] and heavy[0] == ["C0", "C1"] and sorted(light[1]) == ["C2", "C3", "C4", "C9"]
    assert info["cited_by_seed"] == {"S1": ["C0", "C1", "C2", "C3", "C4", "C9"]}
    assert len(out) == 6 and info["cited_light"] == 4
    assert {c.pub_num: bool(c.abstract) for c in out} == {"C0": True, "C1": True, "C2": False, "C3": False, "C4": False, "C9": False}


def test_wide_mode_queries_every_candidate_and_expands_light(monkeypatch):
    monkeypatch.setattr(L, "LOOP_MODE", "wide")
    state = {"summary": "s", "date_cutoff": "20110202", "extraction": {"candidate_inventions": [
        {"id": "inv1", "level": "core", "elements": [
            {"id": "inv1.e1", "text": "a soluble adenylyl cyclase (sAC) inhibitor", "facets": {"named": ["soluble adenylyl cyclase", "sac"], "thing": ["inhibition"], "place": ["prostate cancer"]}}]},
        {"id": "inv2", "level": "application", "elements": [
            {"id": "inv2.e1", "text": "diagnosis", "facets": {"named": [], "thing": ["staining"], "place": ["tissue"]}}]}]}}

    async def fake_facets(els, summary):
        return {e["id"]: {"named": [], "thing": [], "place": [], "apparatus": []} for e in els}
    monkeypatch.setattr("app.llm.facet_elements", fake_facets)
    calls = []

    async def fake_gp(query, num=20, page=0, before=None):
        calls.append((query, num, before))
        L.gp.last_total[query] = 500
        return [_cand(f"US{len(calls)}", f"hit {len(calls)}"), _cand("US2099", "post-cutoff")], None
    monkeypatch.setattr(L.gp, "search", fake_gp)
    monkeypatch.setattr(L.gp, "is_blocked", lambda: False)
    seen = {}

    async def fake_expand(seeds, known, max_cited=200, before=None, light=False, forward=False):
        if "max_cited" not in seen:
            seen.update(seeds=list(seeds), max_cited=max_cited, before=before, light=light)
        c7 = _cand("US7", "cited art")
        c7.raw["bigquery"] = {"cpc_codes": ["A61K31/00", "A61K38/00"]}
        return [c7], {"cpc_subclasses": {"A61K": 2}, "seeds_after_cutoff": ["US2099"], "cited_total": 40, "cited_light": 10}
    monkeypatch.setattr(L, "expand", fake_expand)

    async def fake_similar(seeds, known, before=None, **kw):
        return [_cand("US8", "google neighbour")], {"similar_total": 5, "by_seed": {"US1": ["US8"]}}
    monkeypatch.setattr(L, "similar_neighbours", fake_similar)

    async def fake_neigh(title, cands, cutoff=None, doi="", arxiv_id="", summary="", embed=None):
        paper = Candidate(title="a neighbourhood paper", match_type="Paper", raw={"neigh": {"oa_id": "W1", "source": "references"}})
        return [paper], {"n": 1, "with_oa_id": 1}
    monkeypatch.setattr(L, "paper_neighbourhood", fake_neigh)

    async def fake_bridge(oa_ids):
        return {"W1": [{"patent_pub": "US5000A", "reftype": "exm"}]}
    monkeypatch.setattr("patent_analyzer.recall.bigquery_patents.fetch_citing_patents", fake_bridge, raising=False)
    events = []
    cands, stats = asyncio.run(L.run_loop(state, lambda: 0, lambda: False, lambda k, m, p=None: events.append(k)))
    assert "US8" in {c.pub_num for c in cands} and stats["rounds"][0]["similar_added"] == 1 and stats["rounds"][0]["similar_pubs"] == ["US8"]
    r0 = stats["rounds"][0]
    assert r0["neighbourhood_papers"] == 1 and r0["bridge"] == {"oa_ids": 1, "patents": 1, "error": None} and r0["bridge_pubs"] == ["US5000A"]
    assert "US5000A" in seen["seeds"] and "a neighbourhood paper" in {c.title for c in cands}
    assert [c[1] for c in calls] == [100] * len(calls) and all(c[2] == "priority:20110202" for c in calls)
    kinds = [q["kind"] for q in stats["rounds"][0]["queries"]]
    assert kinds[0] == "wide" and "named+domain" in kinds and "cpc+thing" in kinds   # one element → no natural (needs ≥3 words); CPC round after expansion
    nq = [q for q in stats["rounds"][0]["queries"] if q["kind"] == "named+domain"][0]
    assert nq["facets_used"]["named"] == ["soluble adenylyl cyclase"]   # 'sac' fails validation (not capitalised in the source)
    assert seen["light"] and seen["max_cited"] == L.MAX_CITED_LIGHT and seen["before"] == "20110202"
    pubs = {c.pub_num for c in cands}
    assert "US7" in pubs and "US2099" not in pubs and stats["mode"] == "wide"
    q0 = stats["rounds"][0]["queries"][0]
    assert q0["n"] == 1 and len(q0["pubs"]) == 2 and q0["new"] == 2 and q0["kind"] == "wide"
    assert stats["rounds"][0]["queries"][1]["new"] == 1   # US2099 repeats, one fresh hit
    assert "US7" in stats["rounds"][0]["expanded_pubs"] and stats["rounds"][0]["cpc_top"] == ["A61K31", "A61K38"]
    assert [c["id"] for c in stats["candidates"]] == ["inv1", "inv2"] and events == ["round_done"]


def test_similar_neighbours_ranks_shared_neighbours_and_filters_dates(monkeypatch):
    from patent_analyzer.agentic import expand as E

    async def fake_sim(pubs):
        return {"S1": ["S1", "N1", "N2"], "S2": ["S2", "N1", "N3"]}

    async def fake_light(pubs):
        return {p: {"priority_date": "2015-01-01" if p == "N3" else "2000-01-01", "family_id": "f", "title": f"t{p}"} for p in pubs}
    monkeypatch.setattr("patent_analyzer.recall.bigquery_patents.fetch_similar", fake_sim)
    monkeypatch.setattr(E, "fetch_meta_light", fake_light)
    out, info = asyncio.run(E.similar_neighbours(["S1", "S2"], set(), before="20110101"))
    assert [c.pub_num for c in out][0] == "N1" and "N3" not in {c.pub_num for c in out}
    assert info["similar_total"] == 3 and info["by_seed"] == {"S1": ["N1", "N2"], "S2": ["N1"]}
