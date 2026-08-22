import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.wide import broad_terms, candidate_queries, cpc_queries, specific_terms, wide_queries


def _cand(cid, named, things, apps=None, patent=None):
    return {"id": cid, "cpc_pred": ["H04N7/15"], "elements": [
        {"id": f"{cid}.e0", "facets": {"named": [], "thing": ["video conferencing", "telepresence"], "place": ["meeting room"], "patent": ["teleconferencing apparatus"]}},
        {"id": f"{cid}.e1", "facets": {"named": named, "thing": things, "place": ["radiotherapy"], "apparatus": apps or [], "patent": patent or []}},
        {"id": f"{cid}.e2", "facets": {"named": [], "thing": ["motorized turntable", "rotation stage"], "patent": ["movable display"]}}]}


def test_v6_query_is_specific_group_and_broad_group():
    qs = candidate_queries(_cand("inv1", ["indocyanine green"], ["swivel display", "rotating monitor"]), neigh_terms=["media space", "social presence"])
    kinds = [q["kind"] for q in qs]
    assert kinds == ["element:inv1.e1", "element:inv1.e2", "candidate", "candidate+cpc"]
    e1 = qs[0]
    assert e1["query"].startswith('(("indocyanine green" OR (swivel display) OR (rotating monitor))'.replace('"', "") .replace("(indocyanine green)", "(indocyanine green)")) or e1["query"].startswith("((indocyanine green) OR (swivel display) OR (rotating monitor))")
    assert e1["facets_used"]["specific"] == ["indocyanine green", "swivel display", "rotating monitor"]
    assert e1["facets_used"]["broad"][:3] == ["video conferencing", "telepresence", "teleconferencing apparatus"] and "media space" in e1["facets_used"]["broad"]
    assert "radiotherapy" not in e1["query"]                      # place never scopes
    assert qs[3]["query"].endswith(" CPC=H04N7/low") and qs[3]["query"][:-len(" CPC=H04N7/low")] == qs[2]["query"]
    assert set(qs[2]["elements"]) == {"inv1.e1", "inv1.e2"}     # named element first, then the longest items


def test_broad_and_specific_helpers():
    c = _cand("inv1", [], ["a b"])
    assert broad_terms(c, ["neigh one", "neigh two"])[:2] == ["video conferencing", "telepresence"]
    assert len(broad_terms(c, [f"n{i}" for i in range(20)])) <= 15
    assert specific_terms({"facets": {"named": ["x"], "thing": ["t1", "t2", "t3", "t4"]}}) == ["x", "t1", "t2", "t3"]
    assert candidate_queries({"id": "x", "elements": []}) == []


def test_wide_queries_core_first_then_other_candidates_and_caps():
    cands = [_cand("inv1", [], ["a"]), _cand("inv2", ["sac"], ["c"]), _cand("inv3", ["momp"], ["e"])]
    qs = wide_queries(cands, max_total=6)
    assert [(q["candidate"], q["kind"]) for q in qs] == [
        ("inv1", "element:inv1.e1"), ("inv1", "element:inv1.e2"), ("inv1", "candidate"), ("inv1", "candidate+cpc"), ("inv2", "candidate"), ("inv3", "candidate")]
    assert len({q["query"] for q in qs}) == 6


def test_cpc_queries_use_top_groups_with_core_things():
    qs = cpc_queries([{"id": "inv1", "elements": [{"id": "e0", "facets": {"thing": ["optical tracking", "marker tracking"]}}]}], ["A61B", "A61B5/11", "A61N5", "G06T7"])
    assert [q["query"] for q in qs] == ["((optical tracking) OR (marker tracking)) CPC=A61B5/low", "((optical tracking) OR (marker tracking)) CPC=A61N5/low"]
    assert qs[0]["kind"] == "cpc+thing" and qs[0]["facets_used"]["cpc"] == ["A61B5"]
    assert cpc_queries([], ["A61B"]) == []


def test_title_terms_prefers_repeated_bigrams():
    from patent_analyzer.agentic.wide import terms_query, title_terms
    titles = ["Embodied social proxy: mediating interpersonal connection", "MeBot: a robotic platform for socially embodied telepresence",
              "Telepresence robot design for remote collaboration", "Social telepresence robot with a swiveling display", "the of and"]
    terms = title_terms(titles, top=4)
    assert terms[0] == "telepresence robot" and "embodied" in " ".join(terms)
    q = terms_query("inv1", terms)
    assert q["kind"] == "neigh_terms" and q["query"].startswith("((telepresence robot)")
    assert terms_query("inv1", []) is None
