import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.wide import candidate_queries, cpc_queries, wide_queries


def _cand(cid, named, things, apps=None, patent=None):
    return {"id": cid, "elements": [
        {"id": f"{cid}.e0", "facets": {"named": [], "thing": ["video conferencing", "telepresence"], "place": ["meeting room"], "apparatus": []}},
        {"id": f"{cid}.e1", "facets": {"named": named, "thing": things, "place": ["radiotherapy"], "apparatus": apps or [], "patent": patent or []}}]}


def test_candidate_queries_are_narrow_per_element_with_the_domain():
    qs = candidate_queries(_cand("inv1", ["indocyanine green", "icg"], ["swivel display", "rotating monitor"], ["turntable"]))
    kinds = [q["kind"] for q in qs]
    assert kinds == ["natural", "wide", "element:inv1.e1", "named+domain"]
    assert qs[2]["query"] == "((swivel display) OR (rotating monitor)) ((video conferencing) OR telepresence)"
    assert qs[2]["facets_used"] == {"thing": ["swivel display", "rotating monitor"], "domain": ["video conferencing", "telepresence"]}
    scoped = candidate_queries(_cand("inv1", [], ["swivel display"]), cpc_groups=["H04N7/15"])
    assert [q for q in scoped if q["kind"] == "element:inv1.e1"][0]["query"] == "(swivel display) CPC=H04N7/low"
    assert qs[1]["kind"] == "wide" and "radiotherapy" in qs[1]["query"] and "radiotherapy" not in qs[2]["query"]
    assert qs[3]["query"].startswith('("indocyanine green" OR icg) ((video conferencing) OR telepresence)')
    assert qs[0]["query"] == "video conferencing swivel display" and len(qs[0]["query"].split()) <= 10
    assert candidate_queries({"id": "x", "elements": []}) == []


def test_wide_queries_core_first_then_other_naturals_and_caps():
    cands = [_cand("inv1", [], ["a"], ["b"]), _cand("inv2", ["sac"], ["c"]), _cand("inv3", ["momp"], ["e"])]
    qs = wide_queries(cands, max_total=6)
    assert [(q["candidate"], q["kind"]) for q in qs] == [
        ("inv1", "natural"), ("inv1", "wide"), ("inv1", "element:inv1.e1"), ("inv2", "natural"), ("inv2", "wide"), ("inv3", "natural")]
    assert len({q["query"] for q in qs}) == 6


def test_queries_record_facets_used_and_source_elements():
    cand = {"id": "inv1", "elements": [
        {"id": "inv1.e0", "facets": {"named": ["icg"], "thing": ["perfusion map"], "place": []}},
        {"id": "inv1.e1", "facets": {"named": [], "thing": ["tissue perfusion"], "place": ["hindlimb"]}}]}
    qs = {q["kind"]: q for q in candidate_queries(cand)}
    assert qs["wide"]["facets_used"] == {"thing": ["perfusion map", "tissue perfusion"], "place": ["hindlimb"]} and qs["wide"]["elements"] == ["inv1.e0", "inv1.e1"]
    assert qs["element:inv1.e1"]["facets_used"] == {"thing": ["tissue perfusion"], "domain": ["perfusion map"]}
    assert qs["named+domain"]["facets_used"] == {"named": ["icg"], "domain": ["perfusion map"]}


def test_cpc_queries_use_top_subclasses_with_core_things():
    qs = cpc_queries([{"id": "inv1", "elements": [{"id": "e0", "facets": {"thing": ["optical tracking", "marker tracking"]}}]}], ["A61B", "A61B5/11", "A61N5", "G06T7"])
    assert [q["query"] for q in qs] == ["((optical tracking) OR (marker tracking)) CPC=A61B5/low", "((optical tracking) OR (marker tracking)) CPC=A61N5/low"]
    assert qs[0]["kind"] == "cpc+thing" and qs[0]["facets_used"]["cpc"] == ["A61B5"]
    assert cpc_queries([], ["A61B"]) == []


def test_natural_query_uses_patent_phrasings_when_present():
    cand = {"id": "inv1", "elements": [
        {"id": "inv1.e0", "facets": {"thing": ["telepresence robot"], "patent": ["teleconferencing apparatus"]}},
        {"id": "inv1.e1", "facets": {"thing": ["rotation"], "patent": ["swiveling video monitor"]}}]}
    qs = candidate_queries(cand)
    assert qs[0]["kind"] == "natural" and qs[0]["query"] == "teleconferencing apparatus swiveling video monitor telepresence robot"


def test_title_terms_prefers_repeated_bigrams():
    from patent_analyzer.agentic.wide import terms_query, title_terms
    titles = ["Embodied social proxy: mediating interpersonal connection", "MeBot: a robotic platform for socially embodied telepresence",
              "Telepresence robot design for remote collaboration", "Social telepresence robot with a swiveling display", "the of and"]
    terms = title_terms(titles, top=4)
    assert terms[0] == "telepresence robot" and "embodied" in " ".join(terms)
    q = terms_query("inv1", terms)
    assert q["kind"] == "neigh_terms" and q["query"].startswith("((telepresence robot)")
    assert terms_query("inv1", []) is None


def test_element_queries_rotate_the_predicted_groups():
    cand = {"id": "inv1", "cpc_pred": ["A61N5/10", "A61B6/04", "G01S17/00"], "elements": [
        {"id": "inv1.e0", "facets": {"thing": ["radiotherapy positioning"]}},
        {"id": "inv1.e1", "facets": {"thing": ["optical marker"]}},
        {"id": "inv1.e2", "facets": {"thing": ["fiducial calibration"]}},
        {"id": "inv1.e3", "facets": {"thing": ["motion tracking"]}},
        {"id": "inv1.e4", "facets": {"thing": ["beam gating"]}}]}
    qs = [q for q in candidate_queries(cand) if q["kind"].startswith("element:")]
    assert [q["facets_used"]["cpc"] for q in qs] == [["A61N5"], ["A61B6"], ["G01S17"], ["A61N5"]]
    assert qs[0]["query"] == "(optical marker) CPC=A61N5/low" and qs[1]["query"] == "(fiducial calibration) CPC=A61B6/low"
