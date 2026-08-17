import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.wide import candidate_queries, cpc_queries, wide_queries


def _cand(cid, named, things, apps=None):
    return {"id": cid, "elements": [{"id": f"{cid}.e0", "facets": {"named": named, "thing": things, "place": ["radiotherapy"], "apparatus": apps or []}}]}


def test_candidate_queries_never_use_place_and_lead_with_thing():
    qs = candidate_queries(_cand("inv1", ["indocyanine green", "icg"], ["tissue perfusion imaging", "perfusion map"], ["ccd camera"]))
    assert [q["kind"] for q in qs] == ["thing", "named+thing", "thing+apparatus"]
    assert qs[0]["query"] == '((tissue perfusion imaging) OR (perfusion map))'
    assert qs[1]["query"] == '("indocyanine green" OR icg) ((tissue perfusion imaging) OR (perfusion map))'
    assert qs[2]["query"] == '((tissue perfusion imaging) OR (perfusion map)) (ccd camera)'
    assert all("radiotherapy" not in q["query"] for q in qs)
    assert candidate_queries(_cand("inv2", [], ["vault nanoparticle"]))[0]["kind"] == "thing"
    assert candidate_queries(_cand("inv3", ["x"], [])) == []


def test_wide_queries_core_first_then_other_things_and_caps():
    cands = [_cand("inv1", [], ["a"], ["b"]), _cand("inv2", ["sac"], ["c"]), _cand("inv3", ["momp"], ["e"])]
    qs = wide_queries(cands, max_total=5)
    assert [(q["candidate"], q["kind"]) for q in qs] == [
        ("inv1", "thing"), ("inv1", "thing+apparatus"), ("inv2", "thing"), ("inv3", "thing"), ("inv2", "named+thing")]
    assert len({q["query"] for q in qs}) == 5


def test_queries_record_facets_used_and_source_elements():
    cand = {"id": "inv1", "elements": [
        {"id": "inv1.e0", "facets": {"named": ["icg"], "thing": ["perfusion map"], "place": []}},
        {"id": "inv1.e1", "facets": {"named": [], "thing": ["tissue perfusion"], "place": ["hindlimb"]}}]}
    qs = candidate_queries(cand)
    assert qs[0]["facets_used"] == {"thing": ["perfusion map", "tissue perfusion"]} and qs[0]["elements"] == ["inv1.e0", "inv1.e1"]
    assert qs[1]["facets_used"] == {"named": ["icg"], "thing": ["perfusion map", "tissue perfusion"]}


def test_cpc_queries_use_top_subclasses_with_core_things():
    qs = cpc_queries([_cand("inv1", [], ["optical tracking", "marker tracking"])], ["A61B", "A61N", "G06T"])
    assert [q["query"] for q in qs] == ["CPC=A61B/low ((optical tracking) OR (marker tracking))", "CPC=A61N/low ((optical tracking) OR (marker tracking))"]
    assert qs[0]["kind"] == "cpc+thing" and qs[0]["facets_used"]["cpc"] == ["A61B"]
    assert cpc_queries([], ["A61B"]) == []


def test_natural_language_query_comes_first_when_the_candidate_has_a_concept():
    cand = {"id": "inv1", "concept": "A telepresence robot whose display swivels toward the participant a remote user selects.",
            "elements": [{"id": "inv1.e0", "facets": {"named": [], "thing": ["telepresence robot"], "place": ["meeting"], "apparatus": ["turntable"]}}]}
    qs = candidate_queries(cand)
    assert [q["kind"] for q in qs] == ["natural", "thing", "thing+apparatus"]
    assert qs[0]["query"].startswith("A telepresence robot whose display swivels") and qs[0]["elements"] == ["inv1.e0"]
