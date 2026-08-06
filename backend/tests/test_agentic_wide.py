import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.wide import candidate_queries, wide_queries


def _cand(cid, named, things, places):
    return {"id": cid, "elements": [{"id": f"{cid}.e0", "facets": {"named": named, "thing": things, "place": places}}]}


def test_candidate_queries_shapes():
    qs = candidate_queries(_cand("inv1", ["indocyanine green", "icg"], ["perfusion map", "tissue perfusion"], ["hindlimb"]))
    assert [q["kind"] for q in qs] == ["named", "named+thing", "thing+place"]
    assert qs[0]["query"] == '("indocyanine green" OR icg)'
    assert qs[1]["query"] == '("indocyanine green" OR icg) ((perfusion map) OR (tissue perfusion))'
    assert qs[2]["query"] == '((perfusion map) OR (tissue perfusion)) hindlimb'
    qs = candidate_queries(_cand("inv2", [], ["vault nanoparticle"], ["vaccine"]))
    assert [q["kind"] for q in qs] == ["thing+place", "thing"]
    assert candidate_queries(_cand("inv3", [], [], ["x"])) == []


def test_wide_queries_orders_core_first_then_other_named_and_caps():
    cands = [_cand("inv1", [], ["a"], ["b"]), _cand("inv2", ["sac"], ["c"], ["d"]), _cand("inv3", ["momp"], ["e"], [])]
    qs = wide_queries(cands, max_total=5)
    assert [(q["candidate"], q["kind"]) for q in qs] == [
        ("inv1", "thing+place"), ("inv1", "thing"), ("inv2", "named"), ("inv2", "named+thing"), ("inv3", "named")]
    assert len({q["query"] for q in qs}) == 5
