import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.adjudicate import adjudicate, claim_chart
from patent_analyzer.draft.avoid import candidate_limitations, covered_sets, dependent_claims, plan

E = ["a first sensor", "a controller coupled to the sensor", "wherein the controller halts the motor", "a display showing the halt state"]
CL = [{"id": f"inv1.e{i + 1}", "criterion": e, "weight": 0.25} for i, e in enumerate(E)]
EXT = {"candidate_inventions": [
    {"id": "inv1", "level": "core", "dependent_hints": ["the sensor is a Hall-effect sensor", "the display is an OLED panel"],
     "elements": [{"id": "inv1.e0", "text": "A safety system"}] + [{"id": f"inv1.e{i + 1}", "text": e, "evidence_quote": "q"} for i, e in enumerate(E)]},
    {"id": "inv2", "level": "component", "elements": [
        {"id": "inv2.e0", "text": "A controller"},
        {"id": "inv2.e1", "text": "a watchdog timer resetting the controller", "evidence_quote": "watchdog timer", "kind": "structure"},
        {"id": "inv2.e2", "text": "no quote here", "evidence_quote": ""}]}]}


def _doc(pub, covered, partial=()):
    cr = {}
    for e in E:
        if e in covered:
            cr[e] = {"score": 2, "verified_quotes": ["some verbatim text here"]}
        elif e in partial:
            cr[e] = {"score": 1, "verified_quotes": ["partial verbatim text here"]}
        else:
            cr[e] = {"score": 0, "verified_quotes": []}
    return {"pub_num": pub, "title": pub, "checklist_results": cr}


def _adj_chart(docs):
    adj = adjudicate(CL, docs, single_partial_103=0.7)
    return adj, claim_chart(adj, CL, docs)


def test_pool_and_covered_sets():
    pool = candidate_limitations(EXT, EXT["candidate_inventions"][0])
    assert [p["pid"] for p in pool] == ["hint0", "hint1", "inv2.e1"]          # e2 has no quote -> not in the pool
    assert pool[0]["origin"] == "dependent_hint" and pool[2]["origin"] == "component_element" and pool[2]["evidence_quote"] == "watchdog timer"
    adj, chart = _adj_chart([_doc("A", E[:3]), _doc("B", E[2:])])
    cov = covered_sets(chart, CL)
    assert cov["A"] == {"inv1.e1", "inv1.e2", "inv1.e3"} and cov["B"] == {"inv1.e3", "inv1.e4"}


def test_plan_102_narrows_with_the_first_undisclosed_pool_limitation():
    pool = candidate_limitations(EXT, EXT["candidate_inventions"][0])
    adj, chart = _adj_chart([_doc("A", E), _doc("B", E[:2])])
    assert adj["label"] == "102"
    p = plan(adj, chart, CL, pool, {"hint0": {"A"}, "hint1": set(), "inv2.e1": {"B"}})
    assert p["strategy"] == "narrowed" and p["independent_add"] == "hint1"
    assert p["dependents"] == ["hint0", "inv2.e1"]                  # both disclosed by a charted doc -> pool order
    assert [t["chosen"] for t in p["candidates_tried"]] == [False, True, False]
    assert "A discloses every element" in p["reason"] and "dependent_hint[1]" in p["reason"]


def test_plan_103_combination_needs_a_limitation_undisclosed_by_every_charted_doc_else_unresolved():
    pool = candidate_limitations(EXT, EXT["candidate_inventions"][0])
    adj, chart = _adj_chart([_doc("A", E[:3]), _doc("B", E[2:])])
    assert adj["label"] == "103" and adj["basis"] == "combination"
    p = plan(adj, chart, CL, pool, {"hint0": {"A"}, "hint1": {"B"}, "inv2.e1": {"A", "B"}})
    assert p["strategy"] == "unresolved" and p["independent_add"] is None
    assert "needs input from the inventor" in p["reason"]
    pool[0]["dropped"] = "unsupported"
    p = plan(adj, chart, CL, pool, {"hint1": set(), "inv2.e1": {"A"}})
    assert p["strategy"] == "narrowed" and p["independent_add"] == "hint1"
    assert any(t.get("dropped") == "unsupported" for t in p["candidates_tried"])


def test_plan_primary_partial_keeps_independent_and_marks_distinguishing():
    pool = candidate_limitations(EXT, EXT["candidate_inventions"][0])
    adj, chart = _adj_chart([_doc("A", E[:3])])
    assert adj["label"] == "103" and adj["basis"] == "primary_partial"
    p = plan(adj, chart, CL, pool, {"hint0": {"A"}})
    assert p["strategy"] == "as_is" and p["independent_add"] is None
    assert p["uncovered_elements"] == ["inv1.e4"] and p["first_dependent"] == "hint1" and p["dependents"][0] == "hint1"


def test_plan_allow_and_no_prior_art():
    pool = candidate_limitations(EXT, EXT["candidate_inventions"][0])
    adj, chart = _adj_chart([_doc("A", E[:1])])
    p = plan(adj, chart, CL, pool, {})
    assert p["strategy"] == "as_is" and set(p["uncovered_elements"]) == {"inv1.e2", "inv1.e3", "inv1.e4"}
    p = plan(None, None, CL, pool, {}, has_prior_art=False)
    assert p["strategy"] == "no_prior_art" and p["dependents"] == ["hint0", "hint1", "inv2.e1"]


def test_dependent_claims_further_limitation_check():
    parent = {"limitations": [{"text": "a first sensor"}, {"text": "a controller coupled to the sensor"}]}
    items = [{"pid": "a", "text": "the sensor is a Hall-effect sensor"},
             {"pid": "b", "text": "a controller that is coupled to the sensor"},          # restates a parent limitation
             {"pid": "c", "text": "a relay instead of the controller"},                   # removes a parent element
             {"pid": "d", "text": "the sensor is a Hall effect sensor"}]                  # duplicate of a

    def sim(a, b):
        out = []
        for x, y in zip(a, b):
            out.append(0.95 if {x, y} == {"a controller that is coupled to the sensor", "a controller coupled to the sensor"}
                       or {x, y} == {"the sensor is a Hall-effect sensor", "the sensor is a Hall effect sensor"} else 0.3)
        return out
    kept, rejected = dependent_claims(items, parent, sim, max_n=8)
    assert [k["pid"] for k in kept] == ["a"]
    assert {(r["pid"], r["rejected"]) for r in rejected} == {("b", "not_further_limiting"), ("c", "not_further_limiting"), ("d", "duplicate")}
    kept, rejected = dependent_claims([{"pid": str(i), "text": f"limitation {i}"} for i in range(10)], parent, lambda a, b: [0.1] * len(a), max_n=3)
    assert len(kept) == 3 and all(r["rejected"] == "over_limit" for r in rejected)
