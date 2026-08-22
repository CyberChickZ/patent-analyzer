import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.react_search import compose, run_react
from patent_analyzer.recall.pool import Candidate

ELS = [{"id": "inv1.e0", "text": "A videoconferencing system", "facets": {"thing": ["video conferencing"]}},
       {"id": "inv1.e1", "text": "a motorized turntable", "facets": {"thing": ["motorized turntable", "rotation stage"]}},
       {"id": "inv1.e2", "text": "a gaze controller", "facets": {"thing": ["gaze direction controller"]}}]


def test_compose_keeps_the_fixed_shape_with_cpc_last():
    assert compose(["swivel monitor", "turntable"], ["video conferencing", "telepresence"], "H04N7/15") == \
        "((swivel monitor) OR turntable) ((video conferencing) OR telepresence) CPC=H04N7/low"
    assert compose([], ["video conferencing"], None) == "(video conferencing)".replace("(video conferencing)", "(video conferencing)")


def test_react_loop_learns_terms_marks_coverage_and_stops(monkeypatch):
    calls = {"n": 0}
    searched = []

    async def fake_call(system, user, response_schema=None, thinking_budget=0, model=None):
        calls["n"] += 1
        assert "HISTORY" in user and "inv1.e1" in user
        if calls["n"] == 1:
            return json.dumps({"observation": "start", "decision": "turntable first", "covered_elements": [], "learned_terms": [],
                               "next": {"target_elements": ["inv1.e1"], "specific": ["motorized turntable", "rotation stage"],
                                        "broad": ["video conferencing", "telepresence"], "cpc_group": "H04N7"}, "stop": False})
        if calls["n"] == 2:
            assert "Teleconferencing robot with swiveling video monitor" in user     # it sees the titles
            return json.dumps({"observation": "titles say swiveling monitor", "decision": "use learned words on e2",
                               "covered_elements": ["inv1.e1"], "learned_terms": ["swiveling monitor", "teleconferencing robot"],
                               "next": {"target_elements": ["inv1.e2"], "specific": ["gaze direction controller", "swiveling monitor"],
                                        "broad": ["video conferencing", "teleconferencing robot"], "cpc_group": ""}, "stop": False})
        return json.dumps({"observation": "done", "decision": "all covered", "covered_elements": ["inv1.e2"], "learned_terms": [],
                           "next": {"target_elements": [], "specific": [], "broad": [], "cpc_group": ""}, "stop": True})

    async def fake_search(q):
        searched.append(q)
        return [Candidate(title="Teleconferencing robot with swiveling video monitor", pub_num="US7123285B2", match_type="Patent", year="1997")], 42, "serpapi_patents"

    steps = asyncio.run(run_react(ELS, ["video conferencing"], ["H04N7"], fake_search, lambda: 10, call=fake_call))
    assert searched[0] == "((motorized turntable) OR (rotation stage)) ((video conferencing) OR telepresence) CPC=H04N7/low"
    assert searched[1].startswith("((gaze direction controller) OR (swiveling monitor)) ((video conferencing) OR (teleconferencing robot))")
    assert [s["kind"] for s in steps] == ["react", "react", "react:stop"]
    assert steps[0]["new"] == 1 and steps[1]["new"] == 0 and steps[0]["observation"] == "start" and steps[1]["facets_used"]["specific"][1] == "swiveling monitor"


def test_react_stops_when_budget_is_zero():
    async def fake_call(*a, **k):
        raise AssertionError("must not be called")

    async def fake_search(q):
        return [], 0, "none"
    assert asyncio.run(run_react(ELS, ["x"], [], fake_search, lambda: 0, call=fake_call)) == []
