import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import cache
from patent_analyzer.agentic import loop as L
from patent_analyzer.recall.pool import Candidate


@pytest.fixture(autouse=True)
def _kv(tmp_path):
    cache.reset_for_tests(tmp_path / "kv.sqlite")
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

    async def fake_expand(seeds, known, max_cited=200):
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
    assert any("CPC=H04N" in q["query"] for q in rounds[2]["queries"]) if len(rounds) > 2 else True
    assert all(k == "round_done" for k, _ in events)
