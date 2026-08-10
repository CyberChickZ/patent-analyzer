"""Test LangGraph pipeline structure and routing logic.

These tests verify the graph compiles correctly and routing decisions
work without making any LLM calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.main_graph import build_graph, route_after_idca, route_after_search


def test_graph_compiles():
    g = build_graph()
    assert g is not None
    assert type(g).__name__ == "CompiledStateGraph"


def test_route_absent():
    state = {"status_determination": "Absent", "doc_type": "invention"}
    assert route_after_idca(state) == "report"


def test_route_no_invention_doc():
    state = {"status_determination": "Present",
             "doc_type": "talks_about_invention_but_no_invention"}
    assert route_after_idca(state) == "report"


def test_route_present():
    state = {"status_determination": "Present", "doc_type": "invention"}
    assert route_after_idca(state) == "ssr"


def test_route_implied():
    state = {"status_determination": "Implied", "doc_type": "invention"}
    assert route_after_idca(state) == "ssr"


def test_route_search_failed():
    state = {"status": "failed_recall", "ranked_candidates": []}
    assert route_after_search(state) == "report"


def test_route_search_empty():
    state = {"status": "running", "ranked_candidates": []}
    assert route_after_search(state) == "report"


def test_route_search_has_results():
    state = {"status": "running", "ranked_candidates": [{"title": "x"}]}
    assert route_after_search(state) == "evaluate"


if __name__ == "__main__":
    test_graph_compiles()
    test_route_absent()
    test_route_no_invention_doc()
    test_route_present()
    test_route_implied()
    test_route_search_failed()
    test_route_search_empty()
    test_route_search_has_results()
    print("All graph structure tests passed!")


def test_extraction_subgraph_compiles_inside_main_graph(monkeypatch):
    monkeypatch.delenv("EXTRACTOR", raising=False)
    assert build_graph() is not None
    assert set(build_graph().get_graph().nodes) >= {"idca", "gate_idca", "ssr", "gate_extract", "search", "gate_search",
                                                     "evaluate", "gate_evaluate", "report"}


def _fake_nodes(calls):
    async def idca(state):
        calls.append("idca")
        return {"status_determination": "Present", "doc_type": "invention", "summary": "S", "events": []}

    async def ssr(state):
        calls.append("ssr")
        return {"extraction": {"candidate_inventions": [{"id": "inv1", "elements": [{"id": "inv1.e0", "text": "A method"}]}]},
                "checklist": [{"id": "c1", "criterion": "x"}]}

    async def search(state):
        calls.append("search")
        return {"ranked_candidates": [{"pub_num": "US1", "title": "t"}], "search_stats": {}}

    async def evaluate(state):
        calls.append("evaluate")
        return {"scoring_report": [{"pub_num": "US1", "title": "t", "checklist_results": {"c1": {"score": 2}}}]}

    async def report(state):
        calls.append("report")
        return {"overall_summary": "elements=" + ";".join(
            e.get("text", "") + ("*" if e.get("edited_by_user") else "")
            for c in state["extraction"]["candidate_inventions"] for e in c["elements"])}
    return {"idca": idca, "ssr": ssr, "search": search, "evaluate": evaluate, "report": report}


def test_default_run_never_interrupts():
    import asyncio
    calls = []
    g = build_graph(nodes=_fake_nodes(calls))
    out = asyncio.run(g.ainvoke({"events": [], "pause_after": []}))
    assert "__interrupt__" not in out and calls == ["idca", "ssr", "search", "evaluate", "report"]
    assert out["overall_summary"] == "elements=A method"


def test_pause_after_extract_interrupts_edit_resumes_and_report_sees_the_edit():
    import asyncio
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    calls = []
    g = build_graph(checkpointer=MemorySaver(), nodes=_fake_nodes(calls))
    cfg = {"configurable": {"thread_id": "t1"}}
    out = asyncio.run(g.ainvoke({"events": [], "pause_after": ["extract"]}, cfg))
    assert "__interrupt__" in out and calls == ["idca", "ssr"]
    hi = out["__interrupt__"][0].value
    assert hi["action_request"]["action"] == "review_extract"
    ext = hi["action_request"]["args"]["extraction"]
    ext["candidate_inventions"][0]["elements"][0]["text"] = "A method, reviewed"
    out = asyncio.run(g.ainvoke(Command(resume={"type": "edit", "args": {"extraction": ext}}), cfg))
    assert calls == ["idca", "ssr", "search", "evaluate", "report"]
    assert out["overall_summary"] == "elements=A method, reviewed*"
    assert out["user_edits"][0]["id"] == "inv1.e0" and out["user_edits"][0]["op"] == "edit"
    assert any(e["kind"] == "user_edit" for e in out["events"])



def test_entry_mid_pipeline_runs_only_the_tail():
    import asyncio
    calls = []
    g = build_graph(nodes=_fake_nodes(calls), entry="search")
    out = asyncio.run(g.ainvoke({"events": [], "pause_after": [],
                                 "extraction": {"candidate_inventions": [{"id": "inv1", "elements": [{"id": "inv1.e0", "text": "kept", "edited_by_user": True}]}]}}))
    assert calls == ["search", "evaluate", "report"] and out["overall_summary"] == "elements=kept*"


def test_rerun_phase_replays_from_the_checkpoint_before_it_and_pauses_again():
    import asyncio
    from langgraph.checkpoint.memory import MemorySaver
    calls = []
    g = build_graph(checkpointer=MemorySaver(), nodes=_fake_nodes(calls))
    cfg = {"configurable": {"thread_id": "t2"}}
    asyncio.run(g.ainvoke({"events": [], "pause_after": ["extract"]}, cfg))
    before = next(s for s in g.get_state_history(cfg) if s.next == ("ssr",))
    cid = before.config["configurable"]["checkpoint_id"]
    out = asyncio.run(g.ainvoke(None, {"configurable": {"thread_id": "t2", "checkpoint_id": cid}}))
    assert calls == ["idca", "ssr", "ssr"] and "__interrupt__" in out
