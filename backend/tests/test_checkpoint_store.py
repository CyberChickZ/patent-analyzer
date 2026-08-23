"""Pause in one process, resume from the persisted checkpoint in another
(fresh saver instance = new Cloud Run instance)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.types import Command

from graph.main_graph import build_graph
from patent_analyzer.checkpoint_store import make_checkpointer
from tests.test_graph_structure import _fake_nodes


def test_pause_persist_new_process_resume(tmp_path):
    cfg = {"configurable": {"thread_id": "job-42"}}
    calls1 = []
    g1 = build_graph(checkpointer=make_checkpointer("file", tmp_path), nodes=_fake_nodes(calls1))
    out = asyncio.run(g1.ainvoke({"events": [], "pause_after": ["extract"]}, cfg))
    assert "__interrupt__" in out and calls1 == ["idca", "ssr"]
    assert (tmp_path / "job-42.json").exists()
    del g1  # "process dies"

    calls2 = []
    g2 = build_graph(checkpointer=make_checkpointer("file", tmp_path), nodes=_fake_nodes(calls2))
    snap = g2.get_state(cfg)
    assert snap.next == ("gate_extract",) and snap.values["extraction"]["candidate_inventions"][0]["id"] == "inv1"
    ext = snap.values["extraction"]
    ext["candidate_inventions"][0]["elements"][0]["text"] = "A method, restored"
    out = asyncio.run(g2.ainvoke(Command(resume={"type": "edit", "args": {"extraction": ext}}), cfg))
    assert calls2 == ["search", "evaluate", "draft", "report"]        # nothing before the gate re-ran
    assert out["overall_summary"] == "elements=A method, restored*|draft=x"
    hist = list(g2.get_state_history(cfg))
    assert any(s.next == ("ssr",) for s in hist)              # replay point for "rerun this phase" survives too


def test_memory_backend_is_default():
    from langgraph.checkpoint.memory import InMemorySaver
    assert isinstance(make_checkpointer(), InMemorySaver)
