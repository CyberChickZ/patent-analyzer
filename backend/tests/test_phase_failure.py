"""A phase that raises must leave the job in a state the user can see and act on:
status `error` (never a job stuck on `running`), the message and the failing
phase visible, and rerunnable from that phase.

The LangGraph fact all of this rests on, verified on 0.2.76: when a node raises,
`astream` propagates, the thread stays parked on that node
(`get_state(cfg).next == (node,)`), and re-invoking with input None re-runs it.
"""

import asyncio
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph


class _S(TypedDict, total=False):
    x: int


def _graph(explode):
    def a(s):
        return {"x": 1}

    def search(s):
        if explode["on"]:
            raise RuntimeError("serpapi melted")
        return {"x": 2}

    def draft(s):
        return {"x": 3}

    g = StateGraph(_S)
    g.add_node("idca", a)
    g.add_node("search", search)
    g.add_node("draft", draft)
    g.set_entry_point("idca")
    g.add_edge("idca", "search")
    g.add_edge("search", "draft")
    g.add_edge("draft", END)
    return g.compile(checkpointer=MemorySaver())


def test_langgraph_parks_on_the_failed_node_and_reruns_it():
    """The contract _record_phase_failure and action=rerun_phase depend on."""
    explode = {"on": True}
    graph = _graph(explode)
    cfg = {"configurable": {"thread_id": "t"}}

    async def run(inp):
        async for _ in graph.astream(inp, config=cfg, stream_mode="updates"):
            pass

    try:
        asyncio.run(run({"x": 0}))
    except RuntimeError as e:
        assert "serpapi melted" in str(e)
    else:
        raise AssertionError("the node's exception must propagate")

    st = graph.get_state(cfg)
    assert st.next == ("search",)                  # parked on the node that raised
    assert st.values == {"x": 1}                   # the phase before it is preserved

    explode["on"] = False
    asyncio.run(run(None))                         # input None re-runs exactly that node
    assert graph.get_state(cfg).values == {"x": 3}


def test_record_phase_failure_marks_the_job(monkeypatch):
    import app.main as m
    saved = []
    monkeypatch.setattr(m, "_save_job", saved.append)

    explode = {"on": True}
    graph = _graph(explode)
    cfg = {"configurable": {"thread_id": "t2"}}

    async def run():
        async for _ in graph.astream({"x": 0}, config=cfg, stream_mode="updates"):
            pass

    job = {"id": "jf", "status": "running", "output_dir": "/tmp/x"}
    try:
        asyncio.run(run())
    except Exception as exc:
        m._record_phase_failure(job, graph, cfg, exc)

    assert job["status"] == "error", "a failed job must never stay on 'running'"
    assert "serpapi melted" in job["error"] and "RuntimeError" in job["error"]
    assert job["failed_node"] == "search" and job["failed_phase"] == "search"
    assert job["error_trace"] and "serpapi melted" in job["error_trace"]
    assert job["phase_checkpoints"]["search"], "the retry point must be recorded"
    assert saved and saved[0] is job


# ── the API surface ──

def _client(monkeypatch, job):
    import app.main as m
    monkeypatch.setattr(m, "_save_job", lambda j: None)
    monkeypatch.setattr(m, "_enqueue_job", lambda jid: job.__setitem__("_enqueued", True))
    m.jobs[job["id"]] = job
    return TestClient(m.app)


def _failed_job():
    return {"id": "jerr", "status": "error", "phase": "phase3", "output_dir": "/tmp/x",
            "error": "RuntimeError: serpapi melted", "failed_phase": "search", "failed_node": "search",
            "paused_at": "", "phase_checkpoints": {"search": "ck9"},
            "_hitl_saved_state": {"summary": "S", "ranked_candidates": [{"pub_num": "US1"}],
                                  "search_stats": {"total_unique": 12}}}


def test_failed_job_can_be_rerun_from_the_phase_that_broke(monkeypatch):
    job = _failed_job()
    c = _client(monkeypatch, job)
    r = c.post("/api/jobs/jerr/resume", json={"action": "rerun_phase"})
    assert r.status_code == 200, r.text
    assert r.json()["from_phase"] == "search" and r.json()["retried_node"] == "search"
    assert job["_retry_failed"] is True and job["status"] == "queued" and job.get("_enqueued")
    assert job["hitl_history"][-1]["after"] == "error"


def test_failed_job_rejects_continue(monkeypatch):
    job = _failed_job()
    c = _client(monkeypatch, job)
    r = c.post("/api/jobs/jerr/resume", json={"action": "continue"})
    assert r.status_code == 400 and "rerun_phase" in r.json()["detail"]


def test_running_job_cannot_be_resumed(monkeypatch):
    job = {"id": "jrun", "status": "running", "output_dir": "/tmp/x"}
    c = _client(monkeypatch, job)
    r = c.post("/api/jobs/jrun/resume", json={"action": "rerun_phase"})
    assert r.status_code == 400 and "running" in r.json()["detail"]


def test_state_endpoint_reports_the_failure(monkeypatch):
    c = _client(monkeypatch, _failed_job())
    r = c.get("/api/jobs/jerr/state").json()
    assert r["status"] == "error" and r["failed_phase"] == "search" and r["failed_node"] == "search"
    assert "serpapi melted" in r["error"]
    assert r["paused_at"] == "", "a failed job is not paused"
    # the failed phase's own editable values and context are still served, so the
    # reviewer can look at what the phase had produced before it died
    assert r["values"]["ranked_candidates"] == [{"pub_num": "US1"}]
    assert r["context"]["search_stats"] == {"total_unique": 12}


# ── heartbeat ──
#
# A phase is one graph node, so last_heartbeat only ticked on node boundaries:
# a 20-minute search node was 20 minutes of silence, and /status on any other
# instance (Cloud Run scales out, _active_pipelines is per-instance) would call
# a healthy job a zombie at the 900 s threshold.

def test_heartbeat_ticks_faster_than_the_zombie_threshold():
    import app.main as m
    assert m.HEARTBEAT_S < 900, "the heartbeat must beat well inside the staleness window"
    assert m.HEARTBEAT_S * 3 < 900


def test_pipeline_runs_a_side_channel_heartbeat():
    """The heartbeat is a task started next to the graph stream, not something
    driven by node updates — the whole point is that it ticks mid-node."""
    import inspect

    import app.main as m
    src = inspect.getsource(m._run_langgraph_pipeline)
    assert "async def _heartbeat():" in src
    assert "asyncio.create_task(_heartbeat())" in src
    # and it must be stopped on every exit, including the failure path
    assert "finally:" in src and "hb.cancel()" in src
    body = src[src.index("async def _heartbeat():"):]
    assert "job[\"last_heartbeat\"]" in body and "_save_job(job)" in body
