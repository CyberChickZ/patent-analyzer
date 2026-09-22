"""Events must be visible while the job that produced them is still running."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import event_log


@pytest.fixture(autouse=True)
def local(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUD_STATE", "local")
    monkeypatch.setenv("CLOUD_STATE_LOCAL_DIR", str(tmp_path))


def _e(ts, msg, phase="phase1", kind="info"):
    return {"ts": ts, "phase": phase, "kind": kind, "message": msg}


def test_an_event_is_readable_the_moment_it_is_raised():
    """The whole point. Before this, an event raised inside a node reached the
    job record only when the node RETURNED — a node is a phase, so job
    ea70d51a had four events that existed for 45 seconds and could not be
    read."""
    assert event_log.read("j1") == []
    event_log.append("j1", _e("2026-09-22T18:47:17Z", "Reading PDF"))
    got = event_log.read("j1")
    assert len(got) == 1 and got[0]["message"] == "Reading PDF"


def test_the_same_event_twice_is_one_event():
    """The live sink records it, then the node returns a patch carrying it
    again. Both are right; neither may be counted twice."""
    e = _e("2026-09-22T18:47:17Z", "Reading PDF")
    event_log.append("j1", e)
    event_log.append("j1", dict(e))
    assert len(event_log.read("j1")) == 1


def test_the_reader_answers_with_the_union_oldest_first():
    """The record is authoritative once a node returned; the log is the only
    source while it is running. Neither is a superset at every moment."""
    rec = [_e("2026-09-22T18:47:17Z", "a"), _e("2026-09-22T18:47:20Z", "b")]
    log = [_e("2026-09-22T18:47:20Z", "b"), _e("2026-09-22T18:48:05Z", "c")]
    out = event_log.merge(rec, log)
    assert [x["message"] for x in out] == ["a", "b", "c"]
    assert event_log.merge(None, None) == []
    assert [x["message"] for x in event_log.merge(None, log)] == ["b", "c"]


def test_the_sink_is_how_a_node_reaches_the_log_and_it_never_raises():
    seen = []
    tok = event_log.set_sink(seen.append)
    try:
        event_log.emit(_e("t", "one"))
        assert len(seen) == 1
    finally:
        event_log.reset_sink(tok)
    event_log.emit(_e("t", "two"))
    assert len(seen) == 1, "no sink set: the event goes nowhere and nothing breaks"

    def boom(_):
        raise RuntimeError("watcher exploded")
    tok = event_log.set_sink(boom)
    try:
        event_log.emit(_e("t", "three"))          # must not raise
    finally:
        event_log.reset_sink(tok)


def test_the_log_is_bounded():
    """An unbounded object in GCS is a bill nobody chose."""
    monkey = event_log.MAX_EVENTS
    try:
        event_log.MAX_EVENTS = 3
        for i in range(10):
            event_log.append("j2", _e(f"2026-09-22T18:47:{i:02d}Z", f"m{i}"))
        assert len(event_log.read("j2")) == 3
    finally:
        event_log.MAX_EVENTS = monkey


def test_every_node_hands_its_events_to_the_sink():
    """A node that forgets is a phase that goes dark again."""
    import re
    root = Path(__file__).parent.parent
    for rel in ("nodes/idca.py", "nodes/search.py", "nodes/draft.py", "nodes/report.py",
                "graph/eval_subgraph.py", "graph/extraction_subgraph.py"):
        src = (root / rel).read_text()
        assert re.search(r"event_log\.emit\(", src), f"{rel} raises events nobody can see"


def test_an_event_with_no_phase_inherits_the_running_node_s():
    """A model call does not know which node it is inside, and passing a phase
    down through every call site is six files of plumbing for one string."""
    seen = []
    tok = event_log.set_sink(seen.append)
    try:
        event_log.emit(_e("t1", "node started", phase="phase3"))
        event_log.emit({"ts": "t2", "phase": "", "kind": "llm_start", "message": "calling x"})
        assert seen[-1]["phase"] == "phase3"
    finally:
        event_log.reset_sink(tok)
