"""The running job's events, visible while it is still running.

Until now an event raised inside a node went into a Python list local to that
node and reached the job record only when the node RETURNED. A node is a
phase, so a 20-minute search was 20 minutes in which the job had produced
events and no caller could see one: Harry submitted ea70d51a at 18:47:16Z,
/status said `events: 0` at 18:48:02Z, and at 18:48:31Z returned four events
of which the first was stamped 18:47:17Z. They had existed for 45 seconds.

That is also why the first screen said "0 events / No events yet" while event
one was stamped a second after the job was created. One cause, two symptoms.

So events go to their own small object in GCS, appended as they happen, one
per job, written with a generation precondition — the same rule as every other
piece of cross-request state (leader_deploy.md §G.3). They are kept separate
from the job record on purpose: the record is ~160 KB and rewriting it per
event would be nine megabytes of writes a job, while the event list is ~12 KB.

`emit` is called from inside nodes through a ContextVar sink, so a node does
not have to know what a job record is. It never raises: losing the ability to
watch a job must not be able to fail the job.
"""

from __future__ import annotations

import contextvars
from typing import Callable

from . import cloud_state

#: Set by the runner for the duration of one pipeline run.
_sink: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar(
    "event_sink", default=None)

#: Above this the log stops growing. A job that produces more than this has a
#: problem the log is not going to help with, and an unbounded object in GCS is
#: a bill nobody chose.
MAX_EVENTS = 5000


def key(job_id: str) -> str:
    return f"events:{job_id}"


def set_sink(fn: Callable[[dict], None] | None) -> object:
    return _sink.set(fn)


def reset_sink(token: object) -> None:
    try:
        _sink.reset(token)  # type: ignore[arg-type]
    except Exception:
        _sink.set(None)


#: The phase of the last event that carried one. A model call does not know
#: which node it is inside, and the alternative — passing a phase down through
#: every call site — is six files of plumbing for one string.
_phase: contextvars.ContextVar[str] = contextvars.ContextVar("event_phase", default="")


def current_phase() -> str:
    return _phase.get()


def emit(evt: dict) -> None:
    """Hand one event to whatever is watching this run. Never raises."""
    if evt.get("phase"):
        try:
            _phase.set(evt["phase"])
        except Exception:
            pass
    elif not evt.get("phase"):
        evt["phase"] = _phase.get()
    fn = _sink.get()
    if fn is None:
        return
    try:
        fn(evt)
    except Exception:
        pass


def event_key(e: dict) -> tuple:
    """Same identity the UI dedupes on: an event is its timestamp, phase, kind
    and message. The live append and the node's returned patch carry the same
    event, and neither is wrong — they must simply not both be counted."""
    return (e.get("ts", ""), e.get("phase", ""), e.get("kind", ""), e.get("message", ""))


def append(job_id: str, evt: dict) -> None:
    """Add one event to the job's log in GCS. Never raises."""
    if not job_id or not evt:
        return

    def _f(doc: dict) -> None:
        rows = doc.setdefault("events", [])
        k = event_key(evt)
        if any(event_key(r) == k for r in rows[-40:]):
            return
        if len(rows) < MAX_EVENTS:
            rows.append(evt)
        doc["n"] = len(rows)
    try:
        cloud_state.update(key(job_id), _f)
    except Exception:
        pass


def read(job_id: str) -> list[dict]:
    """The log as stored. Empty when there is none — which is not the same as
    a job with no events, and callers merge rather than replace."""
    try:
        doc, _, _ = cloud_state.read(key(job_id))
        return list(doc.get("events") or [])
    except Exception:
        return []


def merge(record_events: list[dict] | None, logged: list[dict] | None) -> list[dict]:
    """Everything either source has, once each, oldest first.

    The job record is authoritative once a node has returned; the log is the
    only source while it is still running. Neither is a superset of the other
    at every moment, so the answer is the union.
    """
    out: list[dict] = []
    seen: set[tuple] = set()
    for e in list(record_events or []) + list(logged or []):
        k = event_key(e)
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    out.sort(key=lambda e: str(e.get("ts", "")))
    return out
