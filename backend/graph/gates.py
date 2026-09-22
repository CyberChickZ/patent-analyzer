"""Gate nodes: one after each phase. A gate does nothing unless its phase is
in state["pause_after"]; then it calls interrupt() with a HumanInterrupt
(agent-inbox shape) and, on resume, applies the HumanResponse.

The gate holds no LLM call: LangGraph re-runs the interrupting node from
its start on resume ("The node restarts from the beginning of the node
where the interrupt was called"), so the expensive work stays in the phase
node before it. Edits never touch reducer channels (search_results,
eval_results) — those would accumulate, not overwrite.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.types import interrupt

PHASES = ("idca", "extract", "search", "evaluate", "draft")

# what the reviewer may change after each phase (state keys; whole-value replace)
EDITABLE: dict[str, tuple[str, ...]] = {
    "idca": ("summary", "status_determination", "input_mode", "cpc_subclass"),
    "extract": ("extraction", "checklist"),
    "search": ("ranked_candidates",),
    "evaluate": ("scoring_report",),
    "draft": ("draft_claims",),
}
# what the reviewer sees (read-only context next to the editable values)
SHOWN: dict[str, tuple[str, ...]] = {
    "idca": ("source_title", "doc_type", "category"),
    "extract": ("summary",),
    "search": ("search_stats",),
    "evaluate": ("eval_stats",),
    "draft": ("adjudication",),
}


class ActionRequest(TypedDict):
    action: str
    args: dict


class HumanInterruptConfig(TypedDict):
    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str | None


class HumanResponse(TypedDict, total=False):
    type: Literal["accept", "ignore", "response", "edit"]
    args: Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_interrupt(phase: str, state: dict) -> HumanInterrupt:
    args = {k: state.get(k) for k in EDITABLE[phase] if k in state}
    context = {k: state.get(k) for k in SHOWN.get(phase, ()) if k in state}
    return {"action_request": {"action": f"review_{phase}", "args": {**args, "_context": context}},
            "config": {"allow_ignore": False, "allow_respond": True, "allow_edit": True, "allow_accept": True},
            "description": f"Phase '{phase}' finished. Review, edit, then continue."}


def _elements(extraction: dict | None) -> dict[str, dict]:
    out = {}
    for c in (extraction or {}).get("candidate_inventions") or []:
        for e in c.get("elements") or []:
            if e.get("id"):
                out[e["id"]] = e
    return out


def _diff_extraction(before: dict | None, after: dict | None, phase: str) -> list[dict]:
    """Element-level diff in the evals' id/text vocabulary."""
    b, a = _elements(before), _elements(after)
    ts = _now()
    out = []
    for eid, e in b.items():
        if eid not in a:
            out.append({"phase": phase, "kind": "element", "id": eid, "op": "delete", "before": {"text": e.get("text")}, "after": None, "ts": ts})
        elif e.get("text") != a[eid].get("text") or (e.get("facets") or {}) != (a[eid].get("facets") or {}):
            out.append({"phase": phase, "kind": "element", "id": eid, "op": "edit",
                        "before": {"text": e.get("text"), "facets": e.get("facets")},
                        "after": {"text": a[eid].get("text"), "facets": a[eid].get("facets")}, "ts": ts})
            a[eid]["edited_by_user"] = True
    for eid, e in a.items():
        if eid not in b:
            out.append({"phase": phase, "kind": "element", "id": eid, "op": "add", "before": None, "after": {"text": e.get("text")}, "ts": ts})
            e["edited_by_user"] = True
    bc = {c.get("id") for c in (before or {}).get("candidate_inventions") or []}
    ac = {c.get("id") for c in (after or {}).get("candidate_inventions") or []}
    for cid in sorted(bc - ac):
        out.append({"phase": phase, "kind": "candidate", "id": cid, "op": "delete", "before": None, "after": None, "ts": ts})
    return out


def _doc_key(d: dict) -> str:
    return (d.get("pub_num") or d.get("title") or "").strip().lower()


def _diff_docs(before: list[dict] | None, after: list[dict] | None, phase: str) -> list[dict]:
    """Removed documents (by pub_num / title) and score edits on scoring_report rows."""
    ts = _now()
    b = {_doc_key(d): d for d in before or []}
    a = {_doc_key(d): d for d in after or []}
    out = []
    for k, d in b.items():
        if k not in a:
            out.append({"phase": phase, "kind": "doc", "pub_num": d.get("pub_num", ""), "title": (d.get("title") or "")[:120],
                        "op": "remove", "before": None, "after": None, "ts": ts})
            continue
        bcr, acr = d.get("checklist_results") or {}, a[k].get("checklist_results") or {}
        for cid, item in acr.items():
            bs = (bcr.get(cid) or {}).get("score") if isinstance(bcr.get(cid), dict) else None
            as_ = item.get("score") if isinstance(item, dict) else None
            if cid in bcr and bs != as_:
                item["edited_by_user"] = True
                out.append({"phase": phase, "kind": "score", "pub_num": d.get("pub_num", ""), "criterion_id": cid,
                            "op": "edit", "before": bs, "after": as_, "ts": ts})
        if out and out[-1].get("pub_num") == d.get("pub_num") and out[-1]["kind"] == "score":
            a[k]["edited_by_user"] = True
    return out


def _limitations(draft: dict | None) -> dict[str, tuple[dict, dict]]:
    out = {}
    for c in (draft or {}).get("claims") or []:
        for l in c.get("limitations") or []:
            if l.get("lid"):
                out[l["lid"]] = (c, l)
    return out


def _diff_draft(before: dict | None, after: dict | None, phase: str) -> list[dict]:
    """Limitation-level diff of the draft claims (by lid): text edits, deleted
    and added limitations; edited items are marked so the report can say so."""
    b, a = _limitations(before), _limitations(after)
    ts = _now()
    out = []
    for lid, (_, l) in b.items():
        if lid not in a:
            out.append({"phase": phase, "kind": "limitation", "id": lid, "op": "delete", "before": {"text": l.get("text")}, "after": None, "ts": ts})
        elif (l.get("text") or "") != (a[lid][1].get("text") or ""):
            out.append({"phase": phase, "kind": "limitation", "id": lid, "op": "edit",
                        "before": {"text": l.get("text")}, "after": {"text": a[lid][1].get("text")}, "ts": ts})
            a[lid][1]["edited_by_user"] = True
            a[lid][0]["edited_by_user"] = True
    for lid, (c, l) in a.items():
        if lid not in b:
            out.append({"phase": phase, "kind": "limitation", "id": lid, "op": "add", "before": None, "after": {"text": l.get("text")}, "ts": ts})
            l["edited_by_user"] = True
            c["edited_by_user"] = True
    bn = {c.get("no") for c in (before or {}).get("claims") or []}
    an = {c.get("no") for c in (after or {}).get("claims") or []}
    for no in sorted(x for x in bn - an if x is not None):
        out.append({"phase": phase, "kind": "claim", "id": str(no), "op": "delete", "before": None, "after": None, "ts": ts})
    return out


def apply_response(phase: str, state: dict, resp: HumanResponse | None) -> dict:
    """State patch for a HumanResponse. accept/ignore/None → {}; edit → the
    whitelisted values replaced, edited items marked, user_edits appended,
    one `user_edit` event."""
    if not resp or resp.get("type") in (None, "accept", "ignore"):
        return {}
    if resp.get("type") == "response":
        note = {"phase": phase, "kind": "note", "op": "respond", "before": None, "after": str(resp.get("args") or "")[:2000], "ts": _now()}
        return {"user_edits": [note], "events": [{"ts": note["ts"], "phase": phase, "kind": "user_edit", "message": f"reviewer note on {phase}"}]}
    args = (resp.get("args") or {})
    if isinstance(args, dict) and "args" in args and "action" in args:   # full ActionRequest
        args = args["args"] or {}
    patch: dict = {}
    edits: list[dict] = []
    for key in EDITABLE[phase]:
        if key not in args:
            continue
        new = copy.deepcopy(args[key])
        old = state.get(key)
        if key == "extraction":
            edits += _diff_extraction(old, new, phase)
        elif key in ("ranked_candidates", "scoring_report"):
            edits += _diff_docs(old, new, phase)
        elif key == "draft_claims":
            edits += _diff_draft(old, new, phase)
        elif old != new:
            edits.append({"phase": phase, "kind": "field", "id": key, "op": "edit", "before": old if not isinstance(old, str) else old[:2000],
                          "after": new if not isinstance(new, str) else new[:2000], "ts": _now()})
        patch[key] = new
    if not edits:
        return {}
    patch["user_edits"] = edits
    patch["events"] = [{"ts": edits[-1]["ts"], "phase": phase, "kind": "user_edit",
                        "message": f"reviewer changed {len(edits)} item(s) after {phase}",
                        "payload": {"n": len(edits), "kinds": sorted({e['kind'] for e in edits})}}]
    return patch


def make_gate(phase: str):
    assert phase in PHASES

    async def gate(state: dict) -> dict:
        # Close the phase's books *before* the interrupt, so the reviewer's
        # thinking time is not billed to the phase. metering.mark is idempotent:
        # the gate body re-executes in full on resume, and the second pass must
        # not overwrite the real numbers with the ~0 it would measure.
        from patent_analyzer import metering
        metrics = {phase: metering.mark(phase)}
        if phase not in (state.get("pause_after") or []):
            return {"phase_metrics": metrics}
        resp = interrupt(build_interrupt(phase, state))
        patch = apply_response(phase, state, resp)
        patch["paused_at"] = ""
        patch["phase_metrics"] = metrics
        return patch

    gate.__name__ = f"gate_{phase}"
    return gate
