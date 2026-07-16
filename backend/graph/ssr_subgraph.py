"""SSR Subgraph: Structured Search Recipe generation.

Nodes: landscape → technology → checklist → self_check → review → queries
Self-loop: checklist → self_check → (retry checklist or proceed to review)
"""

import asyncio
import json
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph


class SSRState(TypedDict, total=False):
    # Input from parent
    summary: str
    fields_map: list[str]
    cpc_subclass: str
    personas: dict[str, str]
    input_local_path: str

    # Internal
    cpc_context: str
    innovation_axes: list[dict]
    technology_choices: list[dict]
    patent_types: list[str]
    raw_checklist: list[dict]
    checklist: list[dict]
    self_check_ok: bool
    retry_count: int
    feedback: dict | None
    delegation: dict

    # Output events
    events: list[dict]


def _event(phase: str, kind: str, message: str) -> dict:
    from datetime import datetime, timezone
    return {"ts": datetime.now(timezone.utc).isoformat(),
            "phase": phase, "kind": kind, "message": message}


async def landscape_node(state: SSRState) -> dict:
    from app.llm import scan_innovation_landscape

    summary = state["summary"]
    fields_map = state.get("fields_map", [])
    cpc_subclass = state.get("cpc_subclass", "")
    personas = state.get("personas", {})
    _pdf = state.get("input_local_path")
    if _pdf and (not Path(_pdf).exists() or not _pdf.endswith(".pdf")):
        _pdf = None

    cpc_ctx = ""
    cpc_path = Path(__file__).parent.parent / "app" / "cpc_reference.json"
    if cpc_path.exists() and cpc_subclass:
        try:
            data = json.loads(cpc_path.read_text())
            entry = data.get(cpc_subclass, {})
            cpc_ctx = json.dumps(entry, indent=2) if entry else ""
        except Exception:
            pass

    axes = await scan_innovation_landscape(
        summary, fields_map, cpc_subclass, cpc_ctx,
        source_pdf_path=_pdf, persona=personas.get("landscape"))

    return {
        "innovation_axes": axes,
        "cpc_context": cpc_ctx,
        "events": [_event("phase2", "info", f"{len(axes)} innovation axes identified")],
    }


async def technology_node(state: SSRState) -> dict:
    from app.llm import expand_technology_choices

    summary = state["summary"]
    axes = state.get("innovation_axes", [])
    personas = state.get("personas", {})
    _pdf = state.get("input_local_path")
    if _pdf and (not Path(_pdf).exists() or not _pdf.endswith(".pdf")):
        _pdf = None

    tasks = [
        expand_technology_choices(ax, summary, source_pdf_path=_pdf,
                                  persona=personas.get("technology"))
        for ax in axes
    ]
    choices = list(await asyncio.gather(*tasks))
    return {
        "technology_choices": choices,
        "events": [_event("phase2", "info", f"{len(choices)} technology choices expanded")],
    }


async def checklist_node(state: SSRState) -> dict:
    from app.llm import determine_patent_types, generate_checklist_for_type

    summary = state["summary"]
    choices = state.get("technology_choices", [])
    personas = state.get("personas", {})
    feedback = state.get("feedback")
    _pdf = state.get("input_local_path")
    if _pdf and (not Path(_pdf).exists() or not _pdf.endswith(".pdf")):
        _pdf = None

    patent_types = await determine_patent_types(
        summary, choices, source_pdf_path=_pdf)

    cl_tasks = [
        generate_checklist_for_type(pt, summary, choices,
                                     source_pdf_path=_pdf,
                                     persona=personas.get("checklist"))
        for pt in patent_types
    ]
    checklists = list(await asyncio.gather(*cl_tasks))
    combined = []
    for cl in checklists:
        combined.extend(cl)

    events = [_event("phase2", "info", f"Generated {len(combined)} checklist items for {patent_types}")]
    if feedback:
        events.append(_event("phase2", "retry_applied",
                             f"Regenerated checklist with self-check feedback"))

    return {
        "patent_types": patent_types,
        "raw_checklist": combined,
        "self_check_ok": False,
        "events": events,
    }


async def self_check_node(state: SSRState) -> dict:
    from app.llm import self_check

    raw_cl = state.get("raw_checklist", [])
    summary = state.get("summary", "")
    _pdf = state.get("input_local_path")
    if _pdf and (not Path(_pdf).exists() or not _pdf.endswith(".pdf")):
        _pdf = None

    cl_text = json.dumps(raw_cl, indent=2, ensure_ascii=False)[:6000]
    check = await self_check(
        "checklist generation", summary, cl_text, source_pdf_path=_pdf)

    ok = check.get("ok", True)
    retry_count = state.get("retry_count", 0)

    if ok:
        return {
            "self_check_ok": True,
            "checklist": raw_cl,
            "events": [_event("phase2", "self_check_pass", "Checklist passed self-check")],
        }
    else:
        issues = ", ".join(check.get("issues", []))
        return {
            "self_check_ok": False,
            "feedback": {
                "issues": check.get("issues", []),
                "suggestion": check.get("suggestion", ""),
                "previous_response": cl_text[:4000],
            },
            "retry_count": retry_count + 1,
            "events": [_event("phase2", "self_check_fail", f"Issues: {issues[:200]}")],
        }


def should_retry_checklist(state: SSRState) -> str:
    if state.get("self_check_ok", False):
        return "review_checklist"
    if state.get("retry_count", 0) >= 2:
        return "review_checklist"
    return "retry_gen_checklist"


async def review_node(state: SSRState) -> dict:
    from app.llm import review_checklist

    summary = state["summary"]
    choices = state.get("technology_choices", [])
    cl = state.get("checklist") or state.get("raw_checklist", [])
    personas = state.get("personas", {})
    _pdf = state.get("input_local_path")
    if _pdf and (not Path(_pdf).exists() or not _pdf.endswith(".pdf")):
        _pdf = None

    reviewed = await review_checklist(
        cl, summary, choices, source_pdf_path=_pdf,
        persona=personas.get("reviewer"))

    return {
        "checklist": reviewed,
        "events": [_event("phase2", "info", f"Reviewed checklist: {len(reviewed)} items")],
    }


async def queries_node(state: SSRState) -> dict:
    from app.llm import generate_search_queries
    from patent_analyzer.query_builder import build_all_queries

    summary = state["summary"]
    checklist = state.get("checklist", [])
    cpc_subclass = state.get("cpc_subclass", "")
    personas = state.get("personas", {})

    delegation = await generate_search_queries(
        checklist, summary, cpc_subclass, persona=personas.get("plan"))
    delegation = build_all_queries(delegation)
    groups = delegation.get("groups", [])

    return {
        "delegation": delegation,
        "events": [_event("phase2", "info", f"Planned {len(groups)} search groups")],
    }


def build_ssr_subgraph():
    g = StateGraph(SSRState)

    g.add_node("landscape", landscape_node)
    g.add_node("technology", technology_node)
    g.add_node("gen_checklist", checklist_node)
    g.add_node("self_check", self_check_node)
    g.add_node("retry_gen_checklist", checklist_node)
    g.add_node("review_checklist", review_node)
    g.add_node("gen_queries", queries_node)

    g.set_entry_point("landscape")
    g.add_edge("landscape", "technology")
    g.add_edge("technology", "gen_checklist")
    g.add_edge("gen_checklist", "self_check")

    g.add_conditional_edges("self_check", should_retry_checklist, {
        "retry_gen_checklist": "retry_gen_checklist",
        "review_checklist": "review_checklist",
    })
    g.add_edge("retry_gen_checklist", "self_check")

    g.add_edge("review_checklist", "gen_queries")
    g.add_edge("gen_queries", END)

    return g.compile()
