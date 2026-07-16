"""Phase 2: SSR — Structured Search Recipe.

Innovation landscape analysis → technology choices → patent type classification
→ checklist generation → checklist review → search query planning.

In P1 this becomes a subgraph with self-loop harness on checklist generation.
For P0 it's a single linear node.
"""

import json
from pathlib import Path

from state import GraphState


async def ssr_node(state: GraphState) -> dict:
    """Phase 2: generate checklist + search queries from invention summary."""
    from app.llm import (
        determine_patent_types,
        expand_technology_choices,
        generate_checklist_for_type,
        generate_search_queries,
        review_checklist,
        scan_innovation_landscape,
    )

    summary = state["summary"]
    fields_map = state.get("fields_map", [])
    cpc_subclass = state.get("cpc_subclass", "")
    personas = state.get("personas", {})
    _pdf = state.get("input_local_path")
    if _pdf and (not Path(_pdf).exists() or not _pdf.endswith(".pdf")):
        _pdf = None

    events = []

    def _event(kind: str, message: str):
        from datetime import datetime, timezone
        events.append({"ts": datetime.now(timezone.utc).isoformat(),
                        "phase": "phase2", "kind": kind, "message": message})

    _event("start", "Analyzing innovation landscape")

    # Load CPC context
    cpc_ctx = ""
    cpc_path = Path(__file__).parent.parent / "app" / "cpc_reference.json"
    if cpc_path.exists() and cpc_subclass:
        try:
            data = json.loads(cpc_path.read_text())
            entry = data.get(cpc_subclass, {})
            cpc_ctx = json.dumps(entry, indent=2) if entry else ""
        except Exception:
            pass

    # Step 1: Innovation axes
    innovation_axes = await scan_innovation_landscape(
        summary, fields_map, cpc_subclass, cpc_ctx,
        source_pdf_path=_pdf, persona=personas.get("landscape"))
    _event("info", f"{len(innovation_axes)} innovation axes identified")

    if not innovation_axes:
        return {
            "phase": "phase2",
            "innovation_axes": [],
            "checklist": [],
            "delegation": {"groups": []},
            "events": events,
            "phase_results": {"phase2": {"status": "completed", "data": {"axes": 0}}},
        }

    # Step 2: Technology choices (parallel per axis)
    import asyncio
    tech_tasks = [
        expand_technology_choices(ax, summary, source_pdf_path=_pdf,
                                  persona=personas.get("technology"))
        for ax in innovation_axes
    ]
    technology_choices = list(await asyncio.gather(*tech_tasks))
    _event("info", f"{len(technology_choices)} technology choices expanded")

    # Step 3: Patent types
    patent_types = await determine_patent_types(
        summary, technology_choices, source_pdf_path=_pdf)
    _event("info", f"Patent types: {patent_types}")

    # Step 4: Checklist per type
    cl_tasks = [
        generate_checklist_for_type(pt, summary, technology_choices,
                                     source_pdf_path=_pdf,
                                     persona=personas.get("checklist"))
        for pt in patent_types
    ]
    checklists_per_type = list(await asyncio.gather(*cl_tasks))
    combined = []
    for cl in checklists_per_type:
        combined.extend(cl)
    _event("info", f"Generated {len(combined)} checklist items")

    # Step 5: Review checklist
    checklist = await review_checklist(
        combined, summary, technology_choices,
        source_pdf_path=_pdf, persona=personas.get("reviewer"))
    _event("info", f"Reviewed checklist: {len(checklist)} items")

    # Step 6: Search queries
    delegation = await generate_search_queries(
        checklist, summary, cpc_subclass, persona=personas.get("plan"))
    groups = delegation.get("groups", [])
    _event("info", f"Planned {len(groups)} search groups")

    # Build all queries (deterministic)
    from patent_analyzer.query_builder import build_all_queries
    delegation = build_all_queries(delegation)

    return {
        "phase": "phase2",
        "innovation_axes": innovation_axes,
        "technology_choices": technology_choices,
        "patent_types": patent_types,
        "checklist": checklist,
        "delegation": delegation,
        "events": events,
        "phase_results": {"phase2": {
            "status": "completed",
            "data": {"checklist_count": len(checklist), "groups": len(groups)},
        }},
    }
