"""Phase 4: Deep evaluation of prior art candidates.

For P0 this is a single node wrapping evaluate_batch.
In P1 this becomes a subgraph with Send API (Map-Reduce per document).
"""

from datetime import datetime, timezone
from pathlib import Path

from state import GraphState


async def evaluate_node(state: GraphState) -> dict:
    """Phase 4: evaluate each candidate against the checklist."""
    from app.llm import (
        evaluate_batch,
        generate_combination_analysis,
        generate_overall_summary,
    )
    from patent_analyzer.scorer import classify_risk

    summary = state["summary"]
    checklist = state.get("checklist", [])
    ranked_candidates = state.get("ranked_candidates", [])
    personas = state.get("personas", {})
    source_title = state.get("source_title", "")
    input_path = state.get("input_local_path")
    _pdf = input_path if input_path and Path(input_path).exists() and input_path.endswith(".pdf") else None

    events = []

    def _event(kind: str, message: str, payload: dict | None = None):
        evt = {"ts": datetime.now(timezone.utc).isoformat(),
               "phase": "phase4", "kind": kind, "message": message}
        if payload:
            evt["payload"] = payload
        events.append(evt)

    _event("start", f"Evaluating {len(ranked_candidates)} candidates")

    if not ranked_candidates or not checklist:
        return {
            "phase": "phase4",
            "scoring_report": [],
            "overall_summary": "No candidates to evaluate.",
            "novelty_score": 1.0,
            "risk_level": "unknown",
            "events": events,
            "phase_results": {"phase4": {"status": "completed", "data": {"evaluated": 0}}},
        }

    # Cap deep eval candidates
    MAX_EVAL = 20
    eval_docs = ranked_candidates[:MAX_EVAL]
    _event("info", f"Deep-evaluating {len(eval_docs)} documents (max {MAX_EVAL})")

    scoring_report = await evaluate_batch(
        summary, checklist, eval_docs,
        max_concurrent=2,
        source_pdf_path=_pdf,
        source_title=source_title,
        persona=personas.get("evaluate"),
    )
    _event("info", f"Evaluated {len(scoring_report)} documents")

    # Filter source duplicates
    scoring_report = [r for r in scoring_report if not r.get("is_source_duplicate")]

    # Score and sort
    from app.llm import compute_ssr_grounding, compute_eval_grounding, compute_entropy_profile

    for doc in scoring_report:
        cr = doc.get("checklist_results", {})
        total_w, weighted_sum = 0.0, 0.0
        for ci in checklist:
            crit = ci.get("criterion", "")
            w = ci.get("weight", 1.0 / len(checklist))
            match = cr.get(crit, {})
            score = match.get("score")
            if score is None:
                score = 2 if match.get("match") else 0
            weighted_sum += w * (score / 2.0)
            total_w += w
        doc["ewss"] = round(weighted_sum / total_w, 4) if total_w > 0 else 0.0
        n_items = len(cr)
        n_matched = sum(1 for v in cr.values()
                        if isinstance(v, dict) and (v.get("score", 0) >= 2 or v.get("match")))
        doc["css"] = round(n_matched / n_items, 4) if n_items > 0 else 0.0
        doc["similarity_score"] = round(max(doc["ewss"], doc["css"]), 4)

    scoring_report.sort(key=lambda d: d.get("similarity_score", 0), reverse=True)
    top_score = scoring_report[0].get("similarity_score", 0) if scoring_report else 0
    risk_level = classify_risk(top_score)

    # Combination analysis (§103)
    combination_analysis = None
    if len(scoring_report) >= 2:
        try:
            combination_analysis = await generate_combination_analysis(
                summary, scoring_report[:5], persona=personas.get("summary"))
        except Exception:
            pass

    # Overall summary
    overall_summary = ""
    try:
        overall_summary = await generate_overall_summary(
            summary, scoring_report[:10], persona=personas.get("summary"))
    except Exception as e:
        overall_summary = f"(Summary generation failed: {e})"

    _event("info", f"Top score: {top_score:.2%}, risk: {risk_level}")

    return {
        "phase": "phase4",
        "scoring_report": scoring_report,
        "combination_analysis": combination_analysis or "",
        "overall_summary": overall_summary,
        "novelty_score": round(1.0 - top_score, 4),
        "risk_level": risk_level,
        "eval_results": scoring_report,
        "events": events,
        "phase_results": {"phase4": {
            "status": "completed",
            "data": {"evaluated": len(scoring_report), "top_score": round(top_score, 4)},
        }},
    }
