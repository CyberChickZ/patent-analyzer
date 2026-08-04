"""Eval Subgraph: Phase 4 deep evaluation with Send API (Map-Reduce).

Fan-out: one eval_single_doc node per candidate document.
Reduce: merge all results, compute scores, generate summary.
"""

import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, StateGraph


class EvalState(TypedDict, total=False):
    # Input from parent
    summary: str
    checklist: list[dict]
    ranked_candidates: list[dict]
    personas: dict[str, str]
    input_local_path: str
    source_title: str

    # Map-Reduce
    eval_results: Annotated[list[dict], operator.add]

    # Output
    scoring_report: list[dict]
    combination_analysis: str
    overall_summary: str
    novelty_score: float
    risk_level: str
    adjudication: dict
    eval_stats: dict
    events: list[dict]
    phase_results: dict


class SingleDocInput(TypedDict):
    summary: str
    checklist: list[dict]
    doc: dict
    source_pdf_path: str | None
    source_title: str
    persona: str | None


def _event(kind: str, message: str) -> dict:
    from datetime import datetime, timezone
    return {"ts": datetime.now(timezone.utc).isoformat(),
            "phase": "phase4", "kind": kind, "message": message}


def fan_out_docs(state: EvalState) -> list[Send]:
    """Dynamically create one eval task per candidate document."""
    candidates = state.get("ranked_candidates", [])
    checklist = state.get("checklist", [])
    summary = state.get("summary", "")
    personas = state.get("personas", {})
    input_path = state.get("input_local_path", "")
    source_title = state.get("source_title", "")

    _pdf = input_path if input_path and Path(input_path).exists() and input_path.endswith(".pdf") else None

    # Ensure patents get evaluated even if ranked lower than papers
    candidates = state.get("ranked_candidates", [])
    patents_first = sorted(candidates, key=lambda d: (0 if d.get("match_type") == "Patent" or d.get("pub_num", "").startswith("US-") else 1))
    MAX_EVAL = 25
    sends = []
    for doc in patents_first[:MAX_EVAL]:
        sends.append(Send("eval_single_doc", SingleDocInput(
            summary=summary,
            checklist=checklist,
            doc=doc,
            source_pdf_path=_pdf,
            source_title=source_title,
            persona=personas.get("evaluate"),
        )))
    return sends


async def eval_single_doc(input: SingleDocInput) -> dict:
    """Evaluate one document against the checklist. Returns partial state."""
    from app.llm import evaluate_single_document, evaluate_single_document_text

    doc = input["doc"]
    pdf = doc.get("local_pdf", "")
    title = doc.get("title", "")
    match_type = doc.get("match_type", "Paper")
    pub_num = doc.get("pub_num", "")

    from patent_analyzer.quote_verify import pdf_text as _pdf_text, verify_checklist_results

    if pdf and Path(pdf).exists():
        result = await evaluate_single_document(
            input["summary"], input["checklist"], pdf, title, match_type,
            source_pdf_path=input["source_pdf_path"],
            source_title=input["source_title"],
            persona=input["persona"],
        )
        result["source"] = "pdf"
        full_text = _pdf_text(pdf)
        if full_text:
            result["quote_verification"] = verify_checklist_results(result.get("checklist_results", {}), full_text)
    else:
        # full text when we have it (claims from BigQuery + abstract), else abstract/snippet
        claims = (doc.get("claims_text") or "").strip()
        abstract = (doc.get("abstract") or "").strip() or (doc.get("snippet") or "").strip()
        if claims:
            text = f"[ABSTRACT] {abstract}\n\n[CLAIMS]\n{claims}"
            mode = "full_text"
        else:
            text, mode = abstract, "abstract"
        if len(text) >= 120:
            result = await evaluate_single_document_text(
                input["summary"], input["checklist"], text, title, match_type,
                persona=input["persona"], doc_mode=mode,
            )
            if mode == "full_text":
                result["quote_verification"] = verify_checklist_results(result.get("checklist_results", {}), text)
        else:
            result = {"title": title, "match_type": match_type,
                      "checklist_results": {}, "source": "no_content"}

    result["pub_num"] = pub_num
    return {"eval_results": [result]}


async def reduce_eval(state: EvalState) -> dict:
    """Merge all eval results, compute scores, generate summary."""
    from app.llm import generate_combination_analysis, generate_overall_summary
    from patent_analyzer.scorer import classify_risk

    eval_results = state.get("eval_results", [])
    checklist = state.get("checklist", [])
    summary = state.get("summary", "")
    personas = state.get("personas", {})

    # Filter source duplicates
    scoring_report = [r for r in eval_results if not r.get("is_source_duplicate")]

    # Compute scores
    for doc in scoring_report:
        cr = doc.get("checklist_results", {})
        total_w, weighted_sum = 0.0, 0.0
        for ci in checklist:
            crit = ci.get("criterion", "")
            w = ci.get("weight", 1.0 / max(len(checklist), 1))
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

    # §103 combination analysis
    combination_analysis = ""
    if len(scoring_report) >= 2:
        try:
            combination_analysis = await generate_combination_analysis(
                summary, scoring_report[:5], persona=personas.get("summary")) or ""
        except Exception:
            pass

    # Overall summary
    overall_summary = ""
    if scoring_report:
        try:
            overall_summary = await generate_overall_summary(
                summary, scoring_report[:10], persona=personas.get("summary"))
        except Exception as e:
            overall_summary = f"(Summary generation failed: {e})"

    quote_stats = {"docs_verified": 0, "quotes": 0, "verified": 0, "downgraded": 0}
    for r in scoring_report:
        qv = r.get("quote_verification") or {}
        if qv:
            quote_stats["docs_verified"] += 1
            for k in ("quotes", "verified", "downgraded"):
                quote_stats[k] += int(qv.get(k, 0))

    # Rule-based prior-art determination over the verified coverage (novelty_score / risk_level untouched)
    # single_partial_103=0.7: a primary reference covering >=70% of the elements is flagged as §103-pattern risk
    # (PANORAMA App. C.5.3 rule a; H2 eval: macro-F1 .413 -> .450, blocking recall .29 -> .55 on examiner labels)
    from patent_analyzer.adjudicate import adjudicate
    adjudication = adjudicate(checklist, scoring_report, single_partial_103=0.7)

    events = [
        _event("info", f"Evaluated {len(scoring_report)} docs, top score: {top_score:.2%}, risk: {risk_level}"),
        _event("info", f"Quote verification: {quote_stats['verified']}/{quote_stats['quotes']} quotes verified "
                       f"across {quote_stats['docs_verified']} docs, {quote_stats['downgraded']} criteria downgraded"),
        _event("info", f"Determination: {adjudication['risk']} ({adjudication['label']}) — {adjudication['reason']}"),
    ]

    return {
        "scoring_report": scoring_report,
        "combination_analysis": combination_analysis,
        "overall_summary": overall_summary,
        "novelty_score": round(1.0 - top_score, 4),
        "risk_level": risk_level,
        "adjudication": adjudication,
        # GraphState has no `adjudication` channel yet (state.py untouched here); eval_stats carries it to the report
        "eval_stats": {"quote_stats": quote_stats, "evaluated": len(scoring_report), "adjudication": adjudication},
        "events": events,
        "phase_results": {"phase4": {
            "status": "completed",
            "data": {"evaluated": len(scoring_report), "top_score": round(top_score, 4), **quote_stats},
        }},
    }


def build_eval_subgraph():
    g = StateGraph(EvalState)

    g.add_node("eval_single_doc", eval_single_doc)
    g.add_node("reduce_eval", reduce_eval)

    g.add_conditional_edges("__start__", fan_out_docs, ["eval_single_doc"])
    g.add_edge("eval_single_doc", "reduce_eval")
    g.add_edge("reduce_eval", END)

    return g.compile()
