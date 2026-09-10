"""Eval Subgraph: Phase 4 deep evaluation with Send API (Map-Reduce).

Fan-out: one eval_single_doc node per candidate document.
Reduce: merge all results, compute scores, generate summary.
"""

import asyncio
import operator
import os
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.constants import Send
from langgraph.graph import END, StateGraph

# How many of the delivered candidates are fanned out for a deep read. The gap
# between this and what Phase 3 delivered is not a detail: at the defaults the
# search delivers 60 (RERANK_LIMIT) or 120 (M1_DELIVER) and only this many are
# read. It is reported per job rather than left to be inferred — see
# `read_gap` below and report_sections.read_coverage_*.
MAX_EVAL = int(os.environ.get("EVAL_MAX_DOCS", "25"))

# `source` values that mean no text reached a model, or text reached one and
# nothing came back. Either way the reference is unchecked, not cleared.
NOT_READ_SOURCES = {"no_content", "abstract_failed", "abstract_noparse"}

# The Send fan-out starts every document at once: MAX_EVAL documents means
# MAX_EVAL concurrent Vertex calls, each with a PDF or a full claims text
# attached. LLM_RPM smooths the arrival rate but bounds nothing in flight.
_EVAL_SEM = asyncio.Semaphore(int(os.environ.get("EVAL_CONCURRENCY", "4")))


class EvalState(TypedDict, total=False):
    # Input from parent
    summary: str
    checklist: list[dict]
    ranked_candidates: list[dict]
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


def _event(kind: str, message: str) -> dict:
    from datetime import datetime, timezone
    return {"ts": datetime.now(timezone.utc).isoformat(),
            "phase": "phase4", "kind": kind, "message": message}


def fan_out_docs(state: EvalState) -> list[Send]:
    """Dynamically create one eval task per candidate document."""
    candidates = state.get("ranked_candidates", [])
    checklist = state.get("checklist", [])
    summary = state.get("summary", "")
    input_path = state.get("input_local_path", "")
    source_title = state.get("source_title", "")

    _pdf = input_path if input_path and Path(input_path).exists() and input_path.endswith(".pdf") else None

    # Ensure patents get evaluated even if ranked lower than papers
    candidates = state.get("ranked_candidates", [])
    patents_first = sorted(candidates, key=lambda d: (0 if d.get("match_type") == "Patent" or d.get("pub_num", "").startswith("US-") else 1))
    sends = []
    for doc in patents_first[:MAX_EVAL]:
        sends.append(Send("eval_single_doc", SingleDocInput(
            summary=summary,
            checklist=checklist,
            doc=doc,
            source_pdf_path=_pdf,
            source_title=source_title,
        )))
    return sends


def _is_patentish(doc: dict) -> bool:
    import re
    return (doc.get("match_type") == "Patent"
            or bool(re.match(r"^[A-Z]{2}[-\s]?\d", (doc.get("pub_num") or "").upper())))


async def eval_single_doc(input: SingleDocInput) -> dict:
    """Evaluate one document against the checklist. Returns partial state."""
    async with _EVAL_SEM:
        return await _eval_one(input)


async def _eval_one(input: SingleDocInput) -> dict:
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
        )
        # `source = "pdf"` used to be stamped on unconditionally, including on
        # the two paths evaluate_single_document takes when it read nothing (an
        # exception, or a reply it could not parse: both return an empty
        # checklist_results). Those rows then counted as full-PDF reads.
        if result.get("checklist_results"):
            result["source"] = "pdf"
        else:
            result["source"] = "no_content"
            result["no_content_reason"] = ("PDF downloaded but the read returned nothing"
                                           + (f": {str(result['error'])[:100]}" if result.get("error") else ""))
        full_text = _pdf_text(pdf)
        if full_text:
            result["quote_verification"] = verify_checklist_results(result.get("checklist_results", {}), full_text)
    else:
        # full text when we have it (claims from BigQuery + abstract), else abstract/snippet
        claims = (doc.get("claims_text") or "").strip()
        abstract = (doc.get("abstract") or "").strip() or (doc.get("snippet") or "").strip()
        # A paper whose PDF never arrived can still have its body text, fetched
        # from Europe PMC's open-access API in Phase 3 (patent_analyzer.fulltext).
        # That is the article, not a summary of it, so it is read as full text.
        oa_text = (doc.get("oa_full_text") or "").strip()
        if oa_text:
            text = f"[ABSTRACT] {abstract}\n\n[FULL TEXT]\n{oa_text}"
            mode = "full_text"
        elif claims:
            text = f"[ABSTRACT] {abstract}\n\n[CLAIMS]\n{claims}"
            mode = "full_text"
        else:
            text, mode = abstract, "abstract"
        if len(text) >= 120:
            result = await evaluate_single_document_text(
                input["summary"], input["checklist"], text, title, match_type,
                doc_mode=mode,
            )
            if mode == "full_text":
                result["quote_verification"] = verify_checklist_results(result.get("checklist_results", {}), text)
            result.setdefault("source", mode)
            result["text_chars"] = len(text)
            if not result.get("checklist_results"):
                # abstract_failed / abstract_noparse: text went in, nothing came back
                result["no_content_reason"] = f"the {mode} read returned nothing ({result.get('source')})"
        else:
            # Nothing was read. Say which of the two suppliers came up empty,
            # because the answers are different: no claims in our own copy
            # (non-US publication, or not in amie_patents) versus a prior-art
            # PDF download that failed.
            if _is_patentish(doc):
                why = ("no claims in amie_patents (non-US publication or not in our copy) "
                       "and no PDF" if not doc.get("local_pdf") else "PDF unreadable")
            else:
                why = "no PDF and no abstract" if not doc.get("local_pdf") else "PDF unreadable"
            result = {"title": title, "match_type": match_type,
                      "checklist_results": {}, "source": "no_content",
                      "no_content_reason": why, "text_chars": len(text)}

    result["pub_num"] = pub_num
    # Where this row's text came from, carried onto the row itself. The tiers
    # are resolved in Phase 3 and live on `ranked_candidates`, which does not
    # survive into results.json — so without this the finished report can say
    # "evaluated from the abstract" but not *why* there was nothing else, and
    # the missing-full-text list (patent_analyzer.fulltext_gap) would have had
    # to guess it from a title match.
    for k in ("fulltext_tier", "fulltext_detail", "fulltext_url", "fulltext_download",
              "landing_page", "doi", "arxiv_id"):
        if doc.get(k):
            result[k] = doc[k]
    result["had_bq_claims"] = bool((doc.get("claims_text") or "").strip())
    return {"eval_results": [result]}


def _read_gap(delivered: list[dict], fanned: list[dict], scoring_report: list[dict]) -> dict:
    """Delivered vs actually read, with the difference itemised.

    The report has been saying how many references were *evaluated* while a
    reference with nothing to read counted the same as one whose claims were
    read end to end. Three different numbers were being conflated:

      delivered   what Phase 3 handed over (60 at RERANK_LIMIT, 120 at M1_DELIVER)
      fanned_out  what Phase 4 sent to a model at all (EVAL_MAX_DOCS, 25)
      read        those a model saw text for (source != no_content)

    Every document in a gap is an *unchecked* reference, not a cleared one, so
    a determination of "no blocking reference" rests on `read`, not on
    `delivered`. This dict is what report_sections.read_coverage_* prints and
    what the caller asserts on.
    """
    by_source: dict[str, int] = {}
    for r in fanned:
        by_source[r.get("source") or "unknown"] = by_source.get(r.get("source") or "unknown", 0) + 1
    read = [r for r in scoring_report if (r.get("source") or "") not in NOT_READ_SOURCES]
    n_delivered, n_fanned, n_read = len(delivered), len(fanned), len(read)
    reasons: dict[str, int] = {}
    if n_delivered > n_fanned:
        reasons[f"never sent to a model (EVAL_MAX_DOCS={MAX_EVAL})"] = n_delivered - n_fanned
    dupes = len(fanned) - len(scoring_report)
    if dupes > 0:
        reasons["dropped as a duplicate of the source document"] = dupes
    for r in scoring_report:
        if (r.get("source") or "") in NOT_READ_SOURCES:
            why = r.get("no_content_reason") or "nothing to read"
            reasons[why] = reasons.get(why, 0) + 1
    shortfall = n_delivered - n_read
    headline = (f"Delivered {n_delivered} references, deep-read {n_read}"
                + (f" — {shortfall} were never read" if shortfall else " — all of them"))
    return {"delivered": n_delivered, "fanned_out": n_fanned, "read": n_read,
            "shortfall": shortfall, "reasons": reasons, "by_source": by_source,
            "headline": headline, "eval_cap": MAX_EVAL}


async def reduce_eval(state: EvalState) -> dict:
    """Merge all eval results, compute per-doc coverage scores, run the
    rule-based determination. The old LLM novelty summary / combination
    analysis are no longer generated (the report renders the rule verdict
    and, for §103, one explanation call made in the report node); their
    keys stay in the state as empty strings so results.json keeps its shape."""
    from patent_analyzer.scorer import classify_risk

    eval_results = state.get("eval_results", [])
    checklist = state.get("checklist", [])

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

    combination_analysis, overall_summary = "", ""   # legacy keys, no longer LLM-generated

    quote_stats = {"docs_verified": 0, "quotes": 0, "verified": 0, "downgraded": 0}
    for r in scoring_report:
        qv = r.get("quote_verification") or {}
        if qv:
            quote_stats["docs_verified"] += 1
            for k in ("quotes", "verified", "downgraded"):
                quote_stats[k] += int(qv.get(k, 0))

    # Rule-based prior-art determination over the verified coverage (novelty_score / risk_level kept in the state for
    # results.json compatibility; the report no longer shows them)
    # single_partial_103=0.7: a primary reference covering >=70% of the elements is flagged as §103-pattern risk
    # (PANORAMA App. C.5.3 rule a; H2 eval: macro-F1 .413 -> .450, blocking recall .29 -> .55 on examiner labels)
    from patent_analyzer.adjudicate import adjudicate
    adjudication = adjudicate(checklist, scoring_report, single_partial_103=0.7)
    from app.llm import _bri_enabled
    from patent_analyzer.report_sections import determination_label
    adjudication["construction"] = "bri" if _bri_enabled() else "plain"   # how the deep read read the criteria
    # the rendered sentence, stored once so the report, the API and the UI cannot each invent their
    # own wording — the §103 label is a screening flag and it has to read that way everywhere
    adjudication["label_text"] = determination_label(adjudication)

    read_gap = _read_gap(state.get("ranked_candidates") or [], eval_results, scoring_report)

    events = [
        _event("info", f"Evaluated {len(scoring_report)} docs, top score: {top_score:.2%}, risk: {risk_level}"),
        _event("warn" if read_gap["shortfall"] else "info", read_gap["headline"]),
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
        "eval_stats": {"quote_stats": quote_stats, "evaluated": len(scoring_report),
                       "read_gap": read_gap, "adjudication": adjudication},
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
