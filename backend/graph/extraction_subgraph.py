"""Extraction subgraph (line A): candidate inventions -> claim-language elements.

Nodes: candidates (A1) -> elements (A2) -> verify (A3, no LLM) -> self_check (A5)
Self-loop: self_check fails -> elements once more (retry_count <= 1).

Output patch: extraction (schema in outputs/leader_extraction.md §1), checklist
(derived from the core candidate's supported elements, {id, criterion, weight}
so Phase 3/4 keep working), errors, events.

Claim mode (input_mode == "claim_text" or the text starts with "1."): the
independent claims are split with nodes.claim_mode._parse_claim_limitations and
prefilled as element texts; A2 only adds evidence_quote / facets / kind.
"""

import operator
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

MAX_RETRY = 1
_CLAIM_START = re.compile(r"^\s*1\s*\.\s")
_CLAIM_SPLIT = re.compile(r"(?m)^\s*(?=\d+\s*\.\s)")
_DEPENDENT = re.compile(r"\b(?:of|according to|as claimed in|as recited in|as defined in|as in)\s+(?:any\s+(?:one\s+)?of\s+)?claims?\s+\d", re.I)
_INPUT_MODE_KIND = {
    "academic_paper": "paper", "technical_report": "paper", "informal_description": "disclosure",
    "patent_draft": "patent_draft", "claim_text": "patent_draft", "manuscript": "manuscript",
}


class ExtractionState(TypedDict, total=False):
    # input from parent
    summary: str
    document_text: str
    input_local_path: str
    input_mode: str
    doc_type: str
    doc_kind: str
    cpc_subclass: str

    # internal
    full_text: str
    claim_prefill: dict
    candidates: list[dict]
    no_invention_reason: str | None
    raw_extraction: dict
    feedback: dict | None
    retry_count: int
    self_check_ok: bool
    llm_calls: int

    # output
    extraction: dict
    checklist: list[dict]
    errors: dict
    events: Annotated[list[dict], operator.add]


def _event(kind: str, message: str) -> dict:
    return {"ts": datetime.now(timezone.utc).isoformat(), "phase": "phase2", "kind": kind, "message": message}


def resolve_doc_text(state: ExtractionState) -> str:
    """Full document text: the input file when present (state.document_text is
    truncated to 20k chars by idca_node), else state.document_text."""
    text = state.get("document_text") or ""
    path = state.get("input_local_path") or ""
    if path and Path(path).exists():
        try:
            if path.endswith(".pdf"):
                from app.llm import _EXTRACTION_DOC_CAP, _extract_pdf_text
                file_text = _extract_pdf_text(path, max_pages=80, max_chars=_EXTRACTION_DOC_CAP)
            else:
                file_text = Path(path).read_text(errors="ignore")
            if len(file_text) > len(text):
                text = file_text
        except Exception:
            pass
    return text


def doc_kind_of(state: ExtractionState) -> str:
    return state.get("doc_kind") or _INPUT_MODE_KIND.get(state.get("input_mode") or "", "paper")


def is_claim_mode(state: ExtractionState, text: str) -> bool:
    return state.get("input_mode") == "claim_text" or bool(_CLAIM_START.match(text or ""))


def independent_claims(text: str) -> list[str]:
    """Numbered claims that do not refer back to another claim."""
    parts = [p.strip() for p in _CLAIM_SPLIT.split(text or "") if p.strip()]
    return [p for p in parts if re.match(r"^\d+\s*\.\s", p) and not _DEPENDENT.search(p)]


def claim_prefill(text: str) -> tuple[list[dict], dict[str, list[str]]]:
    """Candidates + fixed element texts from the independent claims (regex split)."""
    from nodes.claim_mode import _parse_claim_limitations, _split_preamble
    cands, prefill = [], {}
    for i, claim in enumerate(independent_claims(text), 1):
        parsed = _parse_claim_limitations(claim)
        texts = _split_preamble(parsed["preamble"]) + parsed["limitations"]
        texts = [t for t in texts if len(t.strip()) > 3]
        if not texts:
            continue
        cid = f"inv{len(cands) + 1}"
        concept = re.sub(r"^\s*\d+\s*\.\s*", "", parsed["preamble"] or claim)
        concept = " ".join(concept.split()[:35]).rstrip(":,")
        cands.append({"id": cid, "concept": concept, "level": "core" if not cands else "component", "cpc_pred": []})
        prefill[cid] = texts
    return cands, prefill


async def candidates_node(state: ExtractionState) -> dict:
    """A1: 1-4 candidate inventions (one LLM call; none in claim mode)."""
    from app.llm import extract_candidates

    text = resolve_doc_text(state)
    kind = doc_kind_of(state)
    if is_claim_mode(state, text):
        cands, prefill = claim_prefill(text)
        if cands:
            return {"full_text": text, "doc_kind": "patent_draft", "candidates": cands, "claim_prefill": prefill,
                    "no_invention_reason": None, "retry_count": 0, "llm_calls": 0,
                    "events": [_event("info", f"Claim mode: {len(cands)} independent claims prefilled")]}
    res = await extract_candidates(text, state.get("summary", ""), kind)
    cands = res.get("candidate_inventions") or []
    reason = res.get("no_invention_reason")
    patch = {"full_text": text, "doc_kind": kind, "candidates": cands, "claim_prefill": {},
             "no_invention_reason": reason, "retry_count": 0, "llm_calls": 1,
             "events": [_event("info", f"{len(cands)} candidate inventions: "
                               + "; ".join(f'{c["id"]}[{c["level"]}]' for c in cands))]}
    if not cands:
        patch["extraction"] = {"doc_kind": kind, "candidate_inventions": [], "no_invention_reason": reason}
        patch["checklist"] = []
        patch["errors"] = {"n_elements": 0, "n_unsupported": 0, "quote_survival": None, "claim_ratio_min": None,
                           "llm_calls": 1}
        patch["events"].append(_event("no_invention", f"No candidate invention: {reason}"))
    return patch


def route_after_candidates(state: ExtractionState) -> str:
    return "elements" if state.get("candidates") else END


async def elements_node(state: ExtractionState) -> dict:
    """A2: claim drafts + elements for all candidates (one LLM call)."""
    from app.llm import extract_elements

    raw = await extract_elements(state["full_text"], state.get("candidates") or [],
                                 prefill=state.get("claim_prefill") or None, feedback=state.get("feedback"))
    n = sum(len(c.get("elements") or []) for c in raw.get("candidate_inventions") or [])
    events = [_event("info", f"Extracted {n} elements" + (f" ({raw['error']})" if raw.get("error") else ""))]
    if state.get("feedback"):
        events.append(_event("retry_applied", "Regenerated elements with self-check feedback"))
    return {"raw_extraction": raw, "llm_calls": state.get("llm_calls", 0) + 1, "events": events}


def _claim_ratio(elements: list[dict], method_claim: str) -> float | None:
    from patent_analyzer.quote_verify import normalize
    if not elements or not method_claim:
        return None
    joined = normalize(" ".join(e["text"] for e in elements))
    return round(SequenceMatcher(None, joined, normalize(method_claim), autojunk=False).ratio(), 4)


def verify_extraction(raw: dict, text: str, doc_kind: str, no_invention_reason=None) -> tuple[dict, list[dict], dict]:
    """A3 (pure): snap every quote, mark unsupported, measure element/claim
    agreement. Returns (extraction, checklist, errors)."""
    from evals.extraction_errors import snap_quote
    from patent_analyzer.adapters.paper import locate_marker

    has_markers = "[S" in text and re.search(r"\[S[\d.]+\.P\d+\]", text) is not None
    cands = []
    n_el = n_unsup = 0
    ratios = []
    for c in raw.get("candidate_inventions") or []:
        elements = []
        for e in c.get("elements") or []:
            e = dict(e)
            found, loc = snap_quote(e.get("evidence_quote") or "", text)
            if found:
                span = loc.get("char") if loc else None
                pos = locate_marker(text, span[0]) if (has_markers and span) else {"section": None, "para": None}
                e["evidence_loc"] = {"section": pos["section"], "para": pos["para"], "char": span,
                                     "method": loc.get("method"), "sim": loc.get("sim")}
                e["unsupported"] = False
            else:
                e["evidence_loc"] = None
                e["unsupported"] = True
                n_unsup += 1
            n_el += 1
            elements.append(e)
        form = c.get("primary_form") or "method"
        ratio = _claim_ratio(elements, (c.get("independent_claim_draft") or {}).get(form, ""))
        if ratio is not None:
            ratios.append(ratio)
        cands.append({**c, "elements": elements, "claim_ratio": ratio})

    extraction = {"doc_kind": doc_kind, "candidate_inventions": cands, "no_invention_reason": no_invention_reason}
    core = next((c for c in cands if c.get("level") == "core"), cands[0] if cands else None)
    checklist = []
    if core:
        kept = [e for e in core["elements"] if not e["unsupported"]]
        checklist = [{"id": e["id"], "criterion": e["text"], "weight": 1.0 / len(kept)} for e in kept]
    errors = {"n_elements": n_el, "n_unsupported": n_unsup,
              "quote_survival": round(1 - n_unsup / n_el, 4) if n_el else None,
              "claim_ratio_min": min(ratios) if ratios else None,
              "per_candidate": {c["id"]: {"n": len(c["elements"]),
                                          "unsupported": sum(e["unsupported"] for e in c["elements"]),
                                          "claim_ratio": c["claim_ratio"]} for c in cands}}
    return extraction, checklist, errors


async def verify_node(state: ExtractionState) -> dict:
    extraction, checklist, errors = verify_extraction(
        state.get("raw_extraction") or {}, state["full_text"], state.get("doc_kind", "paper"),
        state.get("no_invention_reason"))
    errors["llm_calls"] = state.get("llm_calls", 0)
    msg = (f"{errors['n_elements']} elements, {errors['n_unsupported']} unsupported quotes, "
           f"checklist={len(checklist)}, claim_ratio_min={errors['claim_ratio_min']}")
    return {"extraction": extraction, "checklist": checklist, "errors": errors, "events": [_event("verified", msg)]}


async def self_check_node(state: ExtractionState) -> dict:
    """A5: is the claim draft faithful to the document? Fail -> retry A2 once."""
    from app.llm import self_check

    cands = (state.get("extraction") or {}).get("candidate_inventions") or []
    retry_count = state.get("retry_count", 0)
    drafts = "\n\n".join(
        f'{c["id"]} ({c.get("level")}): {c.get("concept", "")}\n'
        f'METHOD CLAIM: {(c.get("independent_claim_draft") or {}).get("method", "")}\n'
        f'SYSTEM CLAIM: {(c.get("independent_claim_draft") or {}).get("system", "")}'
        for c in cands)
    n_el = sum(len(c.get("elements") or []) for c in cands)
    if not n_el:
        check = {"ok": False, "issues": ["no elements were extracted"], "suggestion": "output the JSON schema exactly"}
        calls = 0
    else:
        check = await self_check("independent claim drafting", state["full_text"], drafts)
        calls = 1
    ok = bool(check.get("ok", True))
    if ok:
        return {"self_check_ok": True, "llm_calls": state.get("llm_calls", 0) + calls,
                "events": [_event("self_check_pass", "Claim drafts passed self-check")]}
    issues = check.get("issues") or []
    return {"self_check_ok": False, "retry_count": retry_count + 1,
            "llm_calls": state.get("llm_calls", 0) + calls,
            "feedback": {"issues": issues, "suggestion": check.get("suggestion", ""),
                         "previous_response": drafts[:4000]},
            "events": [_event("self_check_fail", f"Issues: {', '.join(map(str, issues))[:200]}")]}


def should_retry_elements(state: ExtractionState) -> str:
    if state.get("self_check_ok") or state.get("retry_count", 0) > MAX_RETRY:
        return END
    return "elements"


def build_extraction_subgraph():
    g = StateGraph(ExtractionState)
    g.add_node("candidates", candidates_node)
    g.add_node("elements", elements_node)
    g.add_node("verify", verify_node)
    g.add_node("self_check", self_check_node)

    g.set_entry_point("candidates")
    g.add_conditional_edges("candidates", route_after_candidates, {"elements": "elements", END: END})
    g.add_edge("elements", "verify")
    g.add_edge("verify", "self_check")
    g.add_conditional_edges("self_check", should_retry_elements, {"elements": "elements", END: END})
    return g.compile()
