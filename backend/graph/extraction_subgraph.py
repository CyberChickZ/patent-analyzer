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
    "disclosure": "disclosure", "patent_draft": "patent_draft", "claim_text": "patent_draft",
    "manuscript": "manuscript",
}


class ExtractionInput(TypedDict, total=False):
    """What the parent graph hands this subgraph — deliberately WITHOUT `events`.

    A compiled subgraph used as a node receives the parent's whole state, and
    this one declares `events` with an `operator.add` reducer, so it arrived
    carrying the four events IDCA had already emitted and returned them in its
    own update patch. app/main.py appends the patch wholesale, so every phase-1
    event reached the feed twice — four duplicates with byte-identical
    microsecond timestamps, which is what gave it away. eval_subgraph declares a
    plain list (overwrite), which is why only phase 1 doubled. Reproduced and
    the fix verified on langgraph 0.2.76 by the M3b agent; the internal reducer
    stays because six nodes in here append to `events` independently.
    """
    summary: str
    document_text: str
    doc_json: dict | None
    input_local_path: str
    input_mode: str
    doc_type: str
    doc_kind: str
    cpc_subclass: str


class ExtractionState(TypedDict, total=False):
    # input from parent
    summary: str
    document_text: str
    doc_json: dict | None
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
    from patent_analyzer import event_log
    evt = {"ts": datetime.now(timezone.utc).isoformat(), "phase": "phase2",
           "kind": kind, "message": message}
    event_log.emit(evt)
    return evt


def raw_file_text(state: ExtractionState) -> str:
    """The input file's own text (fitz for PDFs) — the fallback layer."""
    path = state.get("input_local_path") or ""
    if not path or not Path(path).exists():
        return ""
    try:
        if path.endswith(".pdf"):
            from app.llm import _EXTRACTION_DOC_CAP, _extract_pdf_text
            return _extract_pdf_text(path, max_pages=80, max_chars=_EXTRACTION_DOC_CAP)
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


def resolve_doc_text(state: ExtractionState) -> str:
    """The text layer. With a Doc JSON, state.document_text is its rendered
    marker text and is used as-is (what the model reads = what is verified).
    Without one, the input file's text when it is longer than
    state.document_text (legacy 20k truncation), else state.document_text."""
    text = state.get("document_text") or ""
    if state.get("doc_json"):
        return text
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
    # claim text is already structured: the raw file (not the rendered Doc JSON) decides claim mode
    claim_text = raw_file_text(state) if state.get("doc_json") else text
    if is_claim_mode(state, claim_text):
        cands, prefill = claim_prefill(claim_text)
        if cands:
            return {"full_text": claim_text, "doc_kind": "patent_draft", "candidates": cands, "claim_prefill": prefill,
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


def _headings_by_path(doc_json: dict | None) -> dict[str, str]:
    if not doc_json:
        return {}
    from patent_analyzer.adapters.docjson import iter_doc_json_paragraphs
    return {path: heading for path, heading, _, _ in iter_doc_json_paragraphs(doc_json)}


def verify_extraction(raw: dict, text: str, doc_kind: str, no_invention_reason=None,
                      doc_json: dict | None = None, fallback_text: str = "") -> tuple[dict, list[dict], dict]:
    """A3 (pure): snap every quote, mark unsupported, measure element/claim
    agreement. `text` is the text layer (rendered Doc JSON when `doc_json` is
    given: evidence_loc then carries section path + heading + paragraph and
    source="doc_json"); a quote missing there is tried in `fallback_text` (the
    raw file text) and, when found, located with source="fallback_text" and no
    section. Returns (extraction, checklist, errors) with errors.doc_json_hits /
    fallback_hits."""
    from evals.extraction_errors import snap_quote
    from patent_analyzer.adapters.paper import locate_marker

    has_markers = "[S" in text and re.search(r"\[S[\d.]+\.P\d+\]", text) is not None
    headings = _headings_by_path(doc_json)
    cands = []
    n_el = n_unsup = n_doc = n_fb = 0
    ratios = []
    for c in raw.get("candidate_inventions") or []:
        elements = []
        for e in c.get("elements") or []:
            e = dict(e)
            quote = e.get("evidence_quote") or ""
            found, loc = snap_quote(quote, text)
            if found:
                span = loc.get("char") if loc else None
                pos = locate_marker(text, span[0]) if (has_markers and span) else {"section": None, "para": None}
                e["evidence_loc"] = {"section": pos["section"], "para": pos["para"], "char": span,
                                     "method": loc.get("method"), "sim": loc.get("sim")}
                if doc_json:
                    e["evidence_loc"]["heading"] = headings.get(pos["section"] or "", "") or None
                    e["evidence_loc"]["source"] = "doc_json"
                    n_doc += 1
                e["unsupported"] = False
            else:
                found, loc = snap_quote(quote, fallback_text) if fallback_text else (False, None)
                if found:
                    e["evidence_loc"] = {"section": None, "para": None, "char": loc.get("char") if loc else None,
                                         "method": loc.get("method"), "sim": loc.get("sim"), "source": "fallback_text"}
                    e["unsupported"] = False
                    n_fb += 1
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
              "doc_json_hits": n_doc if doc_json else None, "fallback_hits": n_fb if doc_json else None,
              "per_candidate": {c["id"]: {"n": len(c["elements"]),
                                          "unsupported": sum(e["unsupported"] for e in c["elements"]),
                                          "claim_ratio": c["claim_ratio"]} for c in cands}}
    return extraction, checklist, errors


async def verify_node(state: ExtractionState) -> dict:
    doc_json = state.get("doc_json") or None
    extraction, checklist, errors = verify_extraction(
        state.get("raw_extraction") or {}, state["full_text"], state.get("doc_kind", "paper"),
        state.get("no_invention_reason"), doc_json=doc_json, fallback_text=raw_file_text(state) if doc_json else "")
    errors["llm_calls"] = state.get("llm_calls", 0)
    msg = (f"{errors['n_elements']} elements, {errors['n_unsupported']} unsupported quotes, "
           f"checklist={len(checklist)}, claim_ratio_min={errors['claim_ratio_min']}")
    if doc_json:
        msg += f", located in Doc JSON: {errors['doc_json_hits']}, in raw text only: {errors['fallback_hits']}"
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
    g = StateGraph(ExtractionState, input=ExtractionInput)
    g.add_node("a1_candidates", candidates_node)
    g.add_node("a2_elements", elements_node)
    g.add_node("a3_verify", verify_node)
    g.add_node("a5_self_check", self_check_node)

    g.set_entry_point("a1_candidates")
    g.add_conditional_edges("a1_candidates", route_after_candidates, {"elements": "a2_elements", END: END})
    g.add_edge("a2_elements", "a3_verify")
    g.add_edge("a3_verify", "a5_self_check")
    g.add_conditional_edges("a5_self_check", should_retry_elements, {"elements": "a2_elements", END: END})
    return g.compile()
