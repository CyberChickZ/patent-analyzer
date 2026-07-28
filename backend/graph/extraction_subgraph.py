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
