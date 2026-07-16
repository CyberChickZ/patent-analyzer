"""Phase 1: IDCA — Invention Detection, Classification, and Assignment.

Reads the uploaded PDF, extracts metadata (title, arxiv id, doi),
calls LLM for invention detection + summary, crafts domain-specific personas.
"""

import json
import re
from pathlib import Path

from state import GraphState


def _is_header_boilerplate(s: str) -> bool:
    low = (s or "").lower().strip()
    if not low:
        return True
    bad_prefixes = (
        "abstract", "copyright", "©", "published as", "under review",
        "proceedings of", "preprint", "arxiv:", "submitted to",
        "accepted at", "to appear", "workshop on", "workshop at",
    )
    bad_contains = ("conference paper at", "iclr", "neurips", "icml", "cvpr")
    if low.startswith(bad_prefixes):
        return True
    if any(c in low for c in bad_contains) and len(s) < 80:
        return True
    return False


def _load_cpc_context(cpc_subclass: str) -> str:
    cpc_path = Path(__file__).parent.parent / "app" / "cpc_reference.json"
    if not cpc_path.exists() or not cpc_subclass:
        return ""
    try:
        data = json.loads(cpc_path.read_text())
        entry = data.get(cpc_subclass, {})
        return json.dumps(entry, indent=2) if entry else ""
    except Exception:
        return ""


async def idca_node(state: GraphState) -> dict:
    """Phase 1 node: detect invention, summarize, craft personas."""
    from app.llm import (
        INITIAL_PERSONAS,
        craft_personas,
        detect_and_summarize_invention,
        set_llm_hook,
    )

    input_path = state["input_local_path"]
    events = []

    def _event(kind: str, message: str, payload: dict | None = None):
        from datetime import datetime, timezone
        evt = {"ts": datetime.now(timezone.utc).isoformat(),
               "phase": "phase1", "kind": kind, "message": message}
        if payload:
            evt["payload"] = payload
        events.append(evt)

    _event("start", "Reading PDF and detecting invention")

    source_title = ""
    source_arxiv_id = ""
    source_doi = ""
    document_text = ""

    if input_path.endswith(".pdf"):
        import fitz
        doc = fitz.open(input_path)
        n_pages = len(doc)
        first_page_text = doc[0].get_text() if n_pages > 0 else ""
        meta_title = (doc.metadata or {}).get("title", "") or ""
        if meta_title and len(meta_title) > 20 and not _is_header_boilerplate(meta_title):
            source_title = meta_title.strip()
        else:
            for line in (first_page_text or "").split("\n"):
                line = line.strip()
                if len(line) > 15 and not _is_header_boilerplate(line):
                    source_title = line
                    break
        id_text = "\n".join(doc[i].get_text() for i in range(min(2, n_pages)))
        ax = re.search(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", id_text, re.IGNORECASE)
        if ax:
            source_arxiv_id = ax.group(1).split("v")[0]
        dx = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", id_text, re.IGNORECASE)
        if dx:
            source_doi = dx.group(1).rstrip(".,;)")
        doc.close()
        _event("info", f"PDF: {n_pages} pages")
    else:
        document_text = Path(input_path).read_text()
        for line in document_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("title:"):
                source_title = line[6:].strip()[:200]
                break
            elif len(line) > 15 and not _is_header_boilerplate(line):
                source_title = line[:200]
                break

    _pdf = input_path if input_path.endswith(".pdf") else None
    detection_result = await detect_and_summarize_invention(
        document_text, source_pdf_path=_pdf)

    if not detection_result:
        detection_result = {
            "status_determination": "Present",
            "has_innovation": True,
            "reasoning": "(detection LLM call failed)",
            "doc_type": "invention",
            "category": "None",
            "fields_map": [],
            "source_citation": "",
            "cpc_subclass": "",
            "summary": "(unavailable)",
        }

    status_det = detection_result.get("status_determination", "Present")
    doc_type = detection_result.get("doc_type", "invention")
    fields_map = detection_result.get("fields_map", [])
    cpc_subclass = detection_result.get("cpc_subclass", "")
    summary = detection_result.get("summary") or "(empty summary)"

    if not source_title:
        cite = detection_result.get("source_citation", "")
        if cite and len(cite) > 10:
            source_title = cite.split(".")[0].strip()[:120]

    _event("info", f"status={status_det} · doc_type={doc_type} · cpc={cpc_subclass}")

    # Craft domain-specific personas
    personas = dict(INITIAL_PERSONAS)
    if status_det != "Absent" and doc_type != "talks_about_invention_but_no_invention":
        try:
            cpc_ctx = _load_cpc_context(cpc_subclass)
            personas = await craft_personas(
                doc_type=doc_type,
                fields_map=fields_map,
                cpc_subclass=cpc_subclass,
                cpc_context=cpc_ctx,
                summary_excerpt=summary[:500],
            )
            _event("info", f"Crafted {len(personas)} domain-specific personas")
        except Exception:
            _event("persona_fallback", "Persona crafting failed, using defaults")

    patch: dict = {
        "phase": "phase1",
        "source_title": source_title,
        "source_arxiv_id": source_arxiv_id,
        "source_doi": source_doi,
        "document_text": document_text[:20000],
        "status_determination": status_det,
        "doc_type": doc_type,
        "input_mode": detection_result.get("input_mode", "academic_paper"),
        "category": detection_result.get("category", "None"),
        "fields_map": fields_map,
        "cpc_subclass": cpc_subclass,
        "source_citation": detection_result.get("source_citation", ""),
        "publication_date": detection_result.get("publication_date", ""),
        "summary": summary,
        "reasoning": detection_result.get("reasoning", ""),
        "personas": personas,
        "events": events,
        "phase_results": {"phase1": {
            "status": "completed",
            "data": {
                "status_determination": status_det,
                "doc_type": doc_type,
                "has_innovation": status_det != "Absent",
            },
        }},
    }

    if status_det == "Absent" or doc_type == "talks_about_invention_but_no_invention":
        patch["status"] = "completed"

    return patch
