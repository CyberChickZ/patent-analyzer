"""Phase 1: IDCA — Invention Detection, Classification, and Assignment.

Reads the uploaded PDF, extracts metadata (title, arxiv id, doi),
calls LLM for invention detection + summary.
"""

import asyncio
import json
import os
import re
from pathlib import Path

from state import GraphState

EXPLICIT_INPUT_MODES = ("academic_paper", "manuscript", "disclosure", "patent_draft")


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


async def _text_layer(input_path: str, plain_text: str, input_mode: str, _event) -> tuple[str, dict | None, dict]:
    """(document_text, doc_json, doc_json_stats). Gemini's Doc JSON rendered with
    [S<path>.P<n>] markers is the text layer extraction / self_check read; when
    the call fails (or IDCA_DOC_JSON=0) it is the fitz / plain text. Manuscript
    mode drops the prior-art passages before rendering (one extra LLM call, see
    adapters.manuscript). fallback_paragraphs =
    what the deterministic splitter finds in the plain text, for comparison."""
    from app.llm import build_doc_json
    from patent_analyzer.adapters.docjson import doc_json_stats, render_doc_json
    from patent_analyzer.adapters.manuscript import cut_flat, outline_flat, verdict_lines
    from patent_analyzer.adapters.paper import doc_from_text, iter_paragraphs

    if input_path.endswith(".pdf") and not plain_text:
        from app.llm import _EXTRACTION_DOC_CAP, _extract_pdf_text
        plain_text = _extract_pdf_text(input_path, max_pages=80, max_chars=_EXTRACTION_DOC_CAP)
    n_plain = sum(1 for _ in iter_paragraphs(doc_from_text(plain_text)))

    doc = None
    if os.environ.get("IDCA_DOC_JSON", "1") != "0":
        try:
            doc = await build_doc_json(plain_text, source_pdf_path=input_path if input_path.endswith(".pdf") else None)
        except Exception as e:
            _event("doc_json_failed", f"Doc JSON call failed: {type(e).__name__}: {str(e)[:160]}")
    if not doc:
        stats = {**doc_json_stats(None), "source": "none", "fallback_paragraphs": n_plain}
        _event("doc_json", f"No Doc JSON — text layer is the plain/fitz text ({n_plain} paragraphs)")
        return plain_text, None, stats
    if input_mode == "manuscript":
        items = outline_flat(doc)
        doc, verdicts = await cut_flat(doc)
        _event("prior_art_cut", f"Manuscript: dropped {len(doc.get('dropped_sections') or [])} prior-art sections "
               f"and {len(doc.get('dropped_paragraphs') or [])} background paragraphs"
               + (" — " + "; ".join(doc["dropped_sections"])[:200] if doc.get("dropped_sections") else ""),
               {"verdicts": verdict_lines(items, verdicts)})
    stats = {**doc_json_stats(doc), "source": "gemini", "fallback_paragraphs": n_plain}
    _event("doc_json", f"Doc JSON: {stats['sections']} sections · {stats['paragraphs']} paragraphs · "
           f"{stats['figures']} figures · {stats['equations']} equations · {stats['references_count']} refs "
           f"(plain-text splitter: {n_plain} paragraphs)", {"doc_json_stats": stats})
    return render_doc_json(doc), doc, stats


async def idca_node(state: GraphState) -> dict:
    """Phase 1 node: detect invention, classify, summarize."""
    from app.llm import detect_and_summarize_invention, set_llm_hook

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

    explicit_mode = state.get("input_mode") if state.get("input_mode") in EXPLICIT_INPUT_MODES else ""
    _pdf = input_path if input_path.endswith(".pdf") else None
    detection_result, (text_layer, doc_json, doc_stats) = await asyncio.gather(
        detect_and_summarize_invention(document_text, source_pdf_path=_pdf),
        _text_layer(input_path, document_text, explicit_mode, _event),
    )

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

    if not source_title and doc_json and doc_json.get("title"):
        source_title = doc_json["title"][:200]
    if not source_title:
        cite = detection_result.get("source_citation", "")
        if cite and len(cite) > 10:
            source_title = cite.split(".")[0].strip()[:120]

    _event("info", f"status={status_det} · doc_type={doc_type} · cpc={cpc_subclass}")

    patch: dict = {
        "phase": "phase1",
        "source_title": source_title,
        "source_arxiv_id": source_arxiv_id,
        "source_doi": source_doi,
        "document_text": text_layer,
        "doc_json": doc_json,
        "doc_json_stats": doc_stats,
        "status_determination": status_det,
        "doc_type": doc_type,
        "input_mode": explicit_mode or detection_result.get("input_mode", "academic_paper"),
        "category": detection_result.get("category", "None"),
        "fields_map": fields_map,
        "cpc_subclass": cpc_subclass,
        "source_citation": detection_result.get("source_citation", ""),
        "publication_date": detection_result.get("publication_date", ""),
        "summary": summary,
        "reasoning": detection_result.get("reasoning", ""),
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
