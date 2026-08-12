"""Doc JSON (the IDCA Gemini transcription) -> the single text layer the pipeline
reads, and back from a char offset to (section, paragraph).

Doc JSON = {title, abstract, sections: [{heading, level, paragraphs: [str]}],
            figures: [{label, caption}], equations: [{label, latex}], references_count}
(flat sections with a level, MinerU content_list style; figures / equations kept
apart, GROBID style — see app/llm.py DOC_JSON_SCHEMA).

render_doc_json gives every paragraph a [S<path>.P<n>] marker (abstract = S0,
figure captions and equations are appended as their own sections) so that
adapters.paper.locate_marker maps a quote's char offset back to its paragraph.
"""

from patent_analyzer.adapters.paper import locate_marker  # noqa: F401  (re-export for callers)

FIGURES_HEADING = "Figures"
EQUATIONS_HEADING = "Equations"


def section_paths(sections: list[dict]) -> list[str]:
    """S-paths from the flat level list: levels [1,2,2,1] -> S1, S1.1, S1.2, S2.
    A level deeper than parent+1 is clamped to parent+1."""
    counters: list[int] = []
    out = []
    for sec in sections or []:
        level = max(1, int(sec.get("level") or 1))
        level = min(level, len(counters) + 1)
        counters = counters[:level]
        if len(counters) < level:
            counters.append(0)
        counters[level - 1] += 1
        out.append("S" + ".".join(str(c) for c in counters))
    return out


def _extra_sections(doc: dict) -> list[dict]:
    """Figures / equations as trailing sections: 'Figure 1: caption', '(1) $latex$'."""
    extra = []
    figs = [f"{f['label']}: {f['caption']}" if f.get("label") else f["caption"]
            for f in doc.get("figures") or [] if f.get("caption")]
    if figs:
        extra.append({"heading": FIGURES_HEADING, "level": 1, "paragraphs": figs})
    eqs = [f"({e['label']}) ${e['latex']}$" if e.get("label") else f"${e['latex']}$"
           for e in doc.get("equations") or [] if e.get("latex")]
    if eqs:
        extra.append({"heading": EQUATIONS_HEADING, "level": 1, "paragraphs": eqs})
    return extra


def iter_doc_json_paragraphs(doc: dict):
    """Yield (section_path, heading, para_idx, text) in render order (abstract = S0;
    figure captions / equations last)."""
    if doc.get("abstract"):
        yield "S0", "Abstract", 1, doc["abstract"]
    sections = list(doc.get("sections") or []) + _extra_sections(doc)
    for path, sec in zip(section_paths(sections), sections):
        for j, p in enumerate(sec.get("paragraphs") or [], 1):
            yield path, sec.get("heading") or "", j, p


def render_doc_json(doc: dict, markers: bool = True) -> str:
    """Doc JSON -> text: title, abstract, '#'-headings, one paragraph per line
    block (blank line between), each paragraph prefixed by [S<path>.P<n>] when
    markers=True. Figures and equations are rendered as trailing sections."""
    parts = []
    if doc.get("title"):
        parts += [f"Title: {doc['title']}", ""]
    if doc.get("abstract"):
        parts += ["Abstract", ("[S0.P1] " if markers else "") + doc["abstract"], ""]
    sections = list(doc.get("sections") or []) + _extra_sections(doc)
    for path, sec in zip(section_paths(sections), sections):
        if sec.get("heading"):
            parts.append("#" * max(1, int(sec.get("level") or 1)) + " " + sec["heading"])
        for j, p in enumerate(sec.get("paragraphs") or [], 1):
            parts.append((f"[{path}.P{j}] " if markers else "") + p)
            parts.append("")
        if not sec.get("paragraphs"):
            parts.append("")
    return "\n".join(parts).strip() + "\n"


def doc_json_stats(doc: dict | None) -> dict:
    if not doc:
        return {"sections": 0, "paragraphs": 0, "figures": 0, "equations": 0, "references_count": 0, "chars": 0}
    paras = [p for _, _, _, p in iter_doc_json_paragraphs(doc)]
    return {"sections": len(doc.get("sections") or []), "paragraphs": len(paras),
            "figures": len(doc.get("figures") or []), "equations": len(doc.get("equations") or []),
            "references_count": int(doc.get("references_count") or 0), "chars": sum(len(p) for p in paras)}
