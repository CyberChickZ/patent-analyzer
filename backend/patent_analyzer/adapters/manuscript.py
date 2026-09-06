"""Manuscript adapter: a submission draft is a paper whose prior-art review must
not be mistaken for the authors' own invention. What gets cut is decided by ONE
LLM call over the section outline (app.llm.classify_prior_art_sections), not by a
regex over headings.

Why not the regex (v1, retired 2026-09-18): on the 8 Pap2Pat papers of §H1.7.1 it
matched a heading in exactly one of them. Real papers put their prior art under
`Introduction` / `Materials and Methods`, and only rarely under a heading that
says "Related Work". The regex also had no answer for the usual shape of an
Introduction — background, citations, then "Here we introduce ..." — where
dropping the section drops the one paragraph that states the invention. The LLM
labels each section prior_art / mixed / own_work and, for `mixed`, names the
paragraphs that are purely somebody else's work, so the cut is per paragraph.

The abstract is KEPT. v1 cleared it; a submission draft has an abstract, it is
the authors' own summary of their own contribution, and clearing it has nothing
to do with prior art. (It was also the confound that made the first §H1.7.1
table an abstract ablation rather than a prior-art one.)

Fail-open: a section the model did not judge is kept. Losing the Method costs
more than keeping a background paragraph.
"""

import copy

ADAPTER_VERSION = "v2"


def outline_nested(doc: dict) -> list[dict]:
    """[{id, heading, paragraphs}] for an adapters.paper Doc, ids matching the
    S-paths render_doc / iter_paragraphs assign (S1, S1.1, ...)."""
    out: list[dict] = []

    def walk(secs, path):
        for i, s in enumerate(secs or [], 1):
            sp = f"{path}.{i}" if path else f"S{i}"
            out.append({"id": sp, "heading": s.get("title") or "", "paragraphs": list(s.get("paragraphs") or [])})
            walk(s.get("subsections"), sp)

    walk(doc.get("sections"), "")
    return out


def outline_flat(doc: dict) -> list[dict]:
    """Same, for a Doc JSON (flat sections with a level)."""
    from patent_analyzer.adapters.docjson import section_paths

    secs = doc.get("sections") or []
    return [{"id": p, "heading": s.get("heading") or "", "paragraphs": list(s.get("paragraphs") or [])}
            for p, s in zip(section_paths(secs), secs)]


def _cut(paras: list[str], verdict: dict | None) -> tuple[list[str], list[int]] | None:
    """(paragraphs to keep, 1-based indices dropped), or None when the whole section goes."""
    if not verdict:
        return list(paras), []
    if verdict["verdict"] == "prior_art":
        return None
    drop = {i for i in (verdict.get("prior_art_paragraphs") or []) if 1 <= i <= len(paras)} \
        if verdict["verdict"] == "mixed" else set()
    return [p for i, p in enumerate(paras, 1) if i not in drop], sorted(drop)


def apply_nested(doc: dict, verdicts: dict) -> dict:
    """Cut an adapters.paper Doc by {section_id: verdict}. Dropping a section drops
    its subsections with it. Records `dropped_sections` (headings removed whole) and
    `dropped_paragraphs` [(section_id, para_index)]."""
    out = copy.deepcopy(doc)
    dropped: list[str] = []
    dropped_paras: list[list] = []

    def prune(secs, path):
        kept = []
        for i, s in enumerate(secs or [], 1):
            sp = f"{path}.{i}" if path else f"S{i}"
            cut = _cut(s.get("paragraphs") or [], verdicts.get(sp))
            if cut is None:
                dropped.append(s.get("title") or "")
                continue
            paras, drop_idx = cut
            dropped_paras.extend([sp, j] for j in drop_idx)
            s["paragraphs"] = paras
            s["subsections"] = prune(s.get("subsections"), sp)
            kept.append(s)
        return kept

    out["sections"] = prune(out.get("sections"), "")
    out["kind"] = "manuscript"
    out["dropped_sections"] = dropped
    out["dropped_paragraphs"] = dropped_paras
    return out


def apply_flat(doc: dict, verdicts: dict) -> dict:
    """Cut a Doc JSON by {section_id: verdict}. A dropped section takes the deeper
    entries that follow it (its subsections) with it, whatever they were judged."""
    from patent_analyzer.adapters.docjson import section_paths

    secs = doc.get("sections") or []
    paths = section_paths(secs)
    out = {**doc, "sections": [], "dropped_sections": [], "dropped_paragraphs": []}
    skip_below: int | None = None
    for path, sec in zip(paths, secs):
        level = max(1, int(sec.get("level") or 1))
        if skip_below is not None and level > skip_below:
            out["dropped_sections"].append(sec.get("heading") or "")
            continue
        skip_below = None
        cut = _cut(sec.get("paragraphs") or [], verdicts.get(path))
        if cut is None:
            out["dropped_sections"].append(sec.get("heading") or "")
            skip_below = level
            continue
        paras, drop_idx = cut
        out["dropped_paragraphs"] += [[path, j] for j in drop_idx]
        out["sections"].append({**sec, "paragraphs": paras})
    return out


async def cut_nested(doc: dict) -> tuple[dict, dict]:
    """(cut Doc, verdicts) — one LLM call. Never raises: on failure nothing is cut."""
    from app.llm import classify_prior_art_sections

    items = outline_nested(doc)
    try:
        verdicts = await classify_prior_art_sections(doc.get("title") or "", items)
    except Exception:
        verdicts = {}
    return apply_nested(doc, verdicts), verdicts


async def cut_flat(doc: dict) -> tuple[dict, dict]:
    """(cut Doc JSON, verdicts) — one LLM call. Never raises."""
    from app.llm import classify_prior_art_sections

    items = outline_flat(doc)
    try:
        verdicts = await classify_prior_art_sections(doc.get("title") or "", items)
    except Exception:
        verdicts = {}
    return apply_flat(doc, verdicts), verdicts


def verdict_lines(items: list[dict], verdicts: dict) -> list[str]:
    """Human-readable audit trail: one line per judged section."""
    out = []
    for it in items:
        v = verdicts.get(it["id"])
        if not v:
            continue
        n = f" P{v['prior_art_paragraphs']}" if v.get("prior_art_paragraphs") else ""
        out.append(f"{it['id']} {it['heading'] or '(no heading)'} -> {v['verdict']}{n}: {v['reason']}")
    return out
