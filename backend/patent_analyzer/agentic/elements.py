"""Elements are the unit the search loop works on: one claim-like
limitation each, with search facets (thing / place / apparatus)."""

from __future__ import annotations

import re

_GENERIC = {"device", "member", "element", "portion", "means", "unit", "system", "method",
            "apparatus", "module", "component", "assembly", "configured", "comprising", "wherein",
            "plurality", "first", "second", "third", "said", "least", "based", "using"}


def elements_from_state(state: dict) -> list[dict]:
    """Prefer the extraction subgraph's elements; fall back to SSR checklist
    criteria; fall back to claim limitations. Each: {id, text, source}."""
    ext = state.get("extraction") or {}
    cands = ext.get("candidate_inventions") or []
    if cands:
        core = cands[0]
        out = []
        for e in core.get("elements") or []:
            if e.get("unsupported"):
                continue
            out.append({"id": e["id"], "text": e["text"], "source": "extraction",
                        "facets": e.get("facets") or {}})
        if out:
            return out
    cl = state.get("checklist") or []
    if cl:
        return [{"id": c.get("id") or f"c{i + 1}", "text": c.get("criterion", ""), "source": "checklist",
                 "facets": {}} for i, c in enumerate(cl) if c.get("criterion")]
    doc = state.get("document_text") or ""
    if doc.lstrip().startswith("1."):
        from nodes.claim_mode import _parse_claim_limitations
        lims = _parse_claim_limitations(doc)["limitations"]
        return [{"id": f"lim{i + 1}", "text": t, "source": "claim", "facets": {}} for i, t in enumerate(lims)]
    return []


def candidates_from_state(state: dict) -> list[dict]:
    """Every candidate invention with its supported elements (core first);
    falls back to one pseudo-candidate built from elements_from_state."""
    ext = state.get("extraction") or {}
    cands = ext.get("candidate_inventions") or []
    out = []
    for i, c in enumerate(cands):
        els = [{"id": e["id"], "text": e["text"], "source": "extraction", "facets": e.get("facets") or {}}
               for e in (c.get("elements") or []) if not e.get("unsupported")]
        if els:
            out.append({"id": c.get("id") or f"inv{i + 1}", "level": c.get("level", ""),
                        "concept": c.get("concept", ""), "elements": els})
    if out:
        return out
    els = elements_from_state(state)
    return [{"id": "inv1", "level": "core", "concept": "", "elements": els}] if els else []


def fallback_facets(text: str) -> dict:
    """No-LLM facets: longest content words as 'thing', nothing else."""
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)]
    words = [w for w in words if w not in _GENERIC]
    uniq = list(dict.fromkeys(sorted(words, key=len, reverse=True)))
    return {"thing": uniq[:3], "place": uniq[3:5], "apparatus": []}


_GENERIC_NAMES = {"tablet pc", "camera", "server", "sensor", "display", "monitor", "robot", "controller", "computer",
                  "explicit control", "implicit control", "control", "database", "processor", "software", "hardware"}


def valid_name(term: str, source: str) -> bool:
    """A distinctive name must appear in the source text (element texts +
    summary) and be either a multi-word phrase or an acronym the source
    writes in capitals; generic product words never qualify (H1-01 'tablet
    pc', H1-02 'pss'/'ccd' burned the named queries)."""
    t = " ".join(str(term).lower().split())
    if not t or t in _GENERIC_NAMES or len(t) < 3:
        return False
    src = source or ""
    if t not in src.lower():
        return False
    if " " in t or "-" in t or any(ch.isdigit() for ch in t):
        return True
    return bool(re.search(r"\b" + re.escape(t.upper()) + r"\b", src)) and len(t) <= 8


def merge_facets(base: dict, extra: dict, cap: int = 10, source: str | None = None) -> dict:
    """Union of two facet samples, base first, deduped, capped per facet
    (patent-search-pilot: which words the model reaches for is a coin flip;
    the fix is to toss it twice and merge). Patent-vocabulary phrasings go
    to the front of `thing`; names are validated against `source` when given."""
    out = {}
    for k in ("named", "thing", "place", "apparatus"):
        seen = []
        pre = list((extra or {}).get("patent") or []) + list((base or {}).get("patent") or []) if k == "thing" else []
        for t in pre + list((base or {}).get(k) or []) + list((extra or {}).get(k) or []):
            t = " ".join(str(t).lower().split())
            if t and t not in seen:
                seen.append(t)
        if k == "named" and source is not None:
            seen = [t for t in seen if valid_name(t, source)]
        out[k] = seen[:cap]
    return out


async def attach_facets(elements: list[dict], summary: str) -> list[dict]:
    """One LLM call widens every element's facets (union with the
    extraction's own forms); fallback keeps the loop alive."""
    from app.llm import facet_elements
    got = await facet_elements([{"id": e["id"], "text": e["text"]} for e in elements], summary)
    source = "\n".join(e.get("text", "") for e in elements) + "\n" + (summary or "")
    for e in elements:
        merged = merge_facets(e.get("facets") or {}, got.get(e["id"]) or {}, source=source)
        e["facets"] = merged if merged.get("thing") else fallback_facets(e["text"])
    return elements
