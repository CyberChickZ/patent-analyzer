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


def fallback_facets(text: str) -> dict:
    """No-LLM facets: longest content words as 'thing', nothing else."""
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)]
    words = [w for w in words if w not in _GENERIC]
    uniq = list(dict.fromkeys(sorted(words, key=len, reverse=True)))
    return {"thing": uniq[:3], "place": uniq[3:5], "apparatus": []}


def merge_facets(base: dict, extra: dict, cap: int = 10) -> dict:
    """Union of two facet samples, base first, deduped, capped per facet
    (patent-search-pilot: which words the model reaches for is a coin flip;
    the fix is to toss it twice and merge)."""
    out = {}
    for k in ("thing", "place", "apparatus"):
        seen = []
        for t in list((base or {}).get(k) or []) + list((extra or {}).get(k) or []):
            t = " ".join(str(t).lower().split())
            if t and t not in seen:
                seen.append(t)
        out[k] = seen[:cap]
    return out


async def attach_facets(elements: list[dict], summary: str) -> list[dict]:
    """One LLM call widens every element's facets (union with the
    extraction's own forms); fallback keeps the loop alive."""
    from app.llm import facet_elements
    got = await facet_elements([{"id": e["id"], "text": e["text"]} for e in elements], summary)
    for e in elements:
        merged = merge_facets(e.get("facets") or {}, got.get(e["id"]) or {})
        e["facets"] = merged if merged.get("thing") else fallback_facets(e["text"])
    return elements
