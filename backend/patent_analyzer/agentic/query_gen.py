"""Google Patents boolean queries from element facets.

Google syntax (official help, verified 2026-09-18): space = AND, OR,
parentheses, "phrase", AB=/CL=/TI= field scoping, CPC=, '*' wildcard;
NEAR/ADJ/SAME only affect ranking, never retrieval, so they are appended
as a ranking hint and ignored by the validator. Dates go through the
separate `before` parameter, not the query string.
"""

from __future__ import annotations

MODES = ("strict", "loose", "core")


def _group(terms: list[str], cap: int = 4) -> str:
    ts = []
    for t in terms:
        t = " ".join(str(t).split()).strip().strip('"')
        if t and t not in ts:
            ts.append(t)
    ts = ts[:cap]
    if not ts:
        return ""
    q = " OR ".join(f'"{t}"' if " " in t else t for t in ts)
    return f"({q})" if len(ts) > 1 else q


def boolean_query(element: dict, mode: str = "strict", field: str = "AB", cpc: str | None = None) -> str:
    """strict: thing AND place AND apparatus, scoped to field (AB or CL),
    optional CPC=; loose: thing AND place; core: thing only, no field."""
    f = element.get("facets") or {}
    thing, place, app = _group(f.get("thing") or []), _group(f.get("place") or []), _group(f.get("apparatus") or [])
    if not thing:
        return ""
    if mode == "core":
        return thing
    parts = [thing, place] if mode == "loose" else [thing, place, app]
    parts = [p for p in parts if p]
    body = " ".join(parts)
    q = f"{field}=({body})" if field else body
    if mode == "strict" and cpc:
        q += f" CPC={cpc}"
    if mode == "strict" and thing and place:
        t0 = (f.get("thing") or [""])[0]
        p0 = (f.get("place") or [""])[0]
        if t0 and p0 and " " not in t0 and " " not in p0:
            q += f" ({t0} NEAR/10 {p0})"   # ranking hint only
    return q


def next_mode(mode: str, verdict: str) -> str | None:
    """Relax on too_narrow/zero, tighten on too_broad; None = stop."""
    order = list(MODES)
    i = order.index(mode) if mode in order else 0
    if verdict in ("zero", "too_narrow"):
        return order[i + 1] if i + 1 < len(order) else None
    if verdict == "too_broad":
        return order[i - 1] if i > 0 else None
    return None
