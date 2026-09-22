"""Google Patents boolean queries from element facets.

Google syntax (official help, verified 2026-09-18): space = AND with left
associativity, OR, parentheses, "phrase" = exact, AB=/CL=/TI= field
scoping, CPC=, '*' wildcard; unquoted keywords are stemmed and get close
synonyms automatically; NEAR/ADJ/SAME only affect ranking, never
retrieval, so they are appended as a ranking hint and ignored by the
validator. Dates go through the separate `before` parameter.

A facet is an OR group of surface forms (patent-search-pilot: 6-14 forms
per facet, two words max). A multi-word form is written unquoted inside
its own parentheses — (sound damping) — so each word is stemmed and the
words only have to co-occur in the document, not as an exact phrase.
"""

from __future__ import annotations

MODES = ("strict", "loose", "core")
FORMS_PER_FACET = 8


def _clean(t: str) -> str:
    # a hyphen inside a word is read as Google's minus/NOT ("pan-tilt" → pan NOT tilt):
    # ReAct h1i H1-01 steps 2/3 returned 0, the same queries with "pan tilt" 124k/129k
    return " ".join(str(t).replace("-", " ").split()).strip().strip('"')


def _group(terms: list[str], cap: int = FORMS_PER_FACET) -> str:
    ts = []
    for t in terms:
        t = _clean(t)
        if t and t not in ts:
            ts.append(t)
    ts = ts[:cap]
    if not ts:
        return ""
    q = " OR ".join(f"({t})" if " " in t else t for t in ts)
    return f"({q})" if len(ts) > 1 else q


def _distinct(f: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    """Facets as term lists with a term never repeated across facets (a
    form in `named` is dropped from `thing`, a form in `thing` from
    `place`/`apparatus`): repeating it just lengthens the query without
    narrowing it (H1-06: `accelerometer housing accelerometer`)."""
    seen, out = set(), []
    for k in ("named", "thing", "place", "apparatus"):
        terms = []
        for t in f.get(k) or []:
            key = " ".join(str(t).lower().split()).strip('"')
            if key and key not in seen:
                seen.add(key)
                terms.append(t)
        out.append(terms)
    return out[0], out[1], out[2], out[3]


def _named_group(terms: list[str], cap: int = 6) -> str:
    """Distinctive names are fixed strings: multi-word names stay an exact
    "phrase" (indocyanine green), acronyms a bare word."""
    ts = []
    for t in terms:
        t = _clean(t)
        if t and t not in ts:
            ts.append(t)
    ts = ts[:cap]
    if not ts:
        return ""
    q = " OR ".join(f'"{t}"' if " " in t else t for t in ts)
    return f"({q})" if len(ts) > 1 else q


def cpc_clause(code: str) -> str:
    """Google Patents: `CPC=B60R22` matches exactly that code, `/low` adds the
    children. Measured through SerpAPI (H7 gold probe, 2026-09-18): the
    clause only works at MAIN-GROUP level or deeper (`CPC=H04N7/low` →
    20,312; `CPC=H04N/low`, `CPC=H04N`, `cpc:H04N7` → 0) and only when it
    is the LAST term of the query (leading `CPC=… (a OR b)` → 0). Callers
    append the returned clause at the end."""
    c = (code or "").strip().upper().split("/")[0]
    if not c or len(c) < 5:          # subclass (4 chars) never matches
        return ""
    return f"CPC={c}/low"


def boolean_query(element: dict, mode: str = "strict", field: str = "AB", cpc: str | None = None) -> str:
    """strict: named AND thing AND place AND apparatus, scoped to field (AB
    or CL), optional CPC=; loose: thing AND place (the names are dropped —
    a coinage of the document would otherwise zero every query); core:
    thing only, no field. The names are the element's distinctive
    identifiers (chemical / organism / product names, acronym + expansion)."""
    f = element.get("facets") or {}
    n_terms, t_terms, p_terms, a_terms = _distinct(f)
    named, thing, place, app = _named_group(n_terms), _group(t_terms), _group(p_terms), _group(a_terms)
    if not thing:
        return ""
    if mode == "core":
        return thing
    parts = [thing, place] if mode == "loose" else [named, thing, place, app]
    parts = [p for p in parts if p]
    body = " ".join(parts)
    q = f"{field}=({body})" if field else body
    if mode == "strict" and thing and place:
        t0 = (t_terms or [""])[0]
        p0 = (p_terms or [""])[0]
        if t0 and p0 and " " not in t0 and " " not in p0:
            q += f" ({t0} NEAR/10 {p0})"   # ranking hint only
    if mode == "strict" and cpc and cpc_clause(cpc):
        q += " " + cpc_clause(cpc)        # must be last
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
