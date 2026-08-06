"""Wide queries: recall first, precision later (prune.py).

Two-stage retrieval is the standard shape (Nogueira et al. 2019, "Multi-Stage
Document Ranking with BERT": "Following an initial retrieval stage, which
typically issues a 'bag of words' query against an inverted index, each
subsequent stage re-ranks the set of candidates"). Under a 250-call/month
SerpAPI budget the retrieval stage is 2-3 broad queries per candidate
invention, each one page of 100, no strict/loose retries: too_broad is the
design, the pruner does the narrowing.
"""

from __future__ import annotations

from .query_gen import _group, _named_group

MAX_QUERIES = 10
PER_CANDIDATE = 3


def _union(elements: list[dict], facet: str, cap: int) -> list[str]:
    seen: list[str] = []
    for e in elements:
        for t in (e.get("facets") or {}).get(facet) or []:
            t = " ".join(str(t).lower().split())
            if t and t not in seen:
                seen.append(t)
    return seen[:cap]


def candidate_queries(cand: dict) -> list[dict]:
    """≤3 queries for one candidate invention: the names alone; names AND
    the thing forms; thing AND place (or thing alone when there is no place).
    Without names: thing AND place, then thing alone."""
    els = cand.get("elements") or []
    names = _named_group(_union(els, "named", 6))
    things = _group(_union(els, "thing", 8))
    places = _group(_union(els, "place", 6))
    out = []
    if not things and not names:
        return out
    if names:
        out.append({"kind": "named", "query": names})
        if things:
            out.append({"kind": "named+thing", "query": f"{names} {things}"})
    if things and places:
        out.append({"kind": "thing+place", "query": f"{things} {places}"})
    elif things:
        out.append({"kind": "thing", "query": things})
    if not names and things and places:
        out.append({"kind": "thing", "query": things})
    return out[:PER_CANDIDATE]


def wide_queries(candidates: list[dict], max_total: int = MAX_QUERIES) -> list[dict]:
    """Queries over all candidate inventions, core candidate first, then the
    other candidates' named queries, then the rest — capped at max_total."""
    per = [(c.get("id") or f"inv{i + 1}", candidate_queries(c)) for i, c in enumerate(candidates)]
    out: list[dict] = []
    seen: set[str] = set()

    def _take(cid, q):
        if q["query"] not in seen and len(out) < max_total:
            seen.add(q["query"])
            out.append({"candidate": cid, **q})

    if per:
        for q in per[0][1]:
            _take(per[0][0], q)
    for cid, qs in per[1:]:
        for q in qs:
            if q["kind"].startswith("named"):
                _take(cid, q)
    for cid, qs in per[1:]:
        for q in qs:
            _take(cid, q)
    return out
