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


def _union(elements: list[dict], facet: str, cap: int) -> tuple[list[str], dict[str, list[str]]]:
    """Forms in element order, deduped; also which element each form came from."""
    seen: list[str] = []
    origin: dict[str, list[str]] = {}
    for e in elements:
        for t in (e.get("facets") or {}).get(facet) or []:
            t = " ".join(str(t).lower().split())
            if t and t not in seen:
                seen.append(t)
            if t:
                origin.setdefault(t, []).append(e.get("id", "?"))
    kept = seen[:cap]
    return kept, {t: origin[t] for t in kept}


def candidate_queries(cand: dict) -> list[dict]:
    """≤3 queries for one candidate invention, none scoped by `place`:
    thing forms alone (patent phrasings first); names AND thing (when the
    element has validated names); thing AND apparatus. The place facet
    (radiotherapy, prostate cancer, ...) only ranks: H1-02's gold were
    generic optical-tracking / machine-guidance patents that a domain word
    excluded structurally (H_monitor.md, H1.md §H7). Each query records the
    forms it was built from and the elements they came from."""
    els = cand.get("elements") or []
    n_terms, n_from = _union(els, "named", 6)
    t_terms, t_from = _union(els, "thing", 8)
    a_terms, a_from = _union(els, "apparatus", 6)
    names, things, apps = _named_group(n_terms), _group(t_terms), _group(a_terms)

    def _q(kind, query, **facets):
        used = {k: v for k, v in facets.items() if v}
        els_used = sorted({eid for k, terms in used.items() for t in terms
                           for eid in {"named": n_from, "thing": t_from, "apparatus": a_from}[k].get(t, [])})
        return {"kind": kind, "query": query, "facets_used": used, "elements": els_used}

    out = []
    concept = " ".join(str(cand.get("concept") or "").split())
    if concept:
        # natural-language query: Google Patents ranks free text semantically
        # (r/patentexaminer: "Use natural language to describe the problem
        # being solved, concisely"); counted as its own channel in the funnel
        out.append({"kind": "natural", "query": concept[:300], "facets_used": {"concept": [concept[:120]]},
                    "elements": [e.get("id") for e in els if e.get("id")]})
    if not things:
        return out[:PER_CANDIDATE]
    out.append(_q("thing", things, thing=t_terms))
    if names:
        out.append(_q("named+thing", f"{names} {things}", named=n_terms, thing=t_terms))
    if apps and len(out) < PER_CANDIDATE:
        out.append(_q("thing+apparatus", f"{things} {apps}", thing=t_terms, apparatus=a_terms))
    return out[:PER_CANDIDATE]


def cpc_queries(cands: list[dict], subclasses: list[str], max_total: int = 2) -> list[dict]:
    """`CPC=X/low` AND the core candidate's thing forms, for the top seed
    subclasses (PatentRiff: "A practitioner might start by searching within a
    relevant CPC class and then use keywords to filter the results")."""
    if not cands or not subclasses:
        return []
    from .query_gen import cpc_clause
    core = cands[0]
    t_terms, t_from = _union(core.get("elements") or [], "thing", 8)
    things = _group(t_terms)
    if not things:
        return []
    out = []
    for sc in subclasses[:max_total]:
        out.append({"candidate": core.get("id") or "inv1", "kind": "cpc+thing", "query": f"{cpc_clause(sc)} {things}",
                    "facets_used": {"cpc": [sc], "thing": t_terms},
                    "elements": sorted({e for t in t_terms for e in t_from.get(t, [])})})
    return out


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
            if q["kind"] in ("natural", "thing"):
                _take(cid, q)
    for cid, qs in per[1:]:
        for q in qs:
            _take(cid, q)
    return out
