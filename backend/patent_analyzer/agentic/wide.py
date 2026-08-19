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


def candidate_queries(cand: dict, max_elements: int = 6, cpc_groups: list[str] | None = None) -> list[dict]:
    """Queries for one candidate invention (H7 v3, from the H1-01 funnels):
    one 8-form OR blob per candidate lost the gold that a narrower phrasing
    had reached (Google returns the top 100 of ~1e5; ranking variance
    dominates), so the budget goes to several NARROW queries instead:
      natural       — ≤10 words: patent phrasings of the first elements + domain
      wide          — blob of thing forms ∧ place forms (the h1d shape that hit)
      element:<id>  — per limitation: its thing forms (≤3) ∧ the invention's
                      domain (the preamble's thing forms), no place scoping
      named+domain  — validated distinctive names ∧ domain
    Each query records the forms it was built from and the elements."""
    els = cand.get("elements") or []
    if not els:
        return []
    e0 = els[0]
    dom_terms = [" ".join(str(t).lower().split()) for t in ((e0.get("facets") or {}).get("thing") or [])][:3]
    domain = _group(dom_terms)
    n_terms, n_from = _union(els, "named", 6)
    t_terms, t_from = _union(els, "thing", 8)
    p_terms, p_from = _union(els, "place", 4)
    out = []

    def _q(kind, query, elements, **facets):
        used = {k: v for k, v in facets.items() if v}
        return {"kind": kind, "query": query, "facets_used": used, "elements": elements}

    # natural: short free text, Google ranks it semantically
    words: list[str] = []
    for e in els[:4]:
        for t in ((e.get("facets") or {}).get("patent") or (e.get("facets") or {}).get("thing") or [])[:1]:
            for w in str(t).lower().split():
                if w not in words:
                    words.append(w)
    for t in dom_terms[:1]:
        for w in t.split():
            if w not in words:
                words.append(w)
    if len(words) >= 3:
        out.append(_q("natural", " ".join(words[:10]), [e.get("id") for e in els[:4]], natural=words[:10]))
    things = _group(t_terms)
    places = _group(p_terms)
    if things and places:
        out.append(_q("wide", f"{things} {places}", sorted({eid for t in t_terms for eid in t_from.get(t, [])}), thing=t_terms, place=p_terms))
    elif things:
        out.append(_q("wide", things, sorted({eid for t in t_terms for eid in t_from.get(t, [])}), thing=t_terms))
    from .query_gen import cpc_clause
    groups = [g for g in (cpc_groups if cpc_groups is not None else cand.get("cpc_pred") or []) if cpc_clause(g)]
    scope = cpc_clause(groups[0]) if groups else ""
    for e in els[1:1 + max_elements]:
        f = e.get("facets") or {}
        forms = [" ".join(str(t).lower().split()) for t in (f.get("thing") or [])][:3]
        forms = [t for t in forms if t and t not in dom_terms]
        if not forms:
            continue
        g = _group(forms)
        if scope:
            # gold probe: `(remote participant display) CPC=H04N7/low` → gold rank 19 (48 unscoped);
            # the CPC clause must be the last term
            out.append(_q(f"element:{e.get('id')}", f"{g} {scope}", [e.get("id")], thing=forms, cpc=[groups[0].split("/")[0]]))
        else:
            out.append(_q(f"element:{e.get('id')}", f"{g} {domain}" if domain else g, [e.get("id")], thing=forms, domain=dom_terms))
    if n_terms:
        names = _named_group(n_terms)
        out.append(_q("named+domain", f"{names} {domain}" if domain else names,
                      sorted({eid for t in n_terms for eid in n_from.get(t, [])}), named=n_terms, domain=dom_terms))
    return out


def cpc_queries(cands: list[dict], subclasses: list[str], max_total: int = 2) -> list[dict]:
    """The core candidate's thing forms AND `CPC=<main group>/low` (clause
    last), for the top groups (PatentRiff: "A practitioner might start by
    searching within a relevant CPC class and then use keywords to filter the
    results"). Subclass-level codes are skipped (they match nothing)."""
    if not cands or not subclasses:
        return []
    from .query_gen import cpc_clause
    core = cands[0]
    t_terms, t_from = _union(core.get("elements") or [], "thing", 8)
    things = _group(t_terms)
    if not things:
        return []
    out = []
    for sc in [g for g in subclasses if cpc_clause(g)][:max_total]:
        out.append({"candidate": core.get("id") or "inv1", "kind": "cpc+thing", "query": f"{things} {cpc_clause(sc)}",
                    "facets_used": {"cpc": [sc.split("/")[0]], "thing": t_terms},
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

    # the core candidate gets the budget; other candidates only their natural / wide query
    if per:
        for q in per[0][1]:
            _take(per[0][0], q)
    for cid, qs in per[1:]:
        for q in qs:
            if q["kind"] in ("natural", "wide"):
                _take(cid, q)
    return out


_STOP = set("""a an the of for and or in on to with by from is are be as at into via using based method system
apparatus device toward towards through between over under new novel approach study analysis paper we our its their
""".split())


def title_terms(titles: list[str], top: int = 8) -> list[str]:
    """Most frequent bigrams (then unigrams) across neighbourhood titles —
    the vocabulary the field itself uses, instead of the model's guess."""
    import re
    from collections import Counter
    bi, uni = Counter(), Counter()
    for t in titles:
        words = [w for w in re.findall(r"[a-z][a-z\-]{2,}", (t or "").lower()) if w not in _STOP]
        uni.update(set(words))
        bi.update({f"{a} {b}" for a, b in zip(words, words[1:])})
    out = [g for g, c in bi.most_common() if c >= 2][:top]
    for w, c in uni.most_common():
        if len(out) >= top:
            break
        if c >= 2 and all(w not in g for g in out):
            out.append(w)
    return out[:top]


def terms_query(cand_id: str, terms: list[str]) -> dict | None:
    if not terms:
        return None
    return {"candidate": cand_id, "kind": "neigh_terms", "query": _group(terms), "facets_used": {"neigh_terms": terms}, "elements": []}
