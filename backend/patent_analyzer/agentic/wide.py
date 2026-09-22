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


def broad_terms(cand: dict, neigh_terms: list[str] | None = None, cap: int = 15) -> list[str]:
    """The 'neighbourhood group' (Harry, v6): domain words (preamble thing
    forms) + patent-vocabulary hypernyms (patent facet across elements) +
    the paper neighbourhood's frequent title terms — 8-15 forms that define
    the ~1e5-document field the specific group then narrows."""
    els = cand.get("elements") or []
    out: list[str] = []

    def add(t):
        t = " ".join(str(t).lower().split())
        if t and t not in out:
            out.append(t)
    for t in ((els[0].get("facets") or {}).get("thing") or [])[:4] if els else []:
        add(t)
    for e in els:
        for t in ((e.get("facets") or {}).get("patent") or [])[:2]:
            add(t)
    for t in (neigh_terms or [])[:6]:
        add(t)
    for e in els[1:]:
        if len(out) >= 8:
            break
        for t in ((e.get("facets") or {}).get("thing") or [])[:1]:
            add(t)
    return out[:cap]


def specific_terms(e: dict, cap: int = 4) -> list[str]:
    f = e.get("facets") or {}
    out: list[str] = []
    for t in (f.get("named") or [])[:1] + (f.get("thing") or [])[:3]:
        t = " ".join(str(t).lower().split())
        if t and t not in out:
            out.append(t)
    return out[:cap]


def candidate_queries(cand: dict, max_elements: int = 7, cpc_groups: list[str] | None = None,
                      neigh_terms: list[str] | None = None) -> list[dict]:
    """v6 (Harry, 2026-09-18): query = (specific group OR) AND (neighbourhood group OR).
      element:<id>  — that element's 2-4 specific items ∧ the broad group
      candidate     — items of the two most distinctive elements ∧ the broad group
    plus, when a main group is predicted, ONE CPC variant of the candidate
    query (`… CPC=<group>/low`, clause last) as a peer, not a replacement."""
    els = cand.get("elements") or []
    if not els:
        return []
    broad_list = broad_terms(cand, neigh_terms)
    broad = _group(broad_list, cap=15)
    if not broad:
        return []
    out = []

    def _q(kind, query, elements, **facets):
        return {"kind": kind, "query": query, "facets_used": {k: v for k, v in facets.items() if v}, "elements": elements}

    scored = []
    for e in els[1:1 + max_elements]:
        items = specific_terms(e)
        if not items:
            continue
        out.append(_q(f"element:{e.get('id')}", f"{_group(items)} {broad}", [e.get("id")], specific=items, broad=broad_list))
        # distinctiveness: named present, then longest multi-word item
        scored.append((1 if (e.get("facets") or {}).get("named") else 0, max(len(t) for t in items), e.get("id"), items))
    if scored:
        top2 = sorted(scored, key=lambda x: (-x[0], -x[1]))[:2]
        items = []
        for _, _, _, its in top2:
            for t in its[:2]:
                if t not in items:
                    items.append(t)
        q = f"{_group(items)} {broad}"
        out.append(_q("candidate", q, [x[2] for x in top2], specific=items, broad=broad_list))
        from .query_gen import cpc_clause
        groups = [g for g in (cpc_groups if cpc_groups is not None else cand.get("cpc_pred") or []) if cpc_clause(g)]
        if groups:
            out.append(_q("candidate+cpc", f"{q} {cpc_clause(groups[0])}", [x[2] for x in top2], specific=items, broad=broad_list,
                          cpc=[str(groups[0]).split("/")[0]]))
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


def wide_queries(candidates: list[dict], max_total: int = MAX_QUERIES, neigh_terms: list[str] | None = None) -> list[dict]:
    """Queries over all candidate inventions: the core candidate's element
    queries, its candidate query (+ CPC variant), then the other candidates'
    candidate queries — capped at max_total."""
    per = [(c.get("id") or f"inv{i + 1}", candidate_queries(c, neigh_terms=neigh_terms)) for i, c in enumerate(candidates)]
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
            if q["kind"] == "candidate":
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
