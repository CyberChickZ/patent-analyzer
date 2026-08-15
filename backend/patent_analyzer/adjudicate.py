"""Prior-art determination from verified element coverage.

Deterministic, no LLM. Input is the element list (claim limitations /
checklist criteria) and, per candidate document, the checklist_results that
already went through quote_verify.verify_checklist_results. An element
counts as covered by a document only when the evaluator scored it above
`min_score` AND at least one evidence quote was located in the document
text (verified_quotes non-empty) — an unquoted score is not evidence.

Rules (MPEP 2131 / 2141-2143, PANORAMA App. C.5.3):
  "102"  one document covers every element (allow_missing / min_cover relax it)
  "103"  no single document does, but a greedy union of <= max_combo documents does
  "ALLOW" neither — read as "no blocking reference found among these
          candidates", never as "grantable" (the record is always incomplete).

The label is about blocking risk from the documents at hand; the report
wording follows report_sections' rule (blocking risk only, no grant talk).
"""

from __future__ import annotations

import math


def _criterion(e) -> str:
    if isinstance(e, dict):
        return str(e.get("criterion") or e.get("text") or e.get("id") or "")
    return str(e or "")


def element_covered(item: dict | None, min_score: int = 1, require_quotes: bool = True) -> bool:
    if not isinstance(item, dict):
        return False
    score = item.get("score")
    if score is None:
        score = 2 if item.get("match") else 0
    try:
        score = float(score)
    except (TypeError, ValueError):
        return False
    if score < min_score or score <= 0:
        return False
    if require_quotes:
        return bool(item.get("verified_quotes"))
    return True


def doc_coverage(elements: list, checklist_results: dict | None, min_score: int = 1,
                 require_quotes: bool = True) -> tuple[set[str], set[str]]:
    cr = checklist_results or {}
    covered, missing = set(), set()
    for e in elements:
        c = _criterion(e)
        (covered if element_covered(cr.get(c), min_score, require_quotes) else missing).add(c)
    return covered, missing


def greedy_cover(elements: list[str], doc_sets: list[tuple[str, set[str]]], needed: int,
                 max_combo: int = 3) -> tuple[list[str], set[str]]:
    """Pick up to max_combo documents that together cover >= needed elements.
    Largest single coverage first, then the document adding most new elements."""
    chosen, covered = [], set()
    remaining = list(doc_sets)
    while remaining and len(chosen) < max_combo and len(covered) < needed:
        best = max(remaining, key=lambda ds: (len(ds[1] - covered), len(ds[1])))
        gain = best[1] - covered
        if not gain:
            break
        chosen.append(best[0])
        covered |= best[1]
        remaining = [ds for ds in remaining if ds[0] != best[0]]
    return chosen, covered


def adjudicate(elements: list, docs_results: list[dict], min_cover: float = 1.0,
               allow_missing: int = 0, max_combo: int = 3, min_score: int = 1,
               require_quotes: bool = True, single_partial_103: float | None = None) -> dict:
    """Return {label, basis, risk, reason, n_elements, needed, per_doc_coverage, combo, params}.
    basis: "single" (102) | "combination" (103, union) | "primary_partial" (103, one reference >= single_partial_103) | "none".

    min_cover: fraction of elements a document (or the union) must cover.
    allow_missing: absolute number of elements that may be missing; the
    looser of the two applies. single_partial_103: when set, a best single
    document covering at least this fraction (but not `needed`) yields "103"
    (PANORAMA C.5.3 rule (a), primary reference >= 70%).
    """
    names = [_criterion(e) for e in elements if _criterion(e)]
    n = len(names)
    params = {"min_cover": min_cover, "allow_missing": allow_missing, "max_combo": max_combo,
              "min_score": min_score, "require_quotes": require_quotes,
              "single_partial_103": single_partial_103}
    if n == 0:
        return {"label": "ALLOW", "basis": "none", "risk": "related", "reason": "no elements to compare",
                "n_elements": 0, "needed": 0, "best_single": "", "best_coverage": 0.0,
                "per_doc_coverage": [], "combo": None, "params": params}

    slack = max(allow_missing, int(math.floor(n * (1.0 - min_cover) + 1e-9)))
    needed = max(1, n - slack)

    per_doc = []
    for d in docs_results or []:
        if not isinstance(d, dict):
            continue
        covered, missing = doc_coverage(names, d.get("checklist_results"), min_score, require_quotes)
        per_doc.append({
            "pub_num": d.get("pub_num") or d.get("publication_number") or "",
            "title": d.get("title", ""),
            "covered": [c for c in names if c in covered],
            "missing": [c for c in names if c in missing],
            "coverage": round(len(covered) / n, 4),
            "n_covered": len(covered),
            "source": d.get("source", ""),
        })
    per_doc.sort(key=lambda x: (-x["n_covered"], x["pub_num"]))

    def _key(x):
        return x["pub_num"] or x["title"] or str(id(x))

    best = per_doc[0] if per_doc else None
    best_cov = best["coverage"] if best else 0.0
    combo_docs, combo_covered = greedy_cover(
        names, [(_key(x), set(x["covered"])) for x in per_doc], needed, max_combo)
    combo = {
        "docs": combo_docs,
        "n_covered": len(combo_covered),
        "coverage": round(len(combo_covered) / n, 4),
        "missing": [c for c in names if c not in combo_covered],
    } if combo_docs else None

    if best and best["n_covered"] >= needed:
        label, basis = "102", "single"
        reason = (f"{_key(best)} covers {best['n_covered']}/{n} elements with verified quotes "
                  f"(needed {needed}); a single reference disclosing each element is anticipation (MPEP 2131).")
    elif combo and len(combo_docs) >= 2 and combo["n_covered"] >= needed:
        label, basis = "103", "combination"
        reason = (f"no single document reaches {needed}/{n}; {' + '.join(combo_docs)} together cover "
                  f"{combo['n_covered']}/{n} — every element is known, combination risk under §103 (MPEP 2143 A).")
    elif single_partial_103 is not None and best and best_cov >= single_partial_103:
        label, basis = "103", "primary_partial"
        reason = (f"{_key(best)} alone covers {best['n_covered']}/{n} elements ({best_cov:.0%} >= {single_partial_103:.0%}); "
                  f"the remaining {n - best['n_covered']} would need only a secondary reference or a routine modification "
                  f"— primary-reference combination risk under §103 (MPEP 2143).")
    else:
        label, basis = "ALLOW", "none"
        missing = combo["missing"] if combo else names
        reason = (f"best single coverage {best_cov:.0%}; union of {len(combo_docs)} documents covers "
                  f"{combo['n_covered'] if combo else 0}/{n}; {len(missing)} element(s) have no verified disclosure "
                  f"among the evaluated documents.")

    if label in ("102", "103"):
        risk = "blocking"
    elif best_cov >= 0.5 or (combo and combo["coverage"] >= 0.8):
        risk = "relevant"
    else:
        risk = "related"

    return {"label": label, "basis": basis, "risk": risk, "reason": reason, "n_elements": n, "needed": needed,
            "best_single": _key(best) if best else "", "best_coverage": best_cov,
            "per_doc_coverage": per_doc, "combo": combo, "params": params}


def chart_columns(adj: dict, max_docs: int = 3) -> list[str]:
    """Which documents the claim chart shows, in order: the references the
    rule relied on (best single for "102" / partial "103", the greedy combo
    for a combination "103"), then the next best by coverage, capped."""
    per = adj.get("per_doc_coverage") or []
    keys = [d.get("pub_num") or d.get("title") or "" for d in per if d.get("n_covered", 0) > 0]   # a 0/n column says nothing
    combo = adj.get("combo") or {}
    lead = list(combo.get("docs") or []) if adj.get("label") == "103" and len(combo.get("docs") or []) >= 2 else []
    if adj.get("best_single") and adj["best_single"] not in lead:
        lead.append(adj["best_single"])
    cols = [k for k in lead if k in keys]
    for k in keys:
        if len(cols) >= max_docs:
            break
        if k and k not in cols:
            cols.append(k)
    return cols[:max_docs]


def claim_chart(adj: dict, elements: list, docs_results: list[dict], max_docs: int = 3) -> dict:
    """Examiner-style claim chart from the same evidence the rule used
    (MPEP 2142: the teachings relied upon, per element, with where they
    were found). rows = elements in order; columns = chart_columns(adj);
    cell = {score, n_quotes, n_verified, covered, quote, analysis}."""
    names = [_criterion(e) for e in elements if _criterion(e)]
    p = adj.get("params") or {}
    min_score, require_quotes = int(p.get("min_score", 1)), bool(p.get("require_quotes", True))
    by_key = {}
    for d in docs_results or []:
        if isinstance(d, dict):
            by_key.setdefault(d.get("pub_num") or d.get("publication_number") or d.get("title") or "", d)
    per = {(d.get("pub_num") or d.get("title") or ""): d for d in adj.get("per_doc_coverage") or []}
    cols = chart_columns(adj, max_docs)
    docs = []
    for k in cols:
        src, cov = by_key.get(k) or {}, per.get(k) or {}
        docs.append({"pub_num": src.get("pub_num") or cov.get("pub_num") or "", "title": src.get("title") or cov.get("title") or k,
                     "key": k, "n_covered": cov.get("n_covered", 0), "coverage": cov.get("coverage", 0.0),
                     "url": src.get("patent_link") or src.get("url") or "", "source": src.get("source") or cov.get("source") or ""})
    rows = []
    for name in names:
        cells = []
        for k in cols:
            item = ((by_key.get(k) or {}).get("checklist_results") or {}).get(name) or {}
            score = item.get("score")
            if score is None:
                score = 2 if item.get("match") else 0
            qs = item.get("quote_checks") or []
            vq = item.get("verified_quotes") or []
            cells.append({"score": int(score) if isinstance(score, (int, float)) else 0,
                          "n_quotes": len(qs) if qs else len(item.get("evidence_quotes") or []),
                          "n_verified": len(vq), "covered": element_covered(item, min_score, require_quotes),
                          "quote": (vq[0] if vq else (item.get("evidence_quote") or ""))[:300],
                          "analysis": (item.get("analysis") or "")[:300]})
        rows.append({"element": name, "cells": cells, "covered_by": [cols[i] for i, c in enumerate(cells) if c["covered"]]})
    return {"docs": docs, "rows": rows, "n_elements": len(names),
            "uncovered": [r["element"] for r in rows if not r["covered_by"]]}
