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
