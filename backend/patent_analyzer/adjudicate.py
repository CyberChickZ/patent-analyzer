"""Prior-art determination from verified element coverage.

Deterministic, no LLM. Input is the element list (claim limitations /
checklist criteria) and, per candidate document, the checklist_results that
already went through quote_verify.verify_checklist_results. An element
counts as covered by a document only when the evaluator scored it above
`min_score` AND at least one evidence quote was located in the document
text (verified_quotes non-empty) — an unquoted score is not evidence.

Labels:
  "102"  one document covers every element (allow_missing / min_cover relax it)
  "103"  no single document does, but a greedy union of <= max_combo documents does
  "ALLOW" neither — read as "no blocking reference found among these
          candidates", never as "grantable" (the record is always incomplete).

Only "102" tracks a statutory test (MPEP 2131: a single reference disclosing
every element, arranged as claimed). **Both "103" branches are coverage
heuristics, not the statutory obviousness test.** MPEP 2141 II requires the
Graham inquiries (scope and content of the prior art, differences, level of
ordinary skill, objective evidence) and MPEP 2143 requires an articulated
rationale from I.(A)-(G) — rationale A alone needs four findings — plus a
motivation to combine (2143.01) and a reasonable expectation of success
(2143.02). This function makes none of them: it computes set coverage. The
70% primary-reference threshold is the PANORAMA benchmark's scoring rule
(App. C.5.3 (a)); MPEP 2143 states no percentage anywhere.

The label is about blocking risk from the documents at hand; the report
wording follows report_sections' rule (blocking risk only, no grant talk).
"""

from __future__ import annotations

import math


def abstract_only(d: dict) -> bool:
    """Was this reference read as an abstract and nothing more?

    Measured on a real job: of 104 delivered papers the full text could be
    fetched for 16 — the rest are behind bot protection or simply not open
    access, and 9 of the 17 examiner-cited papers are not OA at all (N7,
    2026-09-18). An abstract says what a paper is about, not everything it
    discloses, so it cannot carry the §102 finding that ONE reference discloses
    every element. It still counts towards a combination, where it stands for
    one teaching rather than for the whole invention.
    """
    mode = str(d.get("text_mode") or d.get("source") or "").lower()
    return mode.startswith("abstract") or bool(d.get("fulltext_tier") == "abstract_only")


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


# Why a "not covered" cell is not covered. The report and the frontend used to
# each re-derive this from `score`, with their own threshold — the matrix used
# score >= 2 while the determination uses element_covered (min_score=1 AND a
# verified quote) — so one page could say "best single covers 4/5" next to a
# badge reading "1/5 covered". The rule lives here and nowhere else now.
UNCOVERED_NOT_EVALUATED = "not_evaluated"      # the evaluator never produced a row
UNCOVERED_NOT_DISCLOSED = "not_disclosed"      # scored 0 / no match
UNCOVERED_BELOW_SCORE = "below_min_score"      # scored, but under min_score
UNCOVERED_NO_QUOTE = "quote_not_verified"      # scored, but no quote was located in the reference


def cell_reason(item: dict | None, min_score: int = 1, require_quotes: bool = True) -> str:
    """"" when the cell IS covered, else why not — same inputs, same order of
    tests, as element_covered."""
    if element_covered(item, min_score, require_quotes):
        return ""
    if not isinstance(item, dict):
        return UNCOVERED_NOT_EVALUATED
    score = item.get("score")
    if score is None:
        score = 2 if item.get("match") else 0
    try:
        score = float(score)
    except (TypeError, ValueError):
        return UNCOVERED_NOT_EVALUATED
    if score <= 0:
        return UNCOVERED_NOT_DISCLOSED
    if score < min_score:
        return UNCOVERED_BELOW_SCORE
    return UNCOVERED_NO_QUOTE


def coverage_cells(elements: list, checklist_results: dict | None, min_score: int = 1,
                   require_quotes: bool = True) -> dict[str, dict]:
    """One cell per (document, element), decided by element_covered.

    Downstream reads `covered`; it must not recompute it from `score`. The other
    keys are for display only: `score` and `n_verified` say what the evaluator
    produced, `reason` says which test the cell failed.
    """
    cr = checklist_results or {}
    cells: dict[str, dict] = {}
    for e in elements:
        c = _criterion(e)
        if not c:
            continue
        item = cr.get(c)
        item = item if isinstance(item, dict) else None
        score = (item or {}).get("score")
        if score is None and item is not None:
            score = 2 if item.get("match") else 0
        vq = (item or {}).get("verified_quotes") or []
        cells[c] = {
            "covered": element_covered(item, min_score, require_quotes),
            "reason": cell_reason(item, min_score, require_quotes),
            "score": score,
            "n_verified": len(vq) if isinstance(vq, (list, tuple)) else int(bool(vq)),
            "evaluated": item is not None,
        }
    return cells


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
               require_quotes: bool = True, single_partial_103: float | None = None,
               invention_cpc=None, findings: dict | None = None) -> dict:
    """Return {label, basis, risk, reason, n_elements, needed, per_doc_coverage, combo, params}.
    basis: "single" (102) | "combination" (103, union) | "primary_partial" (103, one reference >= single_partial_103) | "none".

    min_cover: fraction of elements a document (or the union) must cover.
    allow_missing: absolute number of elements that may be missing; the
    looser of the two applies. single_partial_103: when set, a best single
    document covering at least this fraction (but not `needed`) yields "103".
    That threshold is a screening heuristic taken from the PANORAMA benchmark's
    scoring rule (App. C.5.3 (a), primary reference >= 70%) — it is not a
    statutory standard and has no MPEP basis; see the module docstring.
    """
    names = [_criterion(e) for e in elements if _criterion(e)]
    n = len(names)
    params = {"min_cover": min_cover, "allow_missing": allow_missing, "max_combo": max_combo,
              "min_score": min_score, "require_quotes": require_quotes,
              "single_partial_103": single_partial_103}
    if n == 0:
        return {"label": "ALLOW", "basis": "none", "risk": "related", "reason": "no elements to compare",
                "n_elements": 0, "needed": 0, "best_single": "", "best_coverage": 0.0,
                "per_doc_coverage": [], "combo": None, "params": params, "rule_trace": [],
                "prima_facie": False}

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
            # the same decision, per element, so nothing downstream has to re-derive it
            "cells": coverage_cells(names, d.get("checklist_results"), min_score, require_quotes),
            "coverage": round(len(covered) / n, 4),
            "n_covered": len(covered),
            "source": d.get("source", ""),
            "abstract_only": abstract_only(d),
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

    def _blocked_103(relied: list[str]) -> str:
        """Empty when the findings support a §103, otherwise why they do not.
        With no findings at all the label stays the coverage screening flag it
        has always been — see the module docstring."""
        if findings is None:
            return ""
        from .obviousness import combination_supported
        ok, gaps = combination_supported(findings, relied)
        return "" if ok else gaps

    anticipating = next((x for x in per_doc if x["n_covered"] >= needed and not x.get("abstract_only")), None)
    if anticipating:
        best = anticipating
        best_cov = best["coverage"]
        label, basis = "102", "single"
        reason = (f"{_key(best)} covers {best['n_covered']}/{n} elements with verified quotes "
                  f"(needed {needed}); a single reference disclosing each element is anticipation (MPEP 2131).")
    elif combo and len(combo_docs) >= 2 and combo["n_covered"] >= needed and not _blocked_103(combo_docs):
        label, basis = "103", "combination"
        reason = (f"no single document reaches {needed}/{n}; {' + '.join(combo_docs)} together cover "
                  f"{combo['n_covered']}/{n}, so every element is disclosed somewhere in the art. That is finding (1) "
                  f"of MPEP 2143 I.A only; whether a person of ordinary skill would have combined these references "
                  f"(2143.01 motivation, 2143.02 reasonable expectation of success) is not determined here. Treat this "
                  f"as a §103 screening flag, not an obviousness conclusion.")
    elif (single_partial_103 is not None and best and best_cov >= single_partial_103
          and not _blocked_103([_key(best)])):
        label, basis = "103", "primary_partial"
        reason = (f"{_key(best)} alone covers {best['n_covered']}/{n} elements ({best_cov:.0%} >= {single_partial_103:.0%}); "
                  f"the remaining {n - best['n_covered']} element(s) are not disclosed by it. Flagged as a §103 screening risk by a "
                  f"coverage heuristic (the PANORAMA benchmark's scoring rule, App. C.5.3 (a)) — a heuristic, not a "
                  f"statutory standard: MPEP 2143 sets no percentage, and no rationale, motivation or expectation of "
                  f"success has been established here.")
    else:
        label, basis = "ALLOW", "none"
        missing = combo["missing"] if combo else names
        abs_only = [x for x in per_doc if x.get("abstract_only") and x["n_covered"] >= needed]
        if abs_only:
            basis = "abstract_only"
            reason = (f"{_key(abs_only[0])} appears to cover {abs_only[0]['n_covered']}/{n} elements, but only its "
                      f"ABSTRACT was read — an abstract states what a document is about, not everything it "
                      f"discloses, so it cannot carry a §102 finding on its own. Obtain the full text and "
                      f"re-run the evidence step before relying on it.")
            risk = "relevant"
            return {"label": label, "basis": basis, "risk": risk, "reason": reason, "n_elements": n,
                    "needed": needed, "best_single": _key(best) if best else "", "best_coverage": best_cov,
                    "per_doc_coverage": per_doc, "combo": combo, "params": params, "findings": findings,
                    "rule_trace": [], "prima_facie": False}
        # only call it a missing finding when the coverage was actually there — otherwise the
        # reason the label is ALLOW is that an element has no disclosure, and saying "no motivation
        # to combine" would name the wrong gap
        covered_enough = bool(combo and len(combo_docs) >= 2 and combo["n_covered"] >= needed) or \
            bool(single_partial_103 is not None and best and best_cov >= single_partial_103)
        gaps = _blocked_103(combo_docs or ([_key(best)] if best else [])) if covered_enough else ""
        if gaps:
            # the elements are all there; what is missing is a finding, and saying which one is
            # the whole point of MPEP 2143's "clear articulation" requirement
            basis = "findings_missing"
            reason = (f"the references together cover {combo['n_covered'] if combo else 0}/{n} elements, but the "
                      f"§103 rationale is not made out — {gaps}. MPEP 2143 I.E: if a finding cannot be made, "
                      f"the rationale cannot be used.")
        else:
            reason = (f"best single coverage {best_cov:.0%}; union of {len(combo_docs)} documents covers "
                      f"{combo['n_covered'] if combo else 0}/{n}; {len(missing)} element(s) have no verified "
                      f"disclosure among the evaluated documents.")

    if label in ("102", "103"):
        risk = "blocking"
    elif best_cov >= 0.5 or (combo and combo["coverage"] >= 0.8):
        risk = "relevant"
    else:
        risk = "related"

    adj = {"label": label, "basis": basis, "risk": risk, "reason": reason, "n_elements": n, "needed": needed,
           "best_single": _key(best) if best else "", "best_coverage": best_cov,
           "per_doc_coverage": per_doc, "combo": combo, "params": params}
    adj["findings"] = findings
    adj["rule_trace"] = rule_trace(adj, elements, docs_results, invention_cpc, findings)
    adj["prima_facie"] = prima_facie(adj["rule_trace"])
    return adj


# ── MPEP rule trace ──────────────────────────────────────────────────────────
# Every requirement an obviousness rejection has to satisfy, with the section
# that imposes it and what this deterministic rule can actually say about it.
# Three statuses, and "not_determined" is the honest one for most of §103:
# the rule reads element coverage, and coverage cannot tell you whether a
# person of ordinary skill would have combined two references.
#
# Quotations are from MPEP [R-01.2024] / [R-08.2017], extracted verbatim and
# recorded with their section numbers in outputs/playbook/mpep.md.

MET, NOT_MET, UNDET = "met", "not_met", "not_determined"


def _subclasses(codes) -> set[str]:
    return {str(c).replace(" ", "")[:4].upper() for c in (codes or []) if len(str(c).replace(" ", "")) >= 4}


def analogous_art(doc: dict, invention_cpc) -> tuple[str, str]:
    """MPEP 2141.01(a) I: a reference may support a §103 rejection only if it is
    analogous art — "(1) ... from the same field of endeavor ... or (2) ...
    reasonably pertinent to the problem faced by the inventor".

    Only the first test has a deterministic proxy here: a shared CPC subclass.
    That is a proxy for "same field of endeavor", not the test itself, and the
    second test needs a judgment about the problem that no classification
    carries. So a shared subclass answers `met` and everything else answers
    `not_determined` — never `not_met`, because failing a proxy for one of two
    independent tests is not a finding that a reference is non-analogous.
    """
    want = _subclasses(invention_cpc)
    have = _subclasses(doc.get("cpc_codes") or (doc.get("raw") or {}).get("cpc"))
    if not want or not have:
        return UNDET, "no classification on the record for one side"
    shared = sorted(want & have)
    if shared:
        return MET, f"same CPC subclass {', '.join(shared)} — a proxy for the same field of endeavor"
    return UNDET, (f"no shared CPC subclass ({', '.join(sorted(have))} vs {', '.join(sorted(want))}); "
                   "whether it is reasonably pertinent to the problem is not decided by classification")


def _t(rid: str, mpep: str, requirement: str, status: str, finding: str) -> dict:
    return {"id": rid, "mpep": mpep, "requirement": requirement, "status": status, "finding": finding}


def _from_finding(findings: dict | None, key: str, absent: str) -> tuple[str, str]:
    """(status, finding text) for a requirement the model was asked to evidence.
    Without findings the answer stays `not_determined` — the absence of a call
    is not a negative finding."""
    f = (findings or {}).get(key)
    if not f:
        return UNDET, absent
    quote = f' — "{f["quote"][:160]}"' if f.get("located") and f.get("quote") else ""
    return f["status"], (f.get("reason") or absent)[:300] + quote


def rule_trace(adj: dict, elements: list, docs_results: list[dict] | None = None,
               invention_cpc=None, findings: dict | None = None) -> list[dict]:
    """What the determination rests on, requirement by requirement, with the
    MPEP section for each — and, for the findings this rule cannot make, that
    they were not made. `prima_facie` in the returned adjudication is False
    whenever any required finding is missing, which for §103 is always: no set
    of coverage numbers establishes a motivation to combine."""
    n = int(adj.get("n_elements") or 0)
    label = adj.get("label")
    combo = adj.get("combo") or {}
    per = adj.get("per_doc_coverage") or []
    by_key = {(d.get("pub_num") or d.get("publication_number") or d.get("title") or ""): d
              for d in (docs_results or []) if isinstance(d, dict)}
    relied = list(combo.get("docs") or []) if label == "103" and adj.get("basis") == "combination" else \
        ([adj["best_single"]] if adj.get("best_single") else [])
    missing = [c for c in (combo.get("missing") if combo else None) or []]

    out = [
        _t("graham_a", "2141 II (A)", "Determining the scope and content of the prior art", MET,
           f"{len(per)} references evaluated; each element counted as disclosed only where a verbatim "
           f"quote was located in that reference's text"),
        _t("graham_b", "2141 II (B)", "Ascertaining the differences between the claimed invention and the prior art",
           MET, (f"{len(missing)} of {n} elements have no verified disclosure: " + "; ".join(m[:60] for m in missing[:4]))
           if missing else f"all {n} elements are disclosed across the references relied on"),
        _t("graham_c", "2141 II (C)", "Resolving the level of ordinary skill in the pertinent art",
           *(((findings or {}).get("level_of_ordinary_skill") or {}).get("status") == MET
             and (MET, ((findings or {})["level_of_ordinary_skill"]["stated"])[:300])
             or (UNDET, "no level of ordinary skill was resolved — nothing on this record establishes one"))),
        _t("graham_objective", "2141 II", "Evaluating objective evidence (commercial success, long-felt need, "
           "failure of others, unexpected results)", UNDET,
           "no objective evidence is before the system; an unfiled invention has no prosecution record"),
        _t("hindsight", "2142", "Knowledge of applicant's disclosure must be put aside; the conclusion rests on "
           "facts gleaned from the prior art", MET,
           "the rule reads only each reference's own coverage and quotes; it never reads the invention back into them"),
    ]

    if label == "102":
        out.append(_t("anticipation", "2131", "A single reference discloses each and every element of the claim",
                      MET, str(adj.get("reason", ""))))
        out.append(_t("analogous_not_required", "2131.05", "Non-analogous art is not germane to a §102 rejection",
                      MET, "no analogous-art finding is needed for anticipation"))
        return out

    if label != "103":
        out.append(_t("no_rejection", "2142", "A prima facie case must be supported by evidence", NOT_MET,
                      str(adj.get("reason", ""))))
        return out

    an = (findings or {}).get("analogous") or {}
    for k in relied:
        f = an.get(k)
        if f:                                    # a located quote beats the classification proxy
            st, why = f["status"], (f.get("reason") or "")[:300] + (f' — "{f["quote"][:120]}"' if f.get("located") else "")
        else:
            st, why = analogous_art(by_key.get(k) or {}, invention_cpc)
        out.append(_t(f"analogous:{k}", "2141.01(a) I",
                      f"{k} must be analogous art: same field of endeavor, or reasonably pertinent to the "
                      f"problem faced by the inventor", st, why))
    covered_all = not missing
    out += [
        _t("rationale_a_1", "2143 I.A (1)",
           "A finding that the prior art included each element claimed, though not necessarily in a single "
           "reference, the only difference being the lack of actual combination",
           MET if covered_all else NOT_MET,
           f"{combo.get('n_covered', adj.get('best_coverage', 0) and '')}/{n} elements disclosed across "
           f"{len(relied)} reference(s)" if covered_all else
           f"{len(missing)} element(s) are disclosed by no reference, so finding (1) is not made"),
        _t("rationale_a_2", "2143 I.A (2)",
           "A finding that one of ordinary skill could have combined the elements by known methods, each "
           "element merely performing the same function as it does separately", UNDET,
           "element coverage does not show how the references would be combined"),
        _t("rationale_a_3", "2143 I.A (3)",
           "A finding that one of ordinary skill would have recognised the results of the combination were "
           "predictable", UNDET, "predictability of the combined result is not a coverage question"),
        _t("motivation", "2143.01",
           "A motivation to combine, explicit or implicit: market forces, design incentives, the interrelated "
           "teachings of the references, a known need or problem, or the skilled person's background knowledge",
           *_from_finding(findings, "motivation", "no reason to combine was found on this record")),
        _t("expectation", "2143.02 I",
           "A reasonable expectation of success, in addition to a reason to combine",
           *_from_finding(findings, "expectation_of_success", "not established")),
        _t("articulation", "2143",
           "Clear articulation of the reason why the claimed invention would have been obvious; \"absent some "
           "articulated rationale\" a combination being obvious is not a reason", UNDET,
           "the report's §103 paragraph is written from this rule's output and states the same gaps"),
    ]
    return out


def prima_facie(trace: list[dict]) -> bool:
    """MPEP 2142: the rejection must be a prima facie case supported by
    evidence. Every requirement has to be `met` — an undetermined finding is
    not a finding, and 2143 I.E says in terms that if any of a rationale's
    findings "cannot be made, then this rationale cannot be used"."""
    return bool(trace) and all(t["status"] == MET for t in trace)


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
                     "url": src.get("patent_link") or src.get("url") or "", "source": src.get("source") or cov.get("source") or "",
                     # the chart has to say which columns were read as an abstract: a blank cell
                     # under one of those means "not in the abstract", not "not disclosed"
                     "abstract_only": bool(cov.get("abstract_only")) or abstract_only(src)})
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
