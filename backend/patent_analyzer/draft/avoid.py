"""Avoidance from the evidence matrix (leader_draft §2.2). Pure functions.

covered_sets        claim chart column -> element ids it discloses (score >= 1 with
                    a located quote, the same rule adjudicate.element_covered used)
candidate_limitations  the pool a narrowing limitation may come from: A2's
                    dependent_hints, the component / application candidates'
                    supported elements (refinements are added by the node after
                    the LLM copied their quotes). Nothing here is invented.
plan                the decision table on adjudication.label / basis:
                    ALLOW -> as_is; 102 / 103-combination -> narrowed (one pool
                    limitation none of the charted references discloses goes
                    into the independent claim) or unresolved when every pool
                    limitation is disclosed; 103-primary_partial -> as_is plus
                    the first undisclosed limitation as dependent claim 2;
                    no scoring_report -> no_prior_art.
dependent_claims    MPEP 608.01(n) III: a dependent limitation must further
                    limit (cosine < .85 to every parent limitation, no
                    'instead of' / 'without the'), no duplicates, at most MAX_DEPENDENTS.
"""

from __future__ import annotations

import os
import re

# 9, so that the mirror-form pair comes to exactly 20 claims. 37 CFR 1.16(i) charges an
# excess-claims fee "in excess of 20" — the node writes the same dependents onto the mirror
# independent claim, so N per parent is 2 + 2N in total: N=9 is 20 and free, N=10 is 22 and $400
# (undiscounted). The previous 8 came from a commit message that said only "dedupe, cap 8" and
# left a free slot unused. Real practice sits higher still: 244,179 US B1/B2 grants from 2010 on
# have a median of 14 dependents and only 23.3% have 8 or fewer, while 83.3% keep the total at 20
# or under — which matches USPTO's own FY2023 figure that 83% of applications contained no excess
# claims (N2, outputs/leader_dependent_count.md; leader 2026-09-18).
#
# The cap belongs on the TOTAL, not on the count per parent: 2 + 2N <= 20 only holds while there
# are exactly two independent claims. Once a candidate emits three forms it is 3 + 3N <= 20, i.e.
# N <= 5, and a hard-coded 9 would walk through the fee threshold without saying anything.
MAX_DEPENDENTS = 9
FURTHER_TAU = 0.85


def max_dependents() -> int:
    """Dependent claims kept per independent claim (DRAFT_MAX_DEPENDENTS).

    Read per call, not at import: the eval sweeps set it per run. The default 8
    is a claim-count budget, not a drafting rule — the node writes the same
    dependents onto the mirror-form independent claim, so a set of N per parent
    is 2 + 2N claims in total, and 37 CFR 1.16(i) charges an excess-claims fee
    above 20 total (see outputs/leader_dependent_count.md)."""
    raw = os.environ.get("DRAFT_MAX_DEPENDENTS")
    if not raw:
        return MAX_DEPENDENTS
    try:
        return max(0, int(raw))
    except ValueError:
        return MAX_DEPENDENTS
_NEGATION = re.compile(r"\b(instead of|in place of|without the|omitting|rather than|replacing the|excluding the)\b", re.I)


def criterion_to_element(checklist: list[dict] | None) -> dict[str, str]:
    return {c.get("criterion", ""): c.get("id", "") for c in checklist or [] if c.get("criterion")}


def covered_sets(chart: dict | None, checklist: list[dict] | None) -> dict[str, set[str]]:
    """{chart column key: element ids that column discloses}."""
    c2e = criterion_to_element(checklist)
    out: dict[str, set[str]] = {}
    for i, d in enumerate((chart or {}).get("docs") or []):
        key = d.get("key") or d.get("pub_num") or d.get("title") or f"doc{i}"
        out[key] = {c2e.get(r["element"], r["element"]) for r in chart.get("rows") or [] if r["cells"][i].get("covered")}
    return out


def uncovered_elements(chart: dict | None, checklist: list[dict] | None) -> list[str]:
    c2e = criterion_to_element(checklist)
    return [c2e.get(x, x) for x in (chart or {}).get("uncovered") or []]


def candidate_limitations(extraction: dict | None, core: dict | None) -> list[dict]:
    """The pool, in priority order: dependent_hints (quote to be copied by the
    LLM), then the other candidates' supported elements (quote already located)."""
    pool = []
    core_id = (core or {}).get("id")
    for i, h in enumerate((core or {}).get("dependent_hints") or []):
        h = " ".join(str(h or "").split())
        if h:
            pool.append({"pid": f"hint{i}", "origin": "dependent_hint", "source": f"dependent_hint[{i}]", "text": h,
                         "element_id": f"hint[{i}]", "evidence_quote": "", "evidence_loc": None, "kind": ""})
    for c in (extraction or {}).get("candidate_inventions") or []:
        if not c.get("id") or c.get("id") == core_id or c.get("level") not in ("component", "application"):
            continue
        for e in (c.get("elements") or [])[1:]:
            if e.get("unsupported") or not e.get("text") or not e.get("evidence_quote"):
                continue
            pool.append({"pid": e["id"], "origin": "component_element", "source": f"{c['id']}[{c.get('level')}].{e['id']}",
                         "text": e["text"], "element_id": e["id"], "evidence_quote": e["evidence_quote"],
                         "evidence_loc": e.get("evidence_loc"), "kind": e.get("kind") or ""})
    return pool


def plan(adj: dict | None, chart: dict | None, checklist: list[dict] | None, pool: list[dict],
         disclosed_by: dict[str, set[str]] | None, has_prior_art: bool = True) -> dict:
    """Decision table (see module doc). `disclosed_by` = {pid: chart columns that
    disclose that pool limitation} from the full-text re-evaluation; a pid
    missing there counts as undisclosed only when the check ran (chart present)."""
    disclosed_by = disclosed_by or {}
    usable = [p for p in pool if not p.get("dropped")]
    cols = [d.get("key") or d.get("pub_num") or d.get("title") or "" for d in (chart or {}).get("docs") or []]
    covered = covered_sets(chart, checklist)
    uncovered = uncovered_elements(chart, checklist)
    out = {"strategy": "as_is", "label": (adj or {}).get("label"), "basis": (adj or {}).get("basis"),
           "chart_docs": cols, "covered_set": sorted(set().union(*covered.values()) if covered else set()),
           "uncovered_elements": uncovered, "independent_add": None, "first_dependent": None,
           "dependents": [], "candidates_tried": [], "reason": ""}

    def undisclosed(p) -> bool:
        return not (disclosed_by.get(p["pid"], set()) & set(cols))

    tried = []
    for p in usable:
        tried.append({"pid": p["pid"], "source": p["source"], "disclosed_by": sorted(disclosed_by.get(p["pid"], set()) & set(cols)),
                      "chosen": False})
    dropped = [{"pid": p["pid"], "source": p["source"], "dropped": p["dropped"], "chosen": False} for p in pool if p.get("dropped")]
    out["candidates_tried"] = tried + dropped
    ordered = [p for p in usable if undisclosed(p)] + [p for p in usable if not undisclosed(p)]
    out["dependents"] = [p["pid"] for p in ordered]

    if not has_prior_art or not adj or not adj.get("n_elements"):
        out["strategy"] = "no_prior_art"
        out["reason"] = "no evaluated prior art: independent claims as extracted, dependent claims from the pool; coverage unknown."
        return out

    label, basis = adj.get("label"), adj.get("basis")
    n = adj.get("n_elements", 0)
    if label in ("102", "103") and basis in ("single", "combination"):
        must = [adj.get("best_single")] if basis == "single" else list((adj.get("combo") or {}).get("docs") or [])
        pick = next((p for p in usable if undisclosed(p)), None)
        if pick is None:
            out["strategy"] = "unresolved"
            out["reason"] = (f"{' + '.join(must)} disclose{'s' if len(must) == 1 else ''} every element ({n}/{n}); none of the "
                             f"{len(usable)} candidate limitations from your document ({', '.join(p['source'] for p in usable) or 'none'}) "
                             f"is undisclosed by the {len(cols)} charted references. The independent claim is left as extracted; "
                             f"a distinguishing limitation needs input from the inventor.")
            return out
        out["strategy"] = "narrowed"
        out["independent_add"] = pick["pid"]
        for t in out["candidates_tried"]:
            if t["pid"] == pick["pid"]:
                t["chosen"] = True
        out["dependents"] = [pid for pid in out["dependents"] if pid != pick["pid"]]
        out["reason"] = (f"{' + '.join(must)} disclose{'s' if len(must) == 1 else ''} every element ({n}/{n}, §{label}); the independent "
                         f"claim adds a limitation from {pick['source']} that none of the {len(cols)} charted references "
                         f"({', '.join(cols)}) discloses.")
        return out
    if label == "103" and basis == "primary_partial":
        best = adj.get("best_single")
        missing = next((d.get("missing") for d in adj.get("per_doc_coverage") or [] if (d.get("pub_num") or d.get("title")) == best), None)
        c2e = criterion_to_element(checklist)
        out["uncovered_elements"] = [c2e.get(x, x) for x in missing or []] or uncovered
        pick = next((p for p in usable if undisclosed(p)), None)
        out["first_dependent"] = pick["pid"] if pick else None
        if pick:
            for t in out["candidates_tried"]:
                if t["pid"] == pick["pid"]:
                    t["chosen"] = True
            out["dependents"] = [pick["pid"]] + [pid for pid in out["dependents"] if pid != pick["pid"]]
        out["reason"] = (f"{best} alone discloses {adj.get('best_coverage', 0):.0%} of the elements (§103 primary-reference pattern); "
                         f"the independent claim is kept as extracted with {len(out['uncovered_elements'])} distinguishing element(s) "
                         f"({', '.join(out['uncovered_elements'])})"
                         + (f"; claim 2 adds a limitation from {pick['source']} that none of the charted references discloses." if pick
                            else "; no undisclosed pool limitation was found for a first dependent claim."))
        return out
    out["strategy"] = "as_is"
    out["reason"] = (f"no single reference or combination discloses every element; {len(uncovered)} element(s) have no verified "
                     f"disclosure among the charted references ({', '.join(uncovered) or '—'}) and are marked distinguishing.")
    return out


def dependent_claims(items: list[dict], parent: dict, similarity, max_n: int | None = None,
                     tau: float = FURTHER_TAU) -> tuple[list[dict], list[dict]]:
    """Keep the pool items that further limit `parent`: no negation of a parent
    element, cosine < tau to every parent limitation and to every dependent
    already kept; at most max_n (default: max_dependents()). Returns
    (kept, rejected-with-reason)."""
    max_n = max_dependents() if max_n is None else max_n
    parent_texts = [l.get("text", "") for l in parent.get("limitations") or []]
    kept, rejected = [], []
    for it in items:
        text = it.get("text", "")
        if len(kept) >= max_n:
            rejected.append({**it, "rejected": "over_limit"})
            continue
        if _NEGATION.search(text):
            rejected.append({**it, "rejected": "not_further_limiting", "note": "removes or replaces a parent element (MPEP 608.01(n) III)"})
            continue
        against = parent_texts + [k.get("text", "") for k in kept]
        sims = similarity([text] * len(against), against) if against else []
        hi = max(sims) if sims else 0.0
        if hi >= tau:
            j = sims.index(hi)
            reason = "not_further_limiting" if j < len(parent_texts) else "duplicate"
            rejected.append({**it, "rejected": reason, "sim": round(float(hi), 4),
                             "note": f"{'restates parent limitation' if reason == 'not_further_limiting' else 'duplicates dependent'} '{against[j][:60]}'"})
            continue
        kept.append({**it, "further_limiting_sim": round(float(hi), 4)})
    return kept, rejected
