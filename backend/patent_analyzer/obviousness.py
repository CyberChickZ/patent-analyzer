"""The findings a §103 rejection needs that element coverage cannot supply.

MPEP 2143 I.A lists four findings for the combination rationale and this rule
could only ever make the first: the prior art included each element claimed.
Findings (2) and (3), the motivation of 2143.01, the reasonable expectation of
success of 2143.02 and the analogous-art test of 2141.01(a) are judgments about
what a person of ordinary skill would have done, and no set of coverage numbers
contains them.

So one model call produces them — and only them. The model never sees the
label, never proposes one, and cannot make a finding stick: every finding must
quote the reference it comes from, each quote is located in that reference's
own text (quote_verify, the same double threshold the evidence uses), and a
quote that cannot be found makes the finding `not_met`. The rule then decides
the label from the findings. MPEP 2143 I.E is the authority for what happens
when one is missing: "If any of these findings cannot be made, then this
rationale cannot be used to support a conclusion that the claim would have been
obvious."

This keeps the split the pipeline has everywhere else — the model supplies
located evidence, the rule draws the conclusion.
"""

from __future__ import annotations

import os

from .adjudicate import MET, NOT_MET, UNDET

QUOTE_THRESHOLD = float(os.environ.get("OBV_QUOTE_THRESHOLD", "0.9"))

# The one finding allowed to stand without a quote, and why.
#
# Requiring every rationale to be quotable from a reference is the pre-KSR TSM standard, and KSR
# rejected exactly that. MPEP 2143.01: a motivation "may be found explicitly or IMPLICITLY in
# market forces; design incentives; ... and the BACKGROUND KNOWLEDGE, creativity, and common sense
# of the person of ordinary skill" — the last of those has no quote anywhere by definition. Asking
# for one anyway cost measurable accuracy: on PANORAMA's first 100 the gate took macro-F1 from
# .376 to .335 and §103 F1 from .33 to .22, and the model's reason was almost always "neither
# reference provides an EXPRESS suggestion". We had implemented "can articulate a reason" as "can
# quote a reason".
#
# The other constraint is equally real — MPEP 2143, In re Van Os: "Absent some articulated
# rationale, a finding that a combination of prior art would have been 'common sense' or
# 'intuitive' is no different than merely stating the combination 'would have been obvious.'" So
# the reason still has to be stated, and a §103 resting on it is marked everywhere it appears and
# never counts towards a prima facie case (leader, 2026-09-18).
UNEVIDENCED_SOURCE = "background_knowledge"
UNEVIDENCED_NOTE = "rationale asserted from ordinary skill, not evidenced in the references"

# The six places a motivation may legitimately come from (MPEP 2143.01, verbatim):
# "market forces; design incentives; the 'interrelated teachings of multiple patents'; 'any need
#  or problem known in the field of endeavor at the time of invention and addressed by the
#  patent'; and the background knowledge, creativity, and common sense of the person of ordinary
#  skill."
MOTIVATION_SOURCES = ("market_forces", "design_incentives", "interrelated_teachings",
                      "known_need_or_problem", "background_knowledge")

# MPEP 2141.01(a) I: two independent tests, "it is not necessary for a reference to fulfill both".
ANALOGOUS_TESTS = ("same_field_of_endeavor", "reasonably_pertinent_to_the_problem")

# When ONE reference is relied on there is nothing to combine, so 2143.01's "motivation to
# combine" is the wrong question — and asking it anyway is what sank the first scoring run: 92 of
# 98 instances answered "Only a single reference was provided", which is true and is not a defect.
# A single-reference §103 runs on a modification rationale instead (MPEP 2143 I, verbatim):
#   (B) Simple substitution of one known element for another to obtain predictable results
#   (C) Use of known technique to improve similar devices (methods, or products) in the same way
#   (D) Applying a known technique to a known device (method, or product) ready for improvement
#       to yield predictable results
#   (E) "Obvious to try" - choosing from a finite number of identified, predictable solutions
MODIFICATION_RATIONALES = ("B_simple_substitution", "C_known_technique_same_way",
                           "D_known_technique_ready_for_improvement", "E_obvious_to_try")

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "motivation": {"type": "OBJECT", "properties": {
            "found": {"type": "BOOLEAN"},
            "source": {"type": "STRING", "enum": list(MOTIVATION_SOURCES)},
            "doc": {"type": "STRING"}, "quote": {"type": "STRING"}, "reason": {"type": "STRING"}},
            "required": ["found"]},
        "modification": {"type": "OBJECT", "properties": {
            "found": {"type": "BOOLEAN"},
            "rationale": {"type": "STRING", "enum": list(MODIFICATION_RATIONALES)},
            "doc": {"type": "STRING"}, "quote": {"type": "STRING"}, "reason": {"type": "STRING"}},
            "required": ["found"]},
        "combinable_by_known_methods": {"type": "OBJECT", "properties": {
            "found": {"type": "BOOLEAN"}, "doc": {"type": "STRING"},
            "quote": {"type": "STRING"}, "reason": {"type": "STRING"}},
            "required": ["found"]},
        "predictable_results": {"type": "OBJECT", "properties": {
            "found": {"type": "BOOLEAN"}, "doc": {"type": "STRING"},
            "quote": {"type": "STRING"}, "reason": {"type": "STRING"}},
            "required": ["found"]},
        "expectation_of_success": {"type": "OBJECT", "properties": {
            "found": {"type": "BOOLEAN"}, "doc": {"type": "STRING"},
            "quote": {"type": "STRING"}, "reason": {"type": "STRING"}},
            "required": ["found"]},
        "analogous": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "doc": {"type": "STRING"},
            "test": {"type": "STRING", "enum": list(ANALOGOUS_TESTS)},
            "quote": {"type": "STRING"}, "reason": {"type": "STRING"}},
            "required": ["doc", "test"]}},
        "level_of_ordinary_skill": {"type": "OBJECT", "properties": {
            "stated": {"type": "STRING"}, "doc": {"type": "STRING"},
            "quote": {"type": "STRING"}}, "required": ["stated"]},
    },
    "required": ["motivation", "expectation_of_success", "analogous", "level_of_ordinary_skill"],
}


def _verified(quote: str, text: str) -> tuple[bool, float]:
    from .quote_verify import locate_quote
    if not quote or not text:
        return False, 0.0
    try:
        return locate_quote(quote, text, QUOTE_THRESHOLD)
    except Exception:
        return False, 0.0


def verify(raw: dict, texts: dict[str, str], relied: list[str]) -> dict:
    """Turn the model's answer into findings with a status each.

    `texts` maps a reference's key to its full text; `relied` is the set of
    references the rule is leaning on. A finding is `met` only when it is
    claimed AND its quote is located in the reference it names. Anything else
    is `not_met` — never `met` on the model's word alone.
    """
    raw = raw or {}
    out: dict = {"quotes_checked": 0, "quotes_located": 0}

    def one(node: dict | None, name: str) -> dict:
        node = node or {}
        doc, quote = str(node.get("doc") or ""), str(node.get("quote") or "")
        if not node.get("found"):
            return {"status": NOT_MET, "reason": str(node.get("reason") or "not found in the references"),
                    "doc": doc, "quote": "", "located": False, "evidenced": False}
        if name == "motivation" and node.get("source") == UNEVIDENCED_SOURCE and not quote.strip():
            # stated, not quoted — allowed to stand, and labelled wherever it is shown
            return {"status": MET, "evidenced": False, "source": UNEVIDENCED_SOURCE, "doc": doc,
                    "quote": "", "located": False,
                    "reason": f"{UNEVIDENCED_NOTE}: {str(node.get('reason') or '')}"[:400]}
        out["quotes_checked"] += 1
        ok, score = _verified(quote, texts.get(doc, ""))
        out["quotes_located"] += 1 if ok else 0
        return {"status": MET if ok else NOT_MET, "doc": doc, "quote": quote[:400],
                "located": ok, "evidenced": ok, "score": round(score, 3),
                "reason": str(node.get("reason") or "") if ok else
                          f"quote not located in {doc or 'the named reference'} (best {score:.2f})",
                **({"source": node.get("source")} if name == "motivation" else {})}

    out["motivation"] = one(raw.get("motivation"), "motivation")
    out["modification"] = one(raw.get("modification"), "modification")
    out["combinable_by_known_methods"] = one(raw.get("combinable_by_known_methods"), "combinable")
    out["predictable_results"] = one(raw.get("predictable_results"), "predictable")
    out["expectation_of_success"] = one(raw.get("expectation_of_success"), "expectation")

    by_doc = {str(a.get("doc") or ""): a for a in (raw.get("analogous") or []) if isinstance(a, dict)}
    analog = {}
    for k in relied:
        a = by_doc.get(k)
        if not a:
            analog[k] = {"status": NOT_MET, "reason": "no analogous-art finding for this reference",
                         "located": False, "evidenced": False}
            continue
        quote = str(a.get("quote") or "")
        out["quotes_checked"] += 1
        ok, score = _verified(quote, texts.get(k, ""))
        out["quotes_located"] += 1 if ok else 0
        analog[k] = {"status": MET if ok else NOT_MET, "test": a.get("test"), "quote": quote[:400],
                     "located": ok, "evidenced": ok, "score": round(score, 3),
                     "reason": str(a.get("reason") or "") if ok else
                               f"quote not located in {k} (best {score:.2f})"}
    out["analogous"] = analog

    skill = raw.get("level_of_ordinary_skill") or {}
    stated = str(skill.get("stated") or "").strip()
    # MPEP 2141 II (C) is a finding the record has to support, but unlike the others it is a
    # characterisation rather than a teaching, so it is not quote-gated — it is recorded as
    # stated-or-not and never turns a determination on by itself.
    out["level_of_ordinary_skill"] = {"status": MET if stated else UNDET, "stated": stated,
                                      "doc": str(skill.get("doc") or ""),
                                      "quote": str(skill.get("quote") or "")[:400]}
    return out


FULL_FINDINGS = os.environ.get("OBV_FULL_FINDINGS", "1") != "0"


def unevidenced(findings: dict | None) -> list[str]:
    """Findings that were asserted rather than quoted — for the trace, the
    report, and for keeping them out of a prima facie case."""
    return [k for k, v in (findings or {}).items()
            if isinstance(v, dict) and v.get("status") == MET and v.get("evidenced") is False]


def combination_supported(findings: dict | None, relied: list[str]) -> tuple[bool, str]:
    """Whether a §103 stands on these findings, and if not, which one is missing.

    The rationale depends on how many references are relied on, which is the
    correction the first scoring run forced. With two or more it is the
    combination rationale (2143 I.A) and 2143.01's motivation to combine is the
    right question. With ONE there is nothing to combine, and the question is
    whether a known technique would have been applied to it — 2143 I.(B)/(C)/
    (D)/(E). Asking for a motivation to combine there produced "Only a single
    reference was provided" 92 times out of 98, which is a correct answer to
    the wrong question.

    Either way MPEP 2143.02 I also needs a reasonable expectation of success,
    and 2141.01(a) I bars a non-analogous reference outright. With
    OBV_FULL_FINDINGS the two remaining findings of 2143 I.A — (2) the elements
    could have been combined by known methods, each still performing the same
    function, and (3) the results were predictable — are required for a
    combination as well. MPEP 2143 I.E governs a gap: "this rationale cannot be
    used".
    """
    if not findings:
        return False, "no findings were made"
    gaps = []
    multi = len(relied) >= 2
    if multi:
        if findings.get("motivation", {}).get("status") != MET:
            gaps.append(f"motivation to combine (MPEP 2143.01): {findings.get('motivation', {}).get('reason', '')}")
        if FULL_FINDINGS:
            for key, mpep, what in (("combinable_by_known_methods", "2143 I.A (2)",
                                     "the elements could have been combined by known methods, each still "
                                     "performing the same function"),
                                    ("predictable_results", "2143 I.A (3)",
                                     "the results of the combination were predictable")):
                if findings.get(key, {}).get("status") != MET:
                    gaps.append(f"{what} (MPEP {mpep}): {findings.get(key, {}).get('reason', '')}")
    else:
        if findings.get("modification", {}).get("status") != MET:
            gaps.append(f"a rationale for modifying the single reference (MPEP 2143 I.(B)-(E)): "
                        f"{findings.get('modification', {}).get('reason', '')}")
    if findings.get("expectation_of_success", {}).get("status") != MET:
        gaps.append(f"reasonable expectation of success (MPEP 2143.02 I): "
                    f"{findings.get('expectation_of_success', {}).get('reason', '')}")
    an = findings.get("analogous") or {}
    for k in relied:
        if an.get(k, {}).get("status") != MET:
            gaps.append(f"{k} not shown to be analogous art (MPEP 2141.01(a) I): {an.get(k, {}).get('reason', '')}")
    return (not gaps), "; ".join(gaps)
