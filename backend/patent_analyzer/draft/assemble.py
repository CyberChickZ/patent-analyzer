"""Independent claims rebuilt from the extraction elements (leader_draft §2.1).

The unit of a draft claim is the element, not A2's free-text claim string:
e0 is the preamble, every other supported element is one limitation carrying
its own `basis` (element id + verbatim evidence_quote + evidence_loc, 37 CFR
1.75(d)(1)). The LLM only rewords; `verify_wording` is the hard boundary —
the reworded limitations are re-cut with nodes.claim_mode._parse_claim_limitations
(same count) and each must stay within cosine >= WORDING_TAU of the element
text, else that limitation keeps the rule text.
"""

from __future__ import annotations

import re

WORDING_TAU = 0.85
_PREAMBLE_TAIL = re.compile(r"[\s,;:]*(?:comprising|including|consisting of|the method comprising|the system comprising)?[\s:]*$", re.I)
_WHEREIN = re.compile(r"^(?:wherein|where|in which|such that|whereby)\b", re.I)
_METHOD_HEAD = re.compile(r"^(an?)\s+(?:computer[- ]implemented\s+)?(?:method|process)\s+(?:of|for)\b", re.I)
_SYSTEM_HEAD = re.compile(r"^(an?)\s+(.*?)\b(?:system|apparatus|device|assembly|circuit)\b\s*(?:for|configured to|adapted to|that)?\s*(.*)$", re.I)
_VOWELS = set("aeiou")


_TRAIL_CONJ = re.compile(r"[\s;.,]*\b(?:and|or)\s*[;.,]*$", re.I)


def clean_text(t: str) -> str:
    """One line, no trailing punctuation and no trailing conjunction — the
    renderer supplies the ';' / '; and' between limitations (37 CFR 1.75(i))."""
    t = " ".join((t or "").split()).strip(" ;.,")
    return _TRAIL_CONJ.sub("", t).strip(" ;.,") or t


def _infinitive(word: str) -> str:
    """receiving -> receive, stopping -> stop, training -> train (fallback wording only)."""
    w = word
    if not w.lower().endswith("ing") or len(w) <= 4:
        return w
    base = w[:-3]
    if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in _VOWELS and base[-1] not in "sl":
        return base[:-1]
    if len(base) >= 3 and base[-1] not in _VOWELS and base[-1] not in "wxy" and base[-2] in _VOWELS and base[-3] not in _VOWELS:
        return base + "e"
    if base.endswith("v") or base.endswith("z") or base.endswith("u"):
        return base + "e"
    return base


def gerund_to_infinitive(text: str) -> str:
    words = clean_text(text).split(" ", 1)
    if not words:
        return text
    m = re.match(r"^([A-Za-z\-]+)(.*)$", words[0])
    if not m:
        return clean_text(text)
    head = _infinitive(m.group(1)) + m.group(2)
    head = head[0].lower() + head[1:] if head else head
    return head + (" " + words[1] if len(words) > 1 else "")


def mirror_preamble(e0_text: str, form: str) -> str:
    """The preamble subject in the other statutory form (fallback wording)."""
    base = _PREAMBLE_TAIL.sub("", clean_text(e0_text))
    if form == "system":
        if _METHOD_HEAD.match(base):
            return _METHOD_HEAD.sub(lambda m: f"{m.group(1)} system for", base, count=1)
        return base
    m = _SYSTEM_HEAD.match(base)
    if m and not _METHOD_HEAD.match(base):
        rest = clean_text(m.group(3))
        subject = clean_text(m.group(2))
        if rest:
            return f"A method of {rest}"
        return f"A method of operating {m.group(1).lower()} {subject} {'system' if 'system' in base.lower() else 'apparatus'}".replace("  ", " ")
    return base


def preamble_text(e0_text: str, form: str, primary_form: str) -> str:
    base = _PREAMBLE_TAIL.sub("", clean_text(e0_text)) if form == primary_form else mirror_preamble(e0_text, form)
    return base + ", comprising:"


def limitation_text(element: dict, form: str, primary_form: str) -> str:
    """Rule wording of one element in the given form (step / structure / condition / parameter)."""
    text = clean_text(element.get("text", ""))
    kind = (element.get("kind") or "step").lower()
    if kind in ("condition", "parameter"):
        return text if _WHEREIN.match(text) else "wherein " + text
    if form == primary_form:
        return text
    if form == "system" and kind == "step":
        return "one or more processors configured to " + gerund_to_infinitive(text)
    if form == "method" and kind == "structure":
        return "providing " + text
    return text


def basis_of(element: dict) -> dict:
    return {"element_id": element.get("id", ""), "evidence_quote": element.get("evidence_quote", ""),
            "evidence_loc": element.get("evidence_loc")}


def assemble_independent(cand: dict, form: str, claim_no: int | None = None) -> dict:
    """Claim dict from the candidate's supported elements: e0 -> preamble, the
    rest -> one limitation each, in order, each with its basis. Unsupported
    elements (no located quote) never enter the draft."""
    primary = cand.get("primary_form") or "method"
    els = [e for e in cand.get("elements") or [] if not e.get("unsupported") and e.get("text")]
    if not els:
        return {"no": claim_no, "form": form, "depends_on": None, "preamble": "", "limitations": []}
    e0, rest = els[0], els[1:]
    pre = f"c{claim_no}" if claim_no else "c"
    lims = []
    for i, e in enumerate(rest, 1):
        lims.append({"lid": f"{pre}.l{i}", "text": limitation_text(e, form, primary), "kind": (e.get("kind") or "step").lower(),
                     "origin": "element", "basis": [basis_of(e)], "coverage": {}, "flags": []})
    return {"no": claim_no, "form": form, "depends_on": None, "preamble": preamble_text(e0.get("text", ""), form, primary),
            "preamble_basis": [basis_of(e0)], "limitations": lims}


def dependent_claim(parent: dict, limitation: dict, claim_no: int | None = None) -> dict:
    """One further limitation on `parent` (MPEP 608.01(n): every limitation of
    the parent plus one more; single dependency only)."""
    form = parent.get("form") or "method"
    noun = "method" if form == "method" else "system"
    text = clean_text(limitation.get("text", ""))
    bridge = "wherein" if _WHEREIN.match(text) or limitation.get("kind") in ("condition", "parameter") else "further comprising"
    if bridge == "wherein" and _WHEREIN.match(text):
        text = _WHEREIN.sub("", text).strip()
    lim = {**limitation, "lid": f"c{claim_no}.l1" if claim_no else limitation.get("lid", "c.l1"), "text": text}
    return {"no": claim_no, "form": form, "depends_on": parent.get("no"),
            "preamble": f"The {noun} of claim {parent.get('no')}, {bridge}", "limitations": [lim]}


def render_claim(claim: dict) -> str:
    """37 CFR 1.75(i): each limitation on its own indented line."""
    no = claim.get("no")
    head = f"{no}. " if no is not None else ""
    lims = [clean_text(l.get("text", "")) for l in claim.get("limitations") or []]
    lims = [l for l in lims if l]
    if claim.get("depends_on") is not None:
        return head + clean_text(claim.get("preamble", "")) + (" " + lims[0] if lims else "") + "."
    if not lims:
        return head + clean_text(claim.get("preamble", "")) + "."
    if len(lims) == 1:
        return head + claim.get("preamble", "") + "\n  " + lims[0] + "."
    body = ";\n  ".join(lims[:-1]) + "; and\n  " + lims[-1]
    return head + claim.get("preamble", "") + "\n  " + body + "."


def recut(limitations: list[str]) -> list[str]:
    """What _parse_claim_limitations makes of the limitations joined as one claim."""
    from nodes.claim_mode import _parse_claim_limitations
    body = "; ".join(clean_text(l) for l in limitations if clean_text(l))
    if not body:
        return []
    return _parse_claim_limitations("A method, comprising: " + body + ".")["limitations"]


def difflib_similarity(a: list[str], b: list[str]) -> list[float]:
    """Pairwise similarity without an encoder (fallback when embedding is unavailable)."""
    from difflib import SequenceMatcher
    from patent_analyzer.quote_verify import normalize
    return [SequenceMatcher(None, normalize(x), normalize(y), autojunk=False).ratio() for x, y in zip(a, b)]


def embed_similarity(a: list[str], b: list[str]) -> list[float]:
    """Pairwise te005 cosine (SEMANTIC_SIMILARITY), same encoder as evals/extraction_eval.embed."""
    import numpy as np
    from patent_analyzer.encoders import embed_vertex
    if not a:
        return []
    va = np.asarray(embed_vertex(list(a), "SEMANTIC_SIMILARITY"), dtype=np.float32)
    vb = np.asarray(embed_vertex(list(b), "SEMANTIC_SIMILARITY"), dtype=np.float32)
    va /= np.linalg.norm(va, axis=1, keepdims=True) + 1e-9
    vb /= np.linalg.norm(vb, axis=1, keepdims=True) + 1e-9
    return [float((va[i] * vb[i]).sum()) for i in range(len(a))]


def verify_wording(rule_texts: list[str], llm_texts: list[str] | None, similarity=None, tau: float = WORDING_TAU) -> tuple[list[str], dict]:
    """The wording invariant. Returns (accepted texts, report). A reworded
    limitation is kept only when (a) the LLM returned exactly one text per
    limitation, (b) re-cutting it with _parse_claim_limitations yields exactly
    one limitation (no hidden split on ';' / 'wherein' / dash lists), and
    (c) similarity(rule, llm) >= tau. Anything else keeps the rule text."""
    n = len(rule_texts)
    report = {"n": n, "accepted": 0, "count_ok": True, "per_limitation": []}
    llm_texts = [clean_text(t) for t in (llm_texts or [])]
    if len(llm_texts) != n:
        report["count_ok"] = False
        report["reason"] = f"LLM returned {len(llm_texts)} limitations for {n}"
        report["per_limitation"] = [{"accepted": False, "reason": "count"} for _ in rule_texts]
        return list(rule_texts), report
    similarity = similarity or embed_similarity
    sims = similarity(rule_texts, llm_texts) if n else []
    out = []
    for i, (rule, llm) in enumerate(zip(rule_texts, llm_texts)):
        entry = {"sim": round(float(sims[i]), 4) if i < len(sims) else None}
        if not llm:
            entry.update(accepted=False, reason="empty")
        elif len(recut([llm])) != 1:
            entry.update(accepted=False, reason="splits on recut")
        elif entry["sim"] is None or entry["sim"] < tau:
            entry.update(accepted=False, reason=f"cosine < {tau}")
        else:
            entry.update(accepted=True)
        out.append(llm if entry["accepted"] else rule)
        report["per_limitation"].append(entry)
    report["accepted"] = sum(1 for e in report["per_limitation"] if e["accepted"])
    return out, report


def apply_wording(claim: dict, texts: list[str], report: dict) -> dict:
    """Write the accepted texts back into the claim's limitations (in place)."""
    for l, t, r in zip(claim.get("limitations") or [], texts, report.get("per_limitation") or []):
        l["rule_text"] = l.get("rule_text") or l.get("text")
        l["text"] = t
        l["wording"] = r
    return claim
