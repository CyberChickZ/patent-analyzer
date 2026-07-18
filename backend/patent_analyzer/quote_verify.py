"""Verify that an LLM evidence_quote actually appears in the source text.

Whitespace/punctuation/case are normalized on both sides, then the best
local alignment of the quote inside the document is scored with
difflib. A quote that cannot be located is treated as fabricated and the
associated score is downgraded to 0.
"""

import re
from difflib import SequenceMatcher

_NORM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", _NORM.sub(" ", (text or "").lower())).strip()


def locate_quote(quote: str, document: str, threshold: float = 0.9) -> tuple[bool, float]:
    """Return (found, similarity). Exact normalized substring short-circuits;
    otherwise slide a window of the quote's length over the document."""
    q, d = normalize(quote), normalize(document)
    if len(q) < 10 or not d:
        return False, 0.0
    if q in d:
        return True, 1.0
    n = len(q)
    step = max(1, n // 4)
    best = 0.0
    sm = SequenceMatcher(None, autojunk=False)
    sm.set_seq2(q)
    for start in range(0, max(1, len(d) - n + step), step):
        sm.set_seq1(d[start:start + n])
        if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
            continue
        ratio = sm.ratio()
        if ratio > best:
            best = ratio
            if best >= 0.995:
                break
    return best >= threshold, round(best, 4)


def verify_checklist_results(checklist_results: dict, document: str,
                             threshold: float = 0.9) -> dict:
    """Downgrade unverifiable quotes in place; return verification stats."""
    stats = {"scored": 0, "with_quote": 0, "verified": 0, "downgraded": 0}
    for item in checklist_results.values():
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if score is None:
            score = 2 if item.get("match") else 0
        if score <= 0:
            continue
        stats["scored"] += 1
        quote = (item.get("evidence_quote") or "").strip()
        if quote:
            stats["with_quote"] += 1
            found, sim = locate_quote(quote, document, threshold)
            item["quote_similarity"] = sim
            if found:
                stats["verified"] += 1
                continue
        item["score"], item["match"] = 0, False
        item["quote_unverified"] = True
        stats["downgraded"] += 1
    return stats
