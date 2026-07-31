"""Verify that an LLM evidence_quote actually appears in the source text.

Both sides are NFKC-normalized, lowered and reduced to Unicode word
characters (CJK kept, spaces inside CJK runs dropped), then the best
local alignment of the quote inside the document is scored with
difflib. A quote that cannot be located is treated as fabricated and the
associated score is downgraded to 0.
"""

import re
import unicodedata
from difflib import SequenceMatcher

CJK = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
_NONWORD = re.compile(r"[\W_]+")
_WS = re.compile(r"\s+")
_CJK_SPACE = re.compile(rf"(?<=[{CJK}]) +| +(?=[{CJK}])")


def normalize(text: str) -> str:
    """NFKC (ligatures, full-width punctuation), lower, Unicode word chars only
    (CJK survives), no spaces next to CJK (PDF line breaks split sentences)."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _WS.sub(" ", _NONWORD.sub(" ", t)).strip()
    return _CJK_SPACE.sub("", t)


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


def _quotes_of(item: dict) -> list[str]:
    raw = item.get("evidence_quotes")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raw = [item["evidence_quote"]] if item.get("evidence_quote") else []
    seen, out = set(), []
    for q in raw:
        q = str(q or "").strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:5]


def verify_checklist_results(checklist_results: dict, document: str,
                             threshold: float = 0.9) -> dict:
    """Verify every quote (evidence_quotes list, or the single evidence_quote)
    against the document: char-level locate_quote OR the dual span/bigram
    rule. Keeps the verified quotes, downgrades the score to 0 when none
    survive. Returns verification stats."""
    from .quote_dual import DocIndex, verify_quote_dual
    idx = DocIndex(document) if document else None
    stats = {"scored": 0, "with_quote": 0, "quotes": 0, "verified": 0, "downgraded": 0}
    for item in checklist_results.values():
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if score is None:
            score = 2 if item.get("match") else 0
        if score <= 0:
            continue
        stats["scored"] += 1
        quotes = _quotes_of(item)
        verified, checks = [], []
        for q in quotes:
            stats["quotes"] += 1
            found, sim = locate_quote(q, document, threshold)
            dual, sr, br = verify_quote_dual(q, idx) if idx is not None else (False, 0.0, 0.0)
            ok = found or dual
            checks.append({"quote": q, "verified": ok, "sim": sim, "span": sr, "bigram": br})
            if ok:
                verified.append(q)
                stats["verified"] += 1
        if quotes:
            stats["with_quote"] += 1
        item["evidence_quotes"] = quotes
        item["quote_checks"] = checks
        item["evidence_quote"] = verified[0] if verified else (quotes[0] if quotes else "")
        item["verified_quotes"] = verified
        if verified:
            continue
        item["score"], item["match"] = 0, False
        item["quote_unverified"] = True
        stats["downgraded"] += 1
    return stats
