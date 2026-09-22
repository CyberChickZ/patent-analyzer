"""Verify that an LLM evidence_quote actually appears in the source text.

Both sides are NFKC-normalized (ligatures, full-width punctuation), lowered,
reduced to Unicode word characters (so CJK survives), and whitespace inside
CJK runs is dropped (PDF line breaks fall inside Chinese sentences). The
quote may carry a source label ("(Abstract)", "Claim 1:", "[Page 4]") and
"..." elisions: labels are stripped and every elided segment must be found.
The best local alignment inside the document is scored with difflib; a
second pass ignores all spaces (OCR text layers drop the hyphen and keep
the line break: "accompa nying"). A quote that cannot be located is
treated as fabricated and the associated score is downgraded to 0.
"""

import re
import unicodedata
from difflib import SequenceMatcher

CJK = r"぀-ヿ㐀-䶿一-鿿豈-﫿가-힯"
_NONWORD = re.compile(r"[\W_]+")
_WS = re.compile(r"\s+")
_CJK_SPACE = re.compile(rf"(?<=[{CJK}]) +| +(?=[{CJK}])")
_CJK_CHAR = re.compile(rf"[{CJK}]")
_LATIN_CHAR = re.compile(r"[a-z]")

_LABEL = r"(?:abstract|claims?|para(?:graph)?|page|fig(?:ure)?s?|col(?:umn)?|lines?|section|step|source|see)"
_LEAD_LABEL = re.compile(
    rf"^\s*(?:\[[^\]]{{1,40}}\]|\({_LABEL}[^)]{{0,40}}\)|{_LABEL}\.?\s*[\d.\-–,]*(?:\s*step\s*\d+)?)\s*[:：\-–]?\s*",
    re.IGNORECASE)
_TRAIL_LABEL = re.compile(rf"\s*[\[(]\s*{_LABEL}\b(?:[^()\[\]]|\([^()]*\)|\[[^\[\]]*\])*[\])]\s*$", re.IGNORECASE | re.DOTALL)
_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…)\s*")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _WS.sub(" ", _NONWORD.sub(" ", t)).strip()
    return _CJK_SPACE.sub("", t)


def strip_labels(quote: str) -> str:
    """Drop a leading/trailing source label the model attached to the excerpt."""
    q = unicodedata.normalize("NFKC", quote or "").strip()
    for _ in range(2):
        q = _LEAD_LABEL.sub("", q, count=1)
    q = _TRAIL_LABEL.sub("", q)
    return q.strip()


def quote_segments(quote: str) -> list[str]:
    """Label-stripped quote split on '...' elisions; each part is verified alone."""
    parts = [p for p in _ELLIPSIS.split(strip_labels(quote)) if p and p.strip()]
    return parts or [quote]


def script_mismatch(quote: str, document: str) -> bool:
    """True when the document is mostly CJK but the quote has no CJK at all
    (the model translated the passage; nothing verbatim can be matched)."""
    q = normalize(quote)
    if _CJK_CHAR.search(q):
        return False
    sample = document[:20000]
    n_cjk = len(_CJK_CHAR.findall(sample))
    n_lat = len(_LATIN_CHAR.findall(sample.lower()))
    return n_cjk > 200 and n_cjk > 2 * n_lat


def _best_ratio(q: str, d: str, threshold: float) -> float:
    if q in d:
        return 1.0
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
    return best


def _locate_normalized(q: str, d: str, threshold: float) -> float:
    best = _best_ratio(q, d, threshold)
    if best < threshold and " " in q:
        best = max(best, _best_ratio(q.replace(" ", ""), d.replace(" ", ""), threshold))
    return best


def locate_quote(quote: str, document: str, threshold: float = 0.9) -> tuple[bool, float]:
    """Return (found, similarity). Exact normalized substring short-circuits;
    otherwise slide a window of the quote's length over the document, with a
    space-insensitive second pass. An elided quote ("a ... b") is found only
    when every segment is found; similarity is the weakest segment."""
    d = normalize(document)
    if not d:
        return False, 0.0
    segs = [s for s in (normalize(p) for p in quote_segments(quote)) if len(s) >= 10]
    if not segs:
        return False, 0.0
    worst = 1.0
    for s in segs:
        r = _locate_normalized(s, d, threshold)
        worst = min(worst, r)
        if worst < threshold:
            return False, round(worst, 4)
    return True, round(worst, 4)


def pdf_text(path: str) -> str:
    """Page text for quote verification: hyphens at line ends joined, ligatures
    expanded, lines repeated on most pages (running headers/footers) and bare
    page-number lines dropped. Reading order is PyMuPDF's default (sort=True
    interleaves the columns of OCR'd two-column patents)."""
    try:
        import fitz
    except ImportError:
        return ""
    flags = (fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP | fitz.TEXT_DEHYPHENATE)
    try:
        with fitz.open(path) as doc:
            pages = [page.get_text("text", flags=flags) for page in doc]
    except Exception:
        return ""
    return "\n".join(strip_running_lines(pages))


_PAGE_NO = re.compile(r"^\s*(?:page\s*)?[\divxlc]{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_SHEET = re.compile(r"^\s*sheet\s*\d+\s*of\s*\d+\s*$", re.IGNORECASE)


def strip_running_lines(pages: list[str], min_pages: int = 3, share: float = 0.5) -> list[str]:
    """Remove short lines that recur on >= share of the pages (running
    headers/footers such as 'US 2009/0248944 A1') and page-number lines."""
    n = len(pages)
    counts: dict[str, int] = {}
    per_page = [p.split("\n") for p in pages]
    for lines in per_page:
        for key in {normalize(l) for l in lines if 0 < len(l.strip()) <= 80}:
            if key:
                counts[key] = counts.get(key, 0) + 1
    running = {k for k, c in counts.items() if n >= min_pages and c >= max(min_pages, share * n)}
    out = []
    for lines in per_page:
        kept = []
        for l in lines:
            s = l.strip()
            if not s:
                kept.append(l)
                continue
            if _PAGE_NO.match(s) or _SHEET.match(s) or (len(s) <= 80 and normalize(s) in running):
                continue
            kept.append(l)
        out.append("\n".join(kept))
    return out


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
    stats = {"scored": 0, "with_quote": 0, "quotes": 0, "verified": 0, "downgraded": 0,
             "translated": 0}
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
            dual, sr, br = verify_quote_dual(strip_labels(q), idx) if idx is not None else (False, 0.0, 0.0)
            ok = found or dual
            check = {"quote": q, "verified": ok, "sim": sim, "span": sr, "bigram": br}
            if not ok and script_mismatch(q, document):
                check["reason"] = "translated"
                stats["translated"] += 1
            checks.append(check)
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
