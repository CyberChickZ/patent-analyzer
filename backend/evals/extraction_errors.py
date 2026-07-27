"""Error taxonomy for element extraction (omission / fabrication /
misclassification) plus the quote snapper shared by the extraction subgraph
and the FiNE / Pap2Pat extraction evals.

snap_quote(quote, doc_text) -> (found, loc)
    1. exact: first 60 normalized chars of the quote found in the normalized
       document (coverage_eval.quote_location's rule) -> char span
    2. fuzzy: quote_verify.locate_quote at 0.9 -> best difflib window
    3. dual:  quote_dual span/bigram thresholds -> densest token window
    none pass -> (False, None): the quote was written, not copied.
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from patent_analyzer.quote_verify import locate_quote, normalize
from quote_dual import DocIndex, content_tokens, verify_quote_dual

HEAD_CHARS = 60
FUZZY_TAU = 0.9
_TOK = re.compile(r"[a-z0-9]+")


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """quote_verify.normalize, plus original char index of every output char."""
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text or ""):
        for c in ch.lower():
            if ("a" <= c <= "z") or ("0" <= c <= "9"):
                out.append(c)
                idx.append(i)
            elif out and out[-1] != " ":
                out.append(" ")
                idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def _span(idx: list[int], start: int, end: int) -> list[int]:
    end = max(start + 1, min(end, len(idx)))
    return [idx[start], idx[end - 1] + 1]


def _fuzzy_span(q: str, d: str, idx: list[int]) -> tuple[float, list[int]] | None:
    n = len(q)
    step = max(1, n // 4)
    sm = SequenceMatcher(None, autojunk=False)
    sm.set_seq2(q)
    best, best_start = 0.0, None
    for start in range(0, max(1, len(d) - n + step), step):
        sm.set_seq1(d[start:start + n])
        if sm.real_quick_ratio() < FUZZY_TAU or sm.quick_ratio() < FUZZY_TAU:
            continue
        r = sm.ratio()
        if r > best:
            best, best_start = r, start
            if best >= 0.995:
                break
    if best_start is None:
        return None
    return best, _span(idx, best_start, best_start + n)


def _dual_span(quote: str, doc_text: str) -> list[int] | None:
    need = set(content_tokens(quote))
    if not need:
        return None
    toks = [(m.group(), m.start(), m.end()) for m in _TOK.finditer((doc_text or "").lower())]
    if not toks:
        return None
    n = len(need)
    w = max(2 * n, n + 10)
    counts: dict[str, int] = {}
    matched = best = 0
    best_i = None
    for i, (tok, _, _) in enumerate(toks):
        if tok in need:
            counts[tok] = counts.get(tok, 0) + 1
            if counts[tok] == 1:
                matched += 1
        if i >= w:
            out = toks[i - w][0]
            if out in need:
                counts[out] -= 1
                if counts[out] == 0:
                    matched -= 1
        if matched > best:
            best, best_i = matched, i
    if best_i is None:
        return None
    lo = max(0, best_i - w + 1)
    return [toks[lo][1], toks[best_i][2]]


def snap_quote(quote: str, doc_text: str) -> tuple[bool, dict | None]:
    """(found, {"char": [start, end], "method": exact|fuzzy|dual, "sim": float})."""
    q = normalize(quote)
    if len(q) < 10 or not doc_text:
        return False, None
    d, idx = _normalize_with_map(doc_text)
    pos = d.find(q[:HEAD_CHARS])
    if pos >= 0:
        end = pos + len(q) if d.startswith(q, pos) else pos + len(q[:HEAD_CHARS])
        return True, {"char": _span(idx, pos, end), "method": "exact", "sim": 1.0}
    found, sim = locate_quote(quote, doc_text, FUZZY_TAU)
    if found:
        hit = _fuzzy_span(q, d, idx)
        if hit:
            return True, {"char": hit[1], "method": "fuzzy", "sim": round(hit[0], 4)}
        return True, {"char": None, "method": "fuzzy", "sim": sim}
    ok, sr, br = verify_quote_dual(quote, DocIndex(doc_text))
    if ok:
        return True, {"char": _dual_span(quote, doc_text), "method": "dual", "sim": round(min(sr, br), 4)}
    return False, None
