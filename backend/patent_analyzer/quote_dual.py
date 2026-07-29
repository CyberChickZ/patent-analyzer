"""Dual-threshold quote grounding (eval-only, sits beside patent_analyzer/quote_verify).

A quote passes when both hold:
  span_ratio   >= SPAN_TAU   — share of the quote's content words (stop words
                               removed) that co-occur inside one sliding window
                               of the document, window = max(2*n, n+10) tokens;
  bigram_ratio >= BIGRAM_TAU — share of the quote's adjacent token pairs that
                               appear anywhere in the document.
The first tolerates paraphrase / reordering inside a local span; the second
rejects bags of on-topic words that were never written in that order. This is
looser than quote_verify.locate_quote (char-level difflib >= 0.9) and is the
control arm for the multi-quote coverage eval.
"""

import re

SPAN_TAU = 0.70
BIGRAM_TAU = 0.30
MIN_CONTENT = 3

_TOK = re.compile(r"[a-z0-9]+")
STOP = frozenset("""
a an the and or of to in on at by for with from as is are was were be been being
this that these those it its into onto than then such via each any all one two
which who whom whose where when while there their they them he she his her we our
can may might shall should will would could has have had having do does did not
no nor so if but also more most other some said wherein whereby thereof therein
hereby further least between through over under about above below said
""".split())


def tokens(text: str) -> list[str]:
    return _TOK.findall((text or "").lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOP]


def span_ratio(quote_tokens: list[str], doc_tokens: list[str], window: int | None = None) -> float:
    """Max over document windows of |distinct quote words in window| / |distinct quote words|."""
    need = set(quote_tokens)
    if not need or not doc_tokens:
        return 0.0
    n = len(quote_tokens)
    w = window or max(2 * n, n + 10)
    counts: dict[str, int] = {}
    matched = best = 0
    for i, tok in enumerate(doc_tokens):
        if tok in need:
            counts[tok] = counts.get(tok, 0) + 1
            if counts[tok] == 1:
                matched += 1
        if i >= w:
            out = doc_tokens[i - w]
            if out in need:
                counts[out] -= 1
                if counts[out] == 0:
                    matched -= 1
        if matched > best:
            best = matched
            if best == len(need):
                break
    return best / len(need)


def bigram_ratio(quote_tokens: list[str], doc_bigrams: set[tuple[str, str]]) -> float:
    pairs = list(zip(quote_tokens, quote_tokens[1:]))
    if not pairs:
        return 0.0
    return sum(1 for p in pairs if p in doc_bigrams) / len(pairs)


class DocIndex:
    """Tokenized document + bigram set, built once per document."""

    def __init__(self, text: str):
        self.tokens = tokens(text)
        self.bigrams = set(zip(self.tokens, self.tokens[1:]))


def verify_quote_dual(quote: str, doc, span_tau: float = SPAN_TAU,
                      bigram_tau: float = BIGRAM_TAU) -> tuple[bool, float, float]:
    """(passed, span_ratio, bigram_ratio). `doc` is a str or a DocIndex."""
    idx = doc if isinstance(doc, DocIndex) else DocIndex(doc)
    q_all = tokens(quote)
    q_content = content_tokens(quote)
    if len(q_content) < MIN_CONTENT:
        return False, 0.0, 0.0
    sr = span_ratio(q_content, idx.tokens)
    br = bigram_ratio(q_all, idx.bigrams)
    return (sr >= span_tau and br >= bigram_tau), round(sr, 4), round(br, 4)


def locate_dual(quote: str, refs: list[tuple[str, int | None, str]],
                span_tau: float = SPAN_TAU, bigram_tau: float = BIGRAM_TAU
                ) -> tuple[str, int | None] | None:
    """Best passage (kind, number) among `refs` = [(kind, number, text)] whose
    own text passes the dual test; ranked by (span_ratio, bigram_ratio)."""
    q_all = tokens(quote)
    q_content = content_tokens(quote)
    if len(q_content) < MIN_CONTENT:
        return None
    best, best_ref = (0.0, 0.0), None
    for kind, num, text in refs:
        idx = DocIndex(text)
        sr = span_ratio(q_content, idx.tokens)
        if sr < span_tau:
            continue
        br = bigram_ratio(q_all, idx.bigrams)
        if br < bigram_tau:
            continue
        if (sr, br) > best:
            best, best_ref = (sr, br), (kind, num)
    return best_ref
