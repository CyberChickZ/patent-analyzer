"""Hybrid retrieval primitives: BM25 sparse scoring + reciprocal rank fusion.

Shared by the production rerank path (semantic_search.rerank_hybrid) and the
offline evals (evals/retrieval_eval.py) so both measure the same code.
"""

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Score every doc against the query. Plain BM25, no dependencies."""
    doc_tokens = [tokenize(d) for d in docs]
    doc_lens = np.array([len(t) for t in doc_tokens], dtype=float)
    avgdl = doc_lens.mean() if len(doc_lens) else 1.0
    df = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    n = len(docs)
    tfs = [Counter(toks) for toks in doc_tokens]
    scores = np.zeros(n)
    for term in set(tokenize(query)):
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        tf = np.array([t[term] for t in tfs], dtype=float)
        scores += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_lens / avgdl))
    return scores


def rrf_fuse(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion: each ranking votes 1/(k + rank), votes add up.

    Rank-based, so sparse and dense scores need no scale calibration.
    """
    n = len(score_lists[0])
    fused = np.zeros(n)
    for scores in score_lists:
        order = np.argsort(scores)[::-1]
        pos = np.empty(n, dtype=int)
        pos[order] = np.arange(n)
        fused += 1.0 / (k + pos + 1)
    return fused
