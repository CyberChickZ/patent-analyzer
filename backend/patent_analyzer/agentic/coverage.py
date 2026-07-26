"""Cheap in-loop coverage proxy: which pooled documents plausibly cover
which element. Decides what to search for next; the evidence-grade
judgement stays in Phase 4."""

from __future__ import annotations

import re

import numpy as np


def _text(doc: dict) -> str:
    return " ".join(str(doc.get(k) or "") for k in ("title", "abstract", "snippet", "claims_text")).lower()


def _stem_hit(stems: list[str], text: str) -> bool:
    return any(re.search(r"\b" + re.escape(s.lower()), text) for s in stems if len(s) > 2)


def tag_coverage(elements: list[dict], docs: list[dict], embed=None, tau: float = 0.55) -> dict[str, list[str]]:
    """{element_id: [pub_num or title, ...]} — lexical rule (thing ∧ place stems)
    or, when `embed` is given, cosine(element text, doc text) ≥ tau."""
    out = {e["id"]: [] for e in elements}
    texts = [_text(d) for d in docs]
    sims = None
    if embed is not None and elements and docs:
        ev = embed([e["text"] for e in elements])
        dv = embed([t[:2000] for t in texts])
        sims = ev @ dv.T
    for i, e in enumerate(elements):
        f = e.get("facets") or {}
        for j, (d, t) in enumerate(zip(docs, texts)):
            lex = _stem_hit(f.get("thing") or [], t) and (not f.get("place") or _stem_hit(f.get("place") or [], t))
            sem = sims is not None and float(sims[i, j]) >= tau
            if lex or sem:
                out[e["id"]].append(d.get("pub_num") or d.get("title", "")[:60])
    return out
