"""Candidate pool — normalized cross-channel record + dedupe + multi-source weighting.

Every recall channel produces Candidate dicts with the same shape so the
fusion layer can dedupe by stable identifier (DOI / arXiv ID / patent number /
title hash) and bump the score of candidates that multiple channels agreed on.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Candidate:
    """Normalized prior-art record across all recall channels."""
    title: str = ""
    snippet: str = ""               # short text from the source
    abstract: str = ""              # longer if available
    pdf_link: Any = ""              # str or list[str]
    url: str = ""                   # canonical landing URL
    pub_num: str = ""               # patent number / DOI / arxiv id
    doi: str = ""                   # DOI when known
    arxiv_id: str = ""              # arxiv id when known
    match_type: str = "Paper"       # "Paper" | "Patent"
    year: str = ""
    authors: str = ""
    sources: list[str] = field(default_factory=list)   # which channels found this
    source_score: float = 0.0       # max raw score reported by any source
    raw: dict = field(default_factory=dict)            # channel-specific raw data

    def to_dict(self) -> dict:
        d = asdict(self)
        # Pipeline downstream code expects these legacy field names
        d.setdefault("local_pdf", "")
        return d


_TITLE_NORM = re.compile(r"[^a-z0-9]+")


def _norm_title(t: str) -> str:
    return _TITLE_NORM.sub("", (t or "").lower())[:120]


def _identity_keys(c: Candidate) -> list[str]:
    """Stable keys to dedupe candidates across channels. First match wins."""
    keys = []
    if c.doi:
        keys.append(f"doi:{c.doi.lower()}")
    if c.arxiv_id:
        keys.append(f"arxiv:{c.arxiv_id.lower()}")
    if c.pub_num and c.match_type == "Patent":
        keys.append(f"patent:{c.pub_num.lower().replace(' ', '')}")
    nt = _norm_title(c.title)
    if len(nt) >= 20:
        keys.append(f"title:{nt}")
    return keys


def pool_and_dedupe(channel_results: dict[str, list[Candidate]]) -> list[Candidate]:
    """Merge candidates from multiple channels into a deduped pool.

    Args:
        channel_results: {channel_name: [Candidate, ...]}

    Returns:
        Deduped list. When a candidate appears in N channels, its `sources`
        list contains all N channel names and the multi-source bonus is
        applied to source_score (sqrt(N) — diminishing returns).
    """
    by_key: dict[str, Candidate] = {}
    seen_for_cand: dict[int, set[str]] = {}

    for channel, cands in channel_results.items():
        for c in cands:
            if not c.title and not c.pub_num:
                continue
            keys = _identity_keys(c)
            if not keys:
                continue
            # Find existing candidate by any matching key
            existing: Candidate | None = None
            for k in keys:
                if k in by_key:
                    existing = by_key[k]
                    break
            if existing is None:
                # New candidate
                c.sources = [channel]
                for k in keys:
                    by_key[k] = c
                seen_for_cand[id(c)] = set(keys)
            else:
                # Merge into existing
                if channel not in existing.sources:
                    existing.sources.append(channel)
                # Fill in fields the new record has and existing lacks
                for fname in ("snippet", "abstract", "pdf_link", "url", "doi",
                              "arxiv_id", "year", "authors"):
                    if not getattr(existing, fname) and getattr(c, fname):
                        setattr(existing, fname, getattr(c, fname))
                if c.source_score > existing.source_score:
                    existing.source_score = c.source_score
                # Index this candidate under any new keys we just learned
                for k in keys:
                    if k not in by_key:
                        by_key[k] = existing
                seen_for_cand.setdefault(id(existing), set()).update(keys)

    # Dedupe to unique objects
    unique: list[Candidate] = []
    seen_ids: set[int] = set()
    for c in by_key.values():
        if id(c) in seen_ids:
            continue
        seen_ids.add(id(c))
        unique.append(c)

    # Multi-source consensus bonus
    for c in unique:
        n = len(c.sources)
        if n > 1:
            c.source_score *= math.sqrt(n)
    return unique


def candidates_to_legacy_docs(cands: list[Candidate]) -> list[dict]:
    """Convert Candidate list into the dict shape the existing pipeline uses."""
    out = []
    for c in cands:
        d = {
            "title": c.title,
            "snippet": c.snippet or c.abstract[:300],
            "abstract": c.abstract,
            "pdf_link": c.pdf_link,
            "url": c.url,
            "pub_num": c.pub_num or c.doi or c.arxiv_id,
            "match_type": c.match_type,
            "year": c.year,
            "authors": c.authors,
            "sources": c.sources,
            "source_score": c.source_score,
        }
        out.append(d)
    return out
