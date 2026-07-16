"""SerpAPI recall channel — async wrapper around the sync searcher.

Provides the same (Candidate list, error|None) shape as the other recall
channels so phase 3 can treat all sources symmetrically.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .pool import Candidate
from ..searcher import serpapi_search as _sync_search


def _to_candidate(m: dict, match_type: str) -> Candidate:
    pdf_link = m.get("pdf_link", "")
    if isinstance(pdf_link, list) and pdf_link:
        pdf_link = pdf_link[0]
    return Candidate(
        title=m.get("title", "") or "",
        snippet=m.get("snippet", "") or "",
        abstract="",
        pdf_link=pdf_link,
        url=m.get("url") or m.get("patent_link", "") or "",
        pub_num=m.get("pub_num", "") or "",
        match_type=match_type,
        year=str(m.get("year") or m.get("publication_date") or ""),
        authors=m.get("authors", "") or m.get("inventor", "") or "",
        source_score=0.0,
        raw={"serpapi": m},
    )


def _api_key() -> str:
    return os.environ.get("SERPAPI_KEY", "")


async def search_patents(query: str, max_pages: int = 1) -> tuple[list[Candidate], str | None]:
    if not query:
        return [], "empty query"
    key = _api_key()
    if not key:
        return [], "SERPAPI_KEY not set"
    matches, err = await asyncio.to_thread(
        _sync_search, "google_patents", query, key, None, max_pages, 100,
    )
    cands = [_to_candidate(m, "Patent") for m in matches if m.get("title")]
    return cands, err


async def search_scholar(query: str, max_pages: int = 3) -> tuple[list[Candidate], str | None]:
    if not query:
        return [], "empty query"
    key = _api_key()
    if not key:
        return [], "SERPAPI_KEY not set"
    matches, err = await asyncio.to_thread(
        _sync_search, "google_scholar", query, key, None, max_pages, 20,
    )
    cands = [_to_candidate(m, "Paper") for m in matches if m.get("title")]
    return cands, err
