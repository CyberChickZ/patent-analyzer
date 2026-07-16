"""Semantic Scholar recall channel.

Free, no key required (anonymous = 100 req / 5 min). Set SEMANTIC_SCHOLAR_KEY
env var for higher rate limits once a key is approved.

Endpoints used:
  - GET /graph/v1/paper/search             — keyword + semantic-ish search
  - GET /graph/v1/paper/{id}               — paper details (used by batch)
  - GET /graph/v1/paper/{id}/references    — citation graph (papers this one cites)
  - GET /graph/v1/paper/{id}/citations     — reverse citation graph (who cites this)
  - POST /graph/v1/paper/batch             — batch metadata lookup by id
  - GET /recommendations/v1/papers/forpaper/{id}  — semantically similar papers
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from .pool import Candidate

API_BASE = "https://api.semanticscholar.org/graph/v1"
REC_BASE = "https://api.semanticscholar.org/recommendations/v1"
DEFAULT_FIELDS = "paperId,title,abstract,year,authors,externalIds,openAccessPdf,venue,citationCount"
TIMEOUT = 30.0

# Throttle: anonymous = 100 req/5min ≈ 1 req/3s; with key ≈ 1 req/s
_last_request_time: float = 0.0
_throttle_lock = asyncio.Lock()
MIN_INTERVAL_ANON = 3.5
MIN_INTERVAL_KEYED = 1.1


async def _throttle():
    global _last_request_time
    interval = MIN_INTERVAL_KEYED if _api_key() else MIN_INTERVAL_ANON
    async with _throttle_lock:
        now = asyncio.get_event_loop().time()
        wait = interval - (now - _last_request_time)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request_time = asyncio.get_event_loop().time()


def _api_key() -> str | None:
    return os.environ.get("SEMANTIC_SCHOLAR_KEY") or os.environ.get("S2_API_KEY")


def _headers() -> dict[str, str]:
    h = {"User-Agent": "patent-analyzer/0.3 (https://github.com/CyberChickZ/patent-analyzer)"}
    key = _api_key()
    if key:
        h["x-api-key"] = key
    return h


def _to_candidate(p: dict) -> Candidate:
    if not isinstance(p, dict):
        return Candidate()
    ext = p.get("externalIds") or {}
    pdf = p.get("openAccessPdf") or {}
    pdf_url = pdf.get("url") if isinstance(pdf, dict) else ""
    authors = ", ".join(
        a.get("name", "") for a in (p.get("authors") or []) if isinstance(a, dict)
    )
    arxiv_id = ext.get("ArXiv") or ""
    doi = ext.get("DOI") or ""
    paper_id = p.get("paperId", "")
    pub_num = doi or arxiv_id or paper_id
    abstract = p.get("abstract") or ""
    return Candidate(
        title=p.get("title", "") or "",
        snippet=abstract[:300],
        abstract=abstract,
        pdf_link=pdf_url or "",
        url=f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else "",
        pub_num=str(pub_num),
        doi=str(doi),
        arxiv_id=str(arxiv_id),
        match_type="Paper",
        year=str(p.get("year") or ""),
        authors=authors,
        source_score=float(p.get("citationCount") or 0),
        raw={"semantic_scholar": {"paperId": paper_id, "venue": p.get("venue", "")}},
    )


async def _get(client: httpx.AsyncClient, url: str, params: dict | None = None,
               attempts: int = 4) -> tuple[dict | None, str | None]:
    last_err: str | None = None
    for i in range(attempts):
        try:
            await _throttle()
            resp = await client.get(url, params=params, headers=_headers(), timeout=TIMEOUT)
            if resp.status_code == 429:
                last_err = "HTTP 429 (Semantic Scholar rate limited)"
                # Anonymous rate limit is 100 req / 5min — back off generously
                await asyncio.sleep(5 + i * 10)
                continue
            if resp.status_code == 404:
                return None, "HTTP 404 (not found)"
            if 500 <= resp.status_code < 600:
                last_err = f"HTTP {resp.status_code} (Semantic Scholar server error)"
                await asyncio.sleep(2 + i * 5)
                continue
            if resp.status_code >= 400:
                return None, f"HTTP {resp.status_code} ({resp.text[:200]})"
            return resp.json(), None
        except httpx.TimeoutException:
            last_err = "timeout"
            await asyncio.sleep(1 + i)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(1 + i)
    return None, last_err or "unknown error"


async def search(query: str, limit: int = 30) -> tuple[list[Candidate], str | None]:
    """Keyword + semantic search via /graph/v1/paper/search."""
    if not query:
        return [], "empty query"
    async with httpx.AsyncClient() as client:
        data, err = await _get(client, f"{API_BASE}/paper/search", {
            "query": query[:300],
            "limit": min(max(limit, 1), 100),
            "fields": DEFAULT_FIELDS,
        })
    if err:
        return [], err
    if not data:
        return [], "empty response"
    cands = [_to_candidate(p) for p in (data.get("data") or [])]
    return [c for c in cands if c.title], None


async def recommendations(paper_id: str, limit: int = 30) -> tuple[list[Candidate], str | None]:
    """Get semantically similar papers for a seed paper."""
    if not paper_id:
        return [], "empty paper_id"
    async with httpx.AsyncClient() as client:
        data, err = await _get(client, f"{REC_BASE}/papers/forpaper/{paper_id}", {
            "limit": min(max(limit, 1), 100),
            "fields": DEFAULT_FIELDS,
        })
    if err:
        return [], err
    if not data:
        return [], "empty response"
    cands = [_to_candidate(p) for p in (data.get("recommendedPapers") or [])]
    return [c for c in cands if c.title], None


async def references(paper_id: str, limit: int = 50) -> tuple[list[Candidate], str | None]:
    """Get papers that this paper cites (1-hop backward citation graph)."""
    if not paper_id:
        return [], "empty paper_id"
    async with httpx.AsyncClient() as client:
        data, err = await _get(client, f"{API_BASE}/paper/{paper_id}/references", {
            "limit": min(max(limit, 1), 100),
            "fields": DEFAULT_FIELDS,
        })
    if err:
        return [], err
    if not data:
        return [], "empty response"
    cands = []
    for entry in data.get("data") or []:
        cited = entry.get("citedPaper") if isinstance(entry, dict) else None
        if cited:
            cands.append(_to_candidate(cited))
    return [c for c in cands if c.title], None


async def citations(paper_id: str, limit: int = 50) -> tuple[list[Candidate], str | None]:
    """Get papers that cite this paper (1-hop forward citation graph)."""
    if not paper_id:
        return [], "empty paper_id"
    async with httpx.AsyncClient() as client:
        data, err = await _get(client, f"{API_BASE}/paper/{paper_id}/citations", {
            "limit": min(max(limit, 1), 100),
            "fields": DEFAULT_FIELDS,
        })
    if err:
        return [], err
    if not data:
        return [], "empty response"
    cands = []
    for entry in data.get("data") or []:
        citing = entry.get("citingPaper") if isinstance(entry, dict) else None
        if citing:
            cands.append(_to_candidate(citing))
    return [c for c in cands if c.title], None


async def batch(paper_ids: list[str]) -> tuple[list[Candidate], str | None]:
    """Resolve a list of paper IDs (DOI / arXiv / S2 paperId) in one request."""
    paper_ids = [pid for pid in paper_ids if pid][:500]
    if not paper_ids:
        return [], "empty id list"
    async with httpx.AsyncClient() as client:
        try:
            await _throttle()
            resp = await client.post(
                f"{API_BASE}/paper/batch",
                params={"fields": DEFAULT_FIELDS},
                json={"ids": paper_ids},
                headers=_headers(),
                timeout=TIMEOUT,
            )
            if resp.status_code >= 400:
                return [], f"HTTP {resp.status_code} ({resp.text[:200]})"
            data = resp.json()
        except Exception as e:
            return [], f"{type(e).__name__}: {e}"
    if not isinstance(data, list):
        return [], "unexpected response shape"
    cands = [_to_candidate(p) for p in data if p]
    return cands, None
