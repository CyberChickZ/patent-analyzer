"""OpenAlex recall channel.

Free, polite-pool with mailto + optional API key (OPENALEX_KEY env). Provides:
  - search_works(query)        — keyword search of works
  - expand_by_concepts(ids)    — find more works tagged with the same concepts
  - reconstruct_abstract(idx)  — turn OA inverted index → readable text
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .pool import Candidate

API_BASE = "https://api.openalex.org"
TIMEOUT = 30.0


def _mailto() -> str:
    return os.environ.get("OPENALEX_MAILTO", "patent-analyzer@example.com")


def _api_key() -> str | None:
    return os.environ.get("OPENALEX_KEY")


def _params(extra: dict | None = None) -> dict:
    p = {"mailto": _mailto()}
    key = _api_key()
    if key:
        p["api_key"] = key
    if extra:
        p.update(extra)
    return p


def reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[k] for k in sorted(words.keys()))


def _work_to_candidate(w: dict) -> Candidate:
    if not isinstance(w, dict):
        return Candidate()
    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
    authors = ", ".join(
        (a.get("author") or {}).get("display_name", "")
        for a in (w.get("authorships") or [])[:5]
        if isinstance(a, dict)
    )
    if len(w.get("authorships") or []) > 5:
        authors += " et al."

    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    pdf_url = ""
    oa = w.get("open_access") or {}
    if isinstance(oa, dict):
        pdf_url = oa.get("oa_url") or ""
    if not pdf_url:
        primary = w.get("primary_location") or {}
        if isinstance(primary, dict):
            pdf_url = primary.get("pdf_url") or ""

    arxiv_id = ""
    ids = w.get("ids") or {}
    if isinstance(ids, dict):
        for k, v in ids.items():
            if isinstance(v, str) and "arxiv.org" in v.lower():
                # e.g. https://arxiv.org/abs/2507.15693
                tail = v.rsplit("/", 1)[-1]
                arxiv_id = tail.split("v")[0]
                break

    pub_num = doi or arxiv_id or w.get("id", "")
    return Candidate(
        title=w.get("title", "") or "",
        snippet=abstract[:300],
        abstract=abstract,
        pdf_link=pdf_url,
        url=w.get("id", "") or (f"https://doi.org/{doi}" if doi else ""),
        pub_num=str(pub_num),
        doi=doi,
        arxiv_id=arxiv_id,
        match_type="Paper",
        year=str(w.get("publication_year") or ""),
        authors=authors,
        source_score=float(w.get("cited_by_count") or 0),
        raw={"openalex": {"id": w.get("id", ""),
                          "concepts": [c.get("id", "") for c in (w.get("concepts") or [])[:5]]}},
    )


async def _get(client: httpx.AsyncClient, url: str, params: dict,
               attempts: int = 3) -> tuple[dict | None, str | None]:
    """OpenAlex GET with retry on 429 and 5xx (OpenAlex 5xx is intermittent)."""
    import asyncio
    backoffs = [2, 6, 15]
    last_err: str | None = None
    for i in range(attempts):
        if i > 0:
            await asyncio.sleep(backoffs[min(i - 1, len(backoffs) - 1)])
        try:
            resp = await client.get(url, params=params, timeout=TIMEOUT,
                                     headers={"User-Agent": "patent-analyzer/0.3"})
            if resp.status_code == 429:
                last_err = "HTTP 429 (OpenAlex rate limited)"
                continue
            if 500 <= resp.status_code < 600:
                last_err = f"HTTP {resp.status_code} (OpenAlex server error)"
                continue
            if resp.status_code >= 400:
                return None, f"HTTP {resp.status_code} ({resp.text[:200]})"
            return resp.json(), None
        except httpx.TimeoutException:
            last_err = "timeout"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err or "unknown error after retries"


async def search_works(query: str, limit: int = 30) -> tuple[list[Candidate], str | None]:
    """Free-text search across OpenAlex works."""
    if not query:
        return [], "empty query"
    async with httpx.AsyncClient() as client:
        data, err = await _get(client, f"{API_BASE}/works", _params({
            "search": query[:300],
            "per_page": min(max(limit, 1), 100),
            "select": "id,title,abstract_inverted_index,publication_year,authorships,doi,open_access,primary_location,ids,cited_by_count,concepts",
        }))
    if err:
        return [], err
    if not data:
        return [], "empty response"
    cands = [_work_to_candidate(w) for w in (data.get("results") or [])]
    return [c for c in cands if c.title], None


async def expand_by_concepts(concept_ids: list[str], limit: int = 25) -> tuple[list[Candidate], str | None]:
    """Find more works tagged with the same concepts (after we got some hits)."""
    concept_ids = [c for c in concept_ids if c][:5]
    if not concept_ids:
        return [], "no concept ids"
    concept_filter = "|".join(concept_ids)
    async with httpx.AsyncClient() as client:
        data, err = await _get(client, f"{API_BASE}/works", _params({
            "filter": f"concepts.id:{concept_filter}",
            "sort": "cited_by_count:desc",
            "per_page": min(max(limit, 1), 100),
            "select": "id,title,abstract_inverted_index,publication_year,authorships,doi,open_access,primary_location,ids,cited_by_count,concepts",
        }))
    if err:
        return [], err
    if not data:
        return [], "empty response"
    cands = [_work_to_candidate(w) for w in (data.get("results") or [])]
    return [c for c in cands if c.title], None


async def search_paper_by_title(title: str) -> Candidate | None:
    """Compatibility shim for the old fetch_abstracts.py — get a single best match."""
    if not title:
        return None
    async with httpx.AsyncClient() as client:
        data, err = await _get(client, f"{API_BASE}/works", _params({
            "filter": f"title.search:{title[:200]}",
            "per_page": 1,
            "select": "id,title,abstract_inverted_index,publication_year,authorships,doi,open_access,primary_location,ids,cited_by_count,concepts",
        }))
    if err or not data:
        return None
    results = data.get("results") or []
    if not results:
        return None
    return _work_to_candidate(results[0])
