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

from .. import metering
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
            metering.count("openalex")
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


async def search_works(query: str, limit: int = 30,
                       before_year: str = "") -> tuple[list[Candidate], str | None]:
    """Free-text search across OpenAlex works.

    `before_year` becomes `filter=publication_year:<YYYY+1`, so the ranking we
    page through is already prior art — see semantic_scholar.search for why
    post-filtering costs ~87% of the results.
    """
    if not query:
        return [], "empty query"
    p = {
        "search": query[:300],
        "per_page": min(max(limit, 1), 100),
        "select": "id,title,abstract_inverted_index,publication_year,authorships,doi,open_access,primary_location,ids,cited_by_count,concepts",
    }
    if str(before_year)[:4].isdigit():
        p["filter"] = f"publication_year:<{int(str(before_year)[:4]) + 1}"
    async with httpx.AsyncClient() as client:
        data, err = await _get(client, f"{API_BASE}/works", _params(p))
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


def _title_key(t: str) -> str:
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in (t or "").lower()).split())


async def backfill_dois(cands: list, cap: int = 60) -> dict:
    """Give paper candidates a DOI when they arrived without one.

    41 of the 88 papers a real job could not read had no DOI at all, which is
    the single largest reason nothing further can be done with them: no DOI
    means no Unpaywall, no Europe PMC, no GCS cache key and nothing to put on a
    manual-download list (N7, 2026-09-18). OpenAlex answers a title search for
    free, so the lookup is one request per unresolved paper, rate limited and
    cached, and it only ever fills a field that was empty.

    A match counts only when the returned title is the same title — OpenAlex
    will happily return its closest work for a query that matches nothing, and
    a wrong DOI is worse than none: it points the full-text fetch, the shared
    GCS cache and the manual-download list at the wrong paper at once.

    Title equality on one source is not enough for that, so a *substring* match
    (the loose case, where OpenAlex's title merely contains ours or vice versa)
    is only accepted when Crossref independently returns the same DOI for the
    same title. Exact title equality is accepted on OpenAlex alone. Crossref is
    free and cached, so the cross-check costs one request on the uncertain rows
    and nothing on the certain ones. (N7b, 2026-09-19, on paper-fetch's
    two-source pattern.)
    """
    from ..cache import kv
    from ..runtime_state import SerialLock
    from ..fulltext_sources import crossref_doi_for_title
    todo = [c for c in cands
            if getattr(c, "match_type", "") != "Patent" and not getattr(c, "doi", "")
            and len(_title_key(getattr(c, "title", ""))) > 20][:cap]
    out = {"asked": 0, "filled": 0, "cached": 0, "mismatched": 0,
           "cross_confirmed": 0, "cross_rejected": 0}
    if not todo:
        return out
    store = kv()
    async with httpx.AsyncClient() as client:
        for c in todo:
            key = "oa_doi:" + _title_key(c.title)[:180]
            hit = store.get("recall", key)
            if hit is not None:
                out["cached"] += 1
                if hit.get("doi"):
                    c.doi = hit["doi"]
                    out["filled"] += 1
                continue
            async with SerialLock("openalex", cooldown_s=1.0):
                data, err = await _get(client, f"{API_BASE}/works", _params({
                    "filter": f"title.search:{c.title[:200]}", "per_page": 1, "select": "id,title,doi"}))
            out["asked"] += 1
            doi = ""
            if not err and data:
                r = (data.get("results") or [{}])[0]
                got = _title_key(r.get("title") or "")
                want = _title_key(c.title)
                cand = (r.get("doi") or "").replace("https://doi.org/", "").lower()
                if got and got == want:
                    doi = cand
                elif got and cand and len(got) > 30 and (got in want or want in got):
                    # Loose match: make Crossref agree before believing it.
                    other = await crossref_doi_for_title(c.title)
                    if other and other == cand:
                        doi, out["cross_confirmed"] = cand, out["cross_confirmed"] + 1
                    else:
                        out["cross_rejected"] += 1
                elif got:
                    out["mismatched"] += 1
            store.put("recall", key, {"doi": doi})
            if doi:
                c.doi = doi
                out["filled"] += 1
    return out


async def ids_for_dois(dois: list[str]) -> dict[str, str]:
    """DOI → OpenAlex work id (W…), 50 per request via the pipe-joined
    `filter=doi:` syntax (OpenAlex docs: "You can use OR by putting a pipe
    between values", ≤50 values)."""
    import asyncio
    out: dict[str, str] = {}
    clean = [d.strip().lower().replace("https://doi.org/", "") for d in dois if d and d.strip()]
    clean = list(dict.fromkeys(clean))
    async with httpx.AsyncClient() as client:
        for i in range(0, len(clean), 50):
            chunk = clean[i:i + 50]
            data, err = await _get(client, f"{API_BASE}/works", _params({
                "filter": "doi:" + "|".join(chunk), "per_page": 50, "select": "id,doi"}))
            for w in ((data or {}).get("results") or []):
                doi = (w.get("doi") or "").lower().replace("https://doi.org/", "")
                if doi and w.get("id"):
                    out[doi] = w["id"].rsplit("/", 1)[-1]
            if i + 50 < len(clean):
                await asyncio.sleep(1.0)
    return out
