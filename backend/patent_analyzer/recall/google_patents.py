"""Google Patents direct channel — no SerpAPI, no browser. Opportunistic.

patents.google.com serves its own search through an XHR endpoint that
returns JSON, and each patent page is static HTML with numbered
description paragraphs. Both are fetched with plain HTTP.

Measured 2026-09-17: ~10 requests in a few minutes from one IP triggered
Google's "Sorry" soft-block on both endpoints. So: 2 concurrent, >=1s gap,
and a 15-minute circuit breaker once a block is seen. Full text should be
fetched from BigQuery (bigquery_patents.fetch_by_pub_nums); the page
scraper is a fallback only.
"""

from __future__ import annotations

import asyncio
import html
import re
import urllib.parse

import hashlib
import os

import httpx

from ..cache import kv
from ..runtime_state import Breaker
from .pool import Candidate

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_XHR = "https://patents.google.com/xhr/query"
_PAGE = "https://patents.google.com/patent/{pub}/en"
_sem = asyncio.Semaphore(1)
_MIN_GAP = float(os.environ.get("GP_MIN_GAP", "4.0"))        # community-measured safe pace
_last_call = 0.0
_BLOCK_COOLDOWN = 15 * 60                                   # Sorry page
_SOFT_COOLDOWN = float(os.environ.get("GP_503_COOLDOWN", "90"))   # bare 503 / 429
_breaker = Breaker("google_patents", cooldown_s=_BLOCK_COOLDOWN)
_soft_breaker = Breaker("google_patents_503", cooldown_s=_SOFT_COOLDOWN)
_CACHE_DAYS = 30
last_total: dict[str, int] = {}   # query -> total_num_results from the last live call


def _clean(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).replace("\xa0", " ").strip()


async def _get(url: str, timeout: float = 30) -> httpx.Response | None:
    """Polite GET: 2 concurrent, >=1s apart, retry 429/5xx with backoff."""
    global _last_call
    if _breaker.is_open() or _soft_breaker.is_open():
        return None
    async with _sem:
        for attempt in range(2):
            wait = _MIN_GAP - (asyncio.get_event_loop().time() - _last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            _last_call = asyncio.get_event_loop().time()
            try:
                async with httpx.AsyncClient(headers={"User-Agent": _UA}, timeout=timeout,
                                             follow_redirects=True) as client:
                    r = await client.get(url)
            except httpx.HTTPError:
                r = None
            if r is not None and r.status_code == 200:
                return r
            if r is not None and r.status_code not in (429, 500, 502, 503):
                return r
            if r is not None and "<title>Sorry" in r.text[:400]:
                # Google abuse page: this IP is soft-blocked; trip the shared breaker
                _breaker.trip(f"Sorry page on {url[:80]}")
                return None
            if r is not None and r.status_code in (429, 503):
                # rate signal without the Sorry page: back off 90 s, do not hammer
                _soft_breaker.trip(f"HTTP {r.status_code}")
                return None
            await asyncio.sleep(2.0 * (attempt + 1))
    return r


def is_blocked() -> bool:
    return _breaker.is_open() or _soft_breaker.is_open()


def _cache_key(query: str, num: int, page: int, before: str | None) -> str:
    norm = " ".join(query.lower().split())
    return hashlib.sha1(f"gp|{norm}|{num}|{page}|{before or ''}".encode()).hexdigest()


async def search(query: str, num: int = 20, page: int = 0,
                 before: str | None = None) -> tuple[list[Candidate], str | None]:
    """Keyword search. `before` = 'priority:YYYYMMDD' style date cutoff
    understood by Google Patents (e.g. 'priority:20150101')."""
    if not query.strip():
        return [], "empty query"
    ck = _cache_key(query, num, page, before)
    hit = kv().get("search", ck, max_age_days=_CACHE_DAYS)
    if hit is not None:
        last_total[query] = int(hit.get("total", 0))
        return [Candidate(**c) for c in hit["cands"]], None
    inner = f"q={'+'.join(query.split())}&num={min(num, 100)}&page={page}"
    if before:
        inner += f"&before={before}"
    # the site encodes the inner query string exactly once; ':' must survive
    url = f"{_XHR}?url={urllib.parse.quote(inner, safe='')}&exp="
    r = await _get(url, timeout=20)
    if r is None:
        return [], "google_patents: blocked or unreachable"
    if r.status_code != 200:
        return [], f"google_patents: HTTP {r.status_code}"
    try:
        results = r.json()["results"]
        clusters = results.get("cluster") or []
    except (ValueError, KeyError):
        return [], "google_patents: unexpected response shape"
    total = int(results.get("total_num_results") or 0)
    last_total[query] = total
    out: list[Candidate] = []
    for cl in clusters:
        for item in cl.get("result") or []:
            p = item.get("patent") or {}
            pub = p.get("publication_number", "")
            if not pub:
                continue
            out.append(Candidate(
                title=_clean(p.get("title", "")),
                snippet=_clean(p.get("snippet", "")),
                url=_PAGE.format(pub=pub),
                pdf_link=f"https://patentimages.storage.googleapis.com/{p['pdf']}" if p.get("pdf") else "",
                pub_num=pub,
                match_type="Patent",
                year=str(p.get("priority_date") or p.get("publication_date") or "")[:4],
                authors=_clean(p.get("inventor", "")),
                source_score=1.0,
                raw={"google_patents": {"priority_date": p.get("priority_date"),
                                        "publication_date": p.get("publication_date"),
                                        "assignee": _clean(p.get("assignee", "")), "total": total}},
            ))
    kv().put("search", ck, {"cands": [c.__dict__ for c in out], "total": total})
    return out, None


_ABSTRACT = re.compile(r'<section itemprop="abstract".*?<div[^>]*class="abstract"[^>]*>(.*?)</div>', re.S)
_PARA = re.compile(r'<div id="p-\d+" num="(\d+)" class="description-paragraph">(.*?)</div>', re.S)
_TITLE = re.compile(r'<meta name="DC.title" content="([^"]*)"')
_PDF = re.compile(r'<meta name="citation_pdf_url" content="([^"]+)"')
_PRIORITY = re.compile(r'<time itemprop="priorityDate" datetime="([^"]+)"')


async def fetch_patent(pub_num: str) -> dict | None:
    """Full text of one patent from its static page: title, abstract,
    claims (list), description paragraphs (list, with the office's own
    [nnnn] numbering), pdf_url, priority_date."""
    pub = re.sub(r"[\s\-]", "", pub_num or "")
    if not pub:
        return None
    r = await _get(_PAGE.format(pub=pub))
    if r is None or r.status_code != 200:
        return None
    h = r.text
    claims = []
    for block in re.findall(r'<div id="CLM-\d+"[^>]*>.*?(?=<div id="CLM-\d+"|</section>)', h, re.S):
        txt = _clean(block)
        if txt:
            claims.append(txt)
    paras = [f"[{n}] {_clean(t)}" for n, t in _PARA.findall(h) if _clean(t)]
    m_abs = _ABSTRACT.search(h)
    m_t, m_pdf, m_pri = _TITLE.search(h), _PDF.search(h), _PRIORITY.search(h)
    return {
        "publication_number": pub,
        "title": html.unescape(m_t.group(1)).strip() if m_t else "",
        "abstract": _clean(m_abs.group(1)) if m_abs else "",
        "claims": claims,
        "description": paras,
        "pdf_url": m_pdf.group(1) if m_pdf else "",
        "priority_date": m_pri.group(1) if m_pri else "",
    }
