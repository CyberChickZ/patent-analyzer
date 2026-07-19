"""Google Patents direct channel — no SerpAPI, no browser.

patents.google.com serves its own search through an XHR endpoint that
returns JSON, and each patent page is static HTML with numbered
description paragraphs. Both are fetched with plain HTTP.
"""

from __future__ import annotations

import asyncio
import html
import re
import urllib.parse

import httpx

from .pool import Candidate

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_XHR = "https://patents.google.com/xhr/query"
_PAGE = "https://patents.google.com/patent/{pub}/en"
_sem = asyncio.Semaphore(3)


def _clean(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).replace("\xa0", " ").strip()


async def search(query: str, num: int = 20, page: int = 0,
                 before: str | None = None) -> tuple[list[Candidate], str | None]:
    """Keyword search. `before` = 'priority:YYYYMMDD' style date cutoff
    understood by Google Patents (e.g. 'priority:20150101')."""
    if not query.strip():
        return [], "empty query"
    inner = f"q={'+'.join(query.split())}&num={min(num, 100)}&page={page}"
    if before:
        inner += f"&before={before}"
    # the site encodes the inner query string exactly once; ':' must survive
    url = f"{_XHR}?url={urllib.parse.quote(inner, safe='')}&exp="
    async with _sem:
        try:
            async with httpx.AsyncClient(headers={"User-Agent": _UA}, timeout=20) as client:
                r = await client.get(url)
        except httpx.HTTPError as exc:
            return [], f"google_patents: {type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return [], f"google_patents: HTTP {r.status_code}"
    try:
        clusters = r.json()["results"].get("cluster") or []
    except (ValueError, KeyError):
        return [], "google_patents: unexpected response shape"
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
                                        "assignee": _clean(p.get("assignee", ""))}},
            ))
    return out, None
