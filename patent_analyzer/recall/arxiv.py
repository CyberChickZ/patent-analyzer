"""arXiv recall channel.

Free, no key required. arXiv API returns Atom XML; we parse it without
external dependencies (xml.etree).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from .pool import Candidate

API_URL = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
TIMEOUT = 30.0


def _entry_to_candidate(entry: ET.Element) -> Candidate:
    def _text(tag: str, default: str = "") -> str:
        el = entry.find(f"atom:{tag}", NS)
        return (el.text or "").strip() if el is not None and el.text else default

    title = re.sub(r"\s+", " ", _text("title"))
    summary = re.sub(r"\s+", " ", _text("summary"))
    published = _text("published")
    year = published[:4] if len(published) >= 4 else ""

    arxiv_id = ""
    id_el = entry.find("atom:id", NS)
    if id_el is not None and id_el.text:
        m = re.search(r"arxiv\.org/abs/([^/v\s]+)(v\d+)?", id_el.text)
        if m:
            arxiv_id = m.group(1)

    pdf_link = ""
    abs_link = ""
    for link in entry.findall("atom:link", NS):
        if link.get("type") == "application/pdf":
            pdf_link = link.get("href", "")
        elif link.get("rel") == "alternate":
            abs_link = link.get("href", "")

    authors = ", ".join(
        (a.findtext("atom:name", default="", namespaces=NS) or "").strip()
        for a in entry.findall("atom:author", NS)
    )

    doi = ""
    doi_el = entry.find("arxiv:doi", NS)
    if doi_el is not None and doi_el.text:
        doi = doi_el.text.strip()

    return Candidate(
        title=title,
        snippet=summary[:300],
        abstract=summary,
        pdf_link=pdf_link,
        url=abs_link or f"https://arxiv.org/abs/{arxiv_id}",
        pub_num=arxiv_id or doi,
        doi=doi,
        arxiv_id=arxiv_id,
        match_type="Paper",
        year=year,
        authors=authors,
        source_score=0.0,
        raw={"arxiv": {"published": published}},
    )


async def search(query: str, limit: int = 30) -> tuple[list[Candidate], str | None]:
    """Search arXiv with a free-text query.

    arXiv supports field-prefixed queries (ti:, abs:, all:) but plain text
    works fine — it falls back to the `all:` field.

    Retries on 429 / 5xx with exponential backoff because Cloud Run egress
    IPs share NAT pools that can be temporarily rate-limited.
    """
    import asyncio
    if not query:
        return [], "empty query"
    params = {
        "search_query": f"all:{query[:300]}",
        "start": 0,
        "max_results": min(max(limit, 1), 100),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    backoffs = [3, 8, 20]  # 3 attempts total
    last_err: str | None = None
    for attempt, wait in enumerate([0] + backoffs):
        if wait:
            await asyncio.sleep(wait)
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(
                    API_URL, params=params, timeout=TIMEOUT,
                    headers={"User-Agent": "patent-analyzer/0.3 (https://github.com/CyberChickZ/patent-analyzer)"},
                )
            if resp.status_code == 429:
                last_err = "HTTP 429 (arXiv rate limited)"
                continue
            if 500 <= resp.status_code < 600:
                last_err = f"HTTP {resp.status_code} (arXiv server error)"
                continue
            if resp.status_code >= 400:
                return [], f"HTTP {resp.status_code}"
            if not resp.text.strip():
                return [], "empty response from arXiv"
            try:
                root = ET.fromstring(resp.text)
            except ET.ParseError as e:
                return [], f"XML parse error: {e}"
            entries = root.findall("atom:entry", NS)
            cands = [_entry_to_candidate(e) for e in entries]
            return [c for c in cands if c.title], None
        except httpx.TimeoutException:
            last_err = "timeout"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return [], last_err or "unknown error after retries"
