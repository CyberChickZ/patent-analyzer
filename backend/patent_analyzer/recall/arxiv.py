"""arXiv recall channel.

Free, no key required. arXiv API returns Atom XML; we parse it without
external dependencies (xml.etree).

Transport note (measured 2026-09-18, after the channel_health table in the
report showed this channel returning 0 with "HTTP 406" on every job): the
export API answers httpx with 406 Not Acceptable and an empty body, and keeps
doing so for every header combination tried — default httpx headers, an
explicit `Accept: application/atom+xml`, `Accept: */*`, a curl User-Agent,
`Accept-Encoding: identity`, `Connection: close` — and over HTTP/2 as well
(which returned 429). curl gets 200 and urllib gets 200, three times each, in
the same minute from the same IP. So the block is below the header layer and
the fix is not a header: this module now issues the request with
urllib.request on a thread, which is the same pattern the SerpAPI and USPTO ODP
channels already use.
"""

from __future__ import annotations

import asyncio
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import certifi

from .. import metering
from .pool import Candidate

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
_UA = "patent-analyzer/0.4 (https://github.com/CyberChickZ/patent-analyzer)"

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


def _fetch(params: dict) -> tuple[int, bytes]:
    """Blocking GET, run on a thread. urllib, not httpx — see the module docstring."""
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=_SSL_CTX) as r:
        return r.status, r.read()


async def search(query: str, limit: int = 30) -> tuple[list[Candidate], str | None]:
    """Search arXiv with a free-text query.

    arXiv supports field-prefixed queries (ti:, abs:, all:) but plain text
    works fine — it falls back to the `all:` field.

    Retries on 429 / 5xx with exponential backoff because Cloud Run egress
    IPs share NAT pools that can be temporarily rate-limited.
    """
    if not query:
        return [], "empty query"
    # arXiv chokes on long natural-language queries — keep it short
    short_q = " ".join(query.split())[:200]
    params = {
        "search_query": f"all:{short_q}",
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
            metering.count("arxiv")
            status, body = await asyncio.to_thread(_fetch, params)
            if status == 429:
                last_err = "HTTP 429 (arXiv rate limited)"
                continue
            if 500 <= status < 600:
                last_err = f"HTTP {status} (arXiv server error)"
                continue
            if status >= 400:
                return [], f"HTTP {status}"
            text = body.decode("utf-8", "replace")
            if not text.strip():
                return [], "empty response from arXiv"
            try:
                root = ET.fromstring(text)
            except ET.ParseError as e:
                return [], f"XML parse error: {e}"
            entries = root.findall("atom:entry", NS)
            cands = [_entry_to_candidate(e) for e in entries]
            return [c for c in cands if c.title], None
        except TimeoutError:
            last_err = "timeout"
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                last_err = f"HTTP {e.code}"
                continue
            return [], f"HTTP {e.code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return [], last_err or "unknown error after retries"
