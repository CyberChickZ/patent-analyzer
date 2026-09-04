"""SerpAPI recall channel — async wrapper around the sync searcher, with a
rotating pool of free-tier keys, per-key monthly quota (KV-backed, shared
across instances), and a result cache so re-runs cost no credits.

Keys come from SERPAPI_KEYS (comma-separated) or SERPAPI_KEY; they are
never logged or persisted beyond a short fingerprint.
"""

from __future__ import annotations

import asyncio
import hashlib
import os

from .. import metering
from ..cache import kv
from ..runtime_state import MonthlyQuota
from ..searcher import serpapi_search as _sync_search
from .pool import Candidate

FREE_TIER_PER_KEY = int(os.environ.get("SERPAPI_MONTHLY_PER_KEY", "250"))
_CACHE_DAYS = 30
last_total: dict[str, int] = {}


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


def _keys() -> list[str]:
    raw = os.environ.get("SERPAPI_KEYS") or os.environ.get("SERPAPI_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _fp(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def _quota(key: str) -> MonthlyQuota:
    return MonthlyQuota(f"serpapi:{_fp(key)}", FREE_TIER_PER_KEY)


def quota_status() -> list[dict]:
    return [{"key": _fp(k), "used": _quota(k).used(), "cap": FREE_TIER_PER_KEY} for k in _keys()]


def sync_account() -> list[dict]:
    """Overwrite the local counters with SerpAPI's /account this_month_usage
    (the provider also bills 'no results' answers, which the local counter
    used to release)."""
    import json
    import urllib.request
    out = []
    for k in _keys():
        try:
            with urllib.request.urlopen(f"https://serpapi.com/account?api_key={k}", timeout=15) as r:
                d = json.loads(r.read().decode())
            _quota(k).set_used(int(d.get("this_month_usage", 0)))
            out.append({"key": _fp(k), "used": int(d.get("this_month_usage", 0)), "left": d.get("plan_searches_left")})
        except Exception as exc:
            out.append({"key": _fp(k), "error": str(exc)[:80]})
    return out


_NO_RESULTS = "hasn't returned any results"


def _exhausted(err: str | None) -> bool:
    e = (err or "").lower()
    return any(t in e for t in ("401", "429", "exhaust", "run out", "quota", "limit"))


def _cache_key(engine: str, query: str, max_pages: int, num: int, before: str | None = None,
               scholar: bool = False) -> str:
    norm = " ".join(query.lower().split())
    tail = (f"|{before}" if before else "") + ("|scholar" if scholar else "")
    return hashlib.sha1(f"serp|{engine}|{norm}|{max_pages}|{num}{tail}".encode()).hexdigest()


async def _search(engine: str, query: str, max_pages: int, num: int, match_type: str,
                  before: str | None = None, scholar: bool = False) -> tuple[list[Candidate], str | None]:
    if not query:
        return [], "empty query"
    ck = _cache_key(engine, query, max_pages, num, before, scholar)
    extra = {k: v for k, v in (("before", before), ("scholar", "true" if scholar else "")) if v} or None
    hit = kv().get("search", ck, max_age_days=_CACHE_DAYS)
    if hit is not None:
        metering.count("serpapi:cached")
        last_total[query] = int(hit.get("total", 0))
        return [Candidate(**c) for c in hit["cands"]], None
    keys = _keys()
    if not keys:
        return [], "SERPAPI_KEY not set"
    last_err = None
    for key in keys:
        q = _quota(key)
        if not q.take(max_pages):
            last_err = f"serpapi key {_fp(key)} monthly quota exhausted"
            metering.incident("serpapi", metering.EXHAUSTED, last_err)
            continue
        metering.count(f"serpapi:{engine}")
        matches, err = await asyncio.to_thread(_sync_search, engine, query, key, None, max_pages, num, extra)
        if err and _exhausted(err):
            q.exhaust()
            last_err = f"serpapi key {_fp(key)}: {err}"
            metering.incident("serpapi", metering.EXHAUSTED, last_err)
            continue
        if err and _NO_RESULTS in err:
            # billed by SerpAPI like any answer; cache it as an empty page
            matches, err = [], None
        if err:
            q.release(max_pages)
            metering.incident("serpapi", metering.FAILED, f"{engine}: {err}")
            return [], err
        cands = [_to_candidate(m, m.get("match_type") or match_type) for m in matches if m.get("title")]
        total = int((matches[0].get("total") or 0) if matches else 0)
        last_total[query] = total
        kv().put("search", ck, {"cands": [c.__dict__ for c in cands], "total": total})
        return cands, None
    return [], last_err or "serpapi: all keys exhausted"


async def search_patents(query: str, max_pages: int = 1, before: str | None = None,
                         scholar: bool = False) -> tuple[list[Candidate], str | None]:
    """`before` = Google Patents date bound, e.g. 'priority:20110202'
    (SerpAPI: type:YYYYMMDD, type in priority/filing/publication).
    `scholar=True` = SerpAPI's `scholar` parameter ("controls whether or not
    to include Google Scholar results"): papers come back in the same 100."""
    return await _search("google_patents", query, max_pages, 100, "Patent", before=before, scholar=scholar)


async def search_scholar(query: str, max_pages: int = 3) -> tuple[list[Candidate], str | None]:
    return await _search("google_scholar", query, max_pages, 20, "Paper")
