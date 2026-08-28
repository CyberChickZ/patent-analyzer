"""USPTO Open Data Portal — the third free patent channel, and the file wrapper.

Verified live against the API on 2026-09-18 (the docs site is a JS shell that
serves no text to a fetcher, so every statement below is what the API itself
answered):

  base            https://api.uspto.gov/api/v1
  auth            header `X-API-KEY: <key>` (lower-case `x-api-key` also works;
                  `api-key` and `Authorization: Bearer` both answer 401)
  search          GET /patent/applications/search?q=<lucene>&limit=&offset=
                  -> {"count": <total>, "patentFileWrapperDataBag": [...]}
  one application GET /patent/applications/{appNum}
  file wrapper    GET /patent/applications/{appNum}/documents
                  -> {"count", "documentBag": [{documentCode,
                     documentCodeDescriptionText, officialDate,
                     documentIdentifier, downloadOptionBag:[{mimeTypeIdentifier,
                     downloadUrl}]}]}
  download        GET /download/applications/{appNum}/{documentId}.pdf

Searchable fields that answered (counts from the live index):
  applicationMetaData.cpcClassificationBag:"H04N   7/15"   -> 6,257
  applicationMetaData.cpcClassificationBag:H04N*           -> 423,653
  applicationMetaData.inventionTitle:telepresence          -> 583
  applicationMetaData.inventionTitle:"telepresence robot"  -> 66
  applicationMetaData.filingDate:[2010-01-01 TO 2010-12-31]-> 465,819
  AND of the three                                         -> 66
CPC symbols are stored padded ("H04N   7/15"), so a main group is matched with
a prefix wildcard. `applicationMetaData` carries no abstract and no claims:
this index is title + classification + dates + names only. What it gives that
Google Patents does not is COMPLETENESS — `count` is the real total and
`offset` pages through all of it, against Google's hard top-100.

File wrapper document codes seen on one application: SRNT "Examiner's search
strategy and results", 892 / 1449 / IDS (references cited), CTNF / CTFR
(office actions), NOA, CLM, REM. The SRNT PDF we pulled has no text layer
(scanned) — OCR would be needed to read an examiner's own queries.

Limits (leader, from the key's registration): 60 req/min, concurrency 1,
weekly 1.2M file-wrapper requests and 5M metadata requests. The per-minute
gate and the 1-request-at-a-time lock are shared across processes like the
Semantic Scholar ones; the weekly counters live in the KV.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.parse
import urllib.request

from ..cache import kv
from ..runtime_state import MinuteGate, PeriodQuota, SerialLock, week_key
from .pool import Candidate

BASE = "https://api.uspto.gov/api/v1"
PER_MINUTE = int(os.environ.get("ODP_PER_MINUTE", "60"))
COOLDOWN_S = float(os.environ.get("ODP_COOLDOWN_S", "1.0"))
WEEKLY_METADATA = int(os.environ.get("ODP_WEEKLY_METADATA", "5000000"))
WEEKLY_WRAPPER = int(os.environ.get("ODP_WEEKLY_WRAPPER", "1200000"))
CACHE_DAYS = float(os.environ.get("ODP_CACHE_DAYS", "30"))
_NS = "odp"

call_log: list[dict] = []
_gate = MinuteGate("uspto_odp", PER_MINUTE)


def _key() -> str | None:
    return os.environ.get("USPTO_ODP_API_KEY") or None


def quota(kind: str) -> PeriodQuota:
    cap = WEEKLY_WRAPPER if kind == "wrapper" else WEEKLY_METADATA
    return PeriodQuota(f"odp_{kind}", cap, week_key)


def quota_status() -> list[dict]:
    return [{"kind": k, "used": quota(k).used(), "cap": quota(k).cap, "week": week_key()}
            for k in ("metadata", "wrapper")]


async def _get(path: str, params: dict | None = None, kind: str = "metadata",
               binary: bool = False) -> tuple[object | None, str | None]:
    """One GET, smoothed to PER_MINUTE and serialised with a cooldown; JSON
    responses are KV-cached. Returns (payload, error)."""
    k = _key()
    if not k:
        return None, "USPTO_ODP_API_KEY not set"
    url = f"{BASE}{path}" + (("?" + urllib.parse.urlencode(params)) if params else "")
    ck = urllib.parse.quote(url[len(BASE):], safe="")[:300]
    if not binary:
        hit = kv().get(_NS, ck, max_age_days=CACHE_DAYS)
        if hit is not None:
            call_log.append({"path": path, "cached": True, "status": 200})
            return hit.get("payload"), None
    if not quota(kind).take():
        return None, f"odp weekly {kind} quota exhausted"
    t0 = time.monotonic()

    def _call():
        req = urllib.request.Request(url, headers={"X-API-KEY": k, "Accept": "*/*" if binary else "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    try:
        await _gate.wait()
        async with SerialLock("uspto_odp", COOLDOWN_S):
            status, body = await asyncio.to_thread(_call)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:200]
        call_log.append({"path": path, "error": err, "seconds": round(time.monotonic() - t0, 2)})
        return None, err
    call_log.append({"path": path, "status": status, "bytes": len(body),
                     "seconds": round(time.monotonic() - t0, 2)})
    if binary:
        return body, None
    payload = json.loads(body)
    kv().put(_NS, ck, {"payload": payload})
    return payload, None


def _cpc_prefix(cpc: str) -> str:
    r"""CPC symbols are stored right-padded ("H04N   7/15", "H04N  70/00"), so a
    main group is a prefix match on the padded string with every space escaped.
    Measured 2026-09-18: `H04N\ \ \ 7*` -> 68,691 hits, `H04N*` -> 423,653,
    `H04N7*` -> none, and a quoted phrase with a trailing * matches the whole
    index (13.6M), so neither of those is a main-group filter."""
    c = cpc.split("/")[0].strip().upper()
    sub, num = c[:4], c[4:]
    if not num:
        return f"{sub}*"
    return sub + "\\ " * max(0, 4 - len(num)) + num + "*"


def _q(title_terms: list[str], cpc: str | None, before: str | None) -> str:
    """Lucene query over the metadata index. `before` is YYYYMMDD (the E4
    priority cutoff): the filing date must be strictly earlier."""
    parts = []
    terms = [t for t in (title_terms or []) if t.strip()]
    if terms:
        ors = " OR ".join(f'"{t}"' if " " in t else t for t in terms)
        parts.append(f"applicationMetaData.inventionTitle:({ors})")
    if cpc:
        parts.append(f"applicationMetaData.cpcClassificationBag:{_cpc_prefix(cpc)}")
    if before and len(before) == 8 and before.isdigit():
        d = f"{before[:4]}-{before[4:6]}-{before[6:]}"
        parts.append(f"applicationMetaData.filingDate:[1900-01-01 TO {d}]")
    return " AND ".join(parts)


def _candidate(w: dict) -> Candidate | None:
    m = w.get("applicationMetaData") or {}
    pub = (m.get("earliestPublicationNumber") or "").strip()
    pat = (m.get("patentNumber") or "").strip()
    num = f"US{pat}B2" if pat else pub
    if not num:
        return None
    year = (m.get("grantDate") or m.get("earliestPublicationDate") or m.get("filingDate") or "")[:4]
    return Candidate(title=(m.get("inventionTitle") or "").strip(), pub_num=num, match_type="Patent", year=year,
                     url=f"https://patents.google.com/patent/{num}", sources=["uspto_odp"],
                     raw={"application_number": w.get("applicationNumberText"),
                          "cpc": [c.strip() for c in (m.get("cpcClassificationBag") or [])][:12],
                          "filing_date": m.get("filingDate"), "examiner": m.get("examinerNameText"),
                          "art_unit": m.get("groupArtUnitNumber")})


async def search_patents(title_terms: list[str], cpc: str | None = None, before: str | None = None,
                         size: int = 100, offset: int = 0) -> tuple[list[Candidate], int | None, str | None]:
    """Title-word / CPC / filing-date search. Returns (candidates, total, error).
    The metadata index has no abstract or claims — title words only."""
    q = _q(title_terms, cpc, before)
    if not q:
        return [], None, "empty query"
    data, err = await _get("/patent/applications/search", {"q": q, "limit": min(size, 100), "offset": offset})
    if err and "404" in err:
        return [], 0, None              # the API answers 404 with "No matching records found"
    if err or not isinstance(data, dict):
        return [], None, err or "bad response"
    out = [c for c in (_candidate(w) for w in data.get("patentFileWrapperDataBag") or []) if c]
    return out, data.get("count"), None


async def enumerate_group(title_terms: list[str], cpc: str, before: str | None = None,
                          max_records: int = 3000) -> tuple[list[Candidate], int | None, str | None]:
    """Every application in a CPC main group, before the cutoff, whose title
    carries any of the terms — paged to exhaustion.

    This is what no other channel of ours can do. Google Patents answers a
    query with its top 100 whatever the field size, and 18 blind queries in the
    gold's own main group never surfaced H1-02's US20100010703A1 (§H7.4).
    Through this endpoint the same family is simply there: `inventionTitle:
    guidance AND cpcClassificationBag:G05D\ \ \ 1* AND filingDate:[… TO
    2013-10-31]` returns 162 records and application 12216582 is one of them
    (verified 2026-09-18 by paging all 162).
    """
    out: list[Candidate] = []
    total: int | None = None
    for offset in range(0, max_records, 100):
        got, n, err = await search_patents(title_terms, cpc=cpc, before=before, size=100, offset=offset)
        if err:
            return out, total, err
        total = n if total is None else total
        out.extend(got)
        if not got or (total is not None and len(out) >= total):
            break
    return out, total, None


async def application(app_num: str) -> tuple[dict | None, str | None]:
    data, err = await _get(f"/patent/applications/{app_num}")
    if err or not isinstance(data, dict):
        return None, err or "bad response"
    bag = data.get("patentFileWrapperDataBag") or []
    return (bag[0] if bag else None), None


async def documents(app_num: str) -> tuple[list[dict], str | None]:
    """The file wrapper's document list (office actions, 892/1449 citations,
    SRNT examiner search strategy, …)."""
    data, err = await _get(f"/patent/applications/{app_num}/documents", kind="wrapper")
    if err or not isinstance(data, dict):
        return [], err or "bad response"
    return list(data.get("documentBag") or []), None


async def download(app_num: str, document_id: str, mime: str = "PDF") -> tuple[bytes | None, str | None]:
    ext = {"PDF": "pdf", "XML": "xml", "MS_WORD": "docx"}.get(mime.upper(), "pdf")
    body, err = await _get(f"/download/applications/{app_num}/{document_id}.{ext}", kind="wrapper", binary=True)
    return (body if isinstance(body, (bytes, bytearray)) else None), err


async def find_by_publication(pub_num: str) -> tuple[dict | None, str | None]:
    """Application record for a US publication or patent number (for the file
    wrapper of a gold reference)."""
    n = "".join(ch for ch in pub_num.upper() if ch.isalnum())
    n = n[2:] if n.startswith("US") else n
    n = n.rstrip("AB").rstrip("0123456789") and n or n
    for field in ("applicationMetaData.earliestPublicationNumber", "applicationMetaData.patentNumber"):
        digits = "".join(ch for ch in pub_num if ch.isdigit())
        value = pub_num.upper() if field.endswith("PublicationNumber") else digits
        data, err = await _get("/patent/applications/search", {"q": f'{field}:"{value}"', "limit": 1})
        if isinstance(data, dict) and (data.get("patentFileWrapperDataBag") or []):
            return data["patentFileWrapperDataBag"][0], None
    return None, "not found"
