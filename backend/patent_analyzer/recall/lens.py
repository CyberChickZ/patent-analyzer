"""Lens.org recall channel — Scholarly API (paper -> citing patents bridge) and
Patent API (lookup by Lens id, bool search with CPC prefix + date cutoff).

Auth: token in LENS_API_TOKEN (backend/.env.yaml; never logged, never in a
cache key).  Docs (https://docs.api.lens.org/getting-started.html):
  "For POST Requests, you need to provide your access token in the Request
   Header when accessing the APIs: Example: Authorization: Bearer
   your-access-token"
Endpoints (same page): "[POST] https://api.lens.org/patent/search",
  "[POST] https://api.lens.org/scholarly/search",
  "[GET] https://api.lens.org/subscriptions/patent_api/usage".
Swagger: https://api.lens.org/swagger-ui/index.html -> /swagger.yaml
  (PatentSearchRequest / ScholarlySearchRequest: query, size, from, include,
   exclude, sort, scroll, scroll_id).

Rate limits — the docs give no numbers for a plan; getting-started.html only
says: "The applied rate limits will be included in the following HTTP
response headers: x-rate-limit-remaining-request-per-minute ...
x-rate-limit-remaining-request-per-month ...
x-rate-limit-remaining-record-per-month ... Once you go over any rate limit
you will receive a 429 - Too many requests error with respective messages."
The trial plan's numbers come from GET /subscriptions/{patent_api,
scholarly_api}/usage (checked 2026-09-18):
  patent_api:    1000 REQUEST / 1 MONTH, 100000 RECORD / 1 MONTH,
                 10 REQUEST / 1 MINUTE, maxRecordsPerRequest 100
  scholarly_api: 1000 REQUEST / 1 MONTH, 100000 RECORD / 1 MONTH,
                 20 REQUEST / 1 MINUTE, maxRecordsPerRequest 500
request-patent.html: "You can specify records per page using size (default 20
and max 100-500, refer to your API plan for your max records per request)";
request-scholar.html: "size (default 20 and max 1000)"; both: "For optimal
performance, we recommend limiting the number of items (e.g. lens_ids) in a
single terms query to 10,000."

Fields (request-scholar.html "Searchable Fields"): "ids.doi — Crossref DOI
Identifier", "ids.openalex — OpenAlex Identifier", "patent_citation.lens_id —
ID of Referenced by patents. N.B this field will be deprecated in future, we
recommend using the referenced_by_patent.lens_id field instead."
response-scholar.html: "patent_citations — Array of Patent Citation —
Referenced by patents", "patent_citations_count — Integer — Number of patent
citations" (each item is {"lens_id": "122-064-734-901-067"}).
request-patent.html: "lens_id", "doc_number", "jurisdiction", "kind",
"date_published — Date of publication for the patent document. e.g.
2009-05-22", "class_cpc.symbol — CPC patent classification codes. e.g.
H01R11/01", "reference_cited.npl.lens_id — The Lens Id of the resolved
non-patent literature citations (i.e. scholarly work Lens Id)",
"reference_cited.npl.external_id — The resolved external identifier(s) for
cited non-patent literature (DOI, PubMed ID, ...)"; CPC prefix per the
documented string query: '{ "query" : "class_cpc.symbol:Y02E10\\/70" }' ->
we send query_string "class_cpc.symbol:H04N7*".
response-patent.html: "biblio.invention_title", "biblio.classifications_cpc",
"families.simple_family — Simple patent family (based on DOCDB simple patent
family)" (members[].document_id{jurisdiction,doc_number,kind,date} +
lens_id).  The docs' sample shows "family_id": 212482337 inside
simple_family, but the live API rejects include=families.simple_family.family_id
("Unrecognized fields") and returns no family_id — we key a family by its
member publication numbers instead.

Attribution (https://about.lens.org/policies/ "Attribution"): "When using any
data obtained through our Services internally, in a product or service, or
including data in a redistribution, please acknowledge The Lens by including
the URL https://www.lens.org/, or a link to a relevant search or collection
on the lens.org domain. ... Please use the expression 'Enabled by The Lens'
or 'Data Sourced from The Lens' and the Lens.org URL. Please ensure that the
Lens logo and link is prominently displayed at the point of delivery of the
services or data derived from Lens, to the end-user."
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

import httpx

from ..cache import kv
from ..runtime_state import MinuteGate
from .pool import Candidate

API_BASE = "https://api.lens.org"
TIMEOUT = 60.0
_CACHE_NS = "lens"
_CACHE_DAYS = 30

# trial plan, from /subscriptions/*/usage (see module docstring)
PER_MINUTE = {"patent": 10, "scholarly": 20}
MAX_RECORDS = {"patent": 100, "scholarly": 500}

call_log: list[dict] = []
_gates: dict[str, MinuteGate] = {}

_SCHOLARLY_INCLUDE = ["lens_id", "title", "external_ids", "patent_citations", "patent_citations_count"]
_PATENT_INCLUDE = ["lens_id", "jurisdiction", "doc_number", "kind", "date_published", "publication_type",
                   "biblio.invention_title", "biblio.classifications_cpc.classifications.symbol",
                   "families.simple_family"]


def _token() -> str | None:
    return os.environ.get("LENS_API_TOKEN") or None


def _gate(endpoint: str) -> MinuteGate:
    if endpoint not in _gates:
        _gates[endpoint] = MinuteGate(f"lens_{endpoint}", PER_MINUTE[endpoint])
    return _gates[endpoint]


def _cache_key(endpoint: str, body: dict) -> str:
    return f"{endpoint}:" + hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _summary(body: dict) -> str:
    return json.dumps(body.get("query"), sort_keys=True)[:240]


async def _post(endpoint: str, body: dict) -> tuple[dict | None, str | None]:
    """POST one search body; KV-cached 30 days; MinuteGate-smoothed. Returns
    (response json, err). Never raises on HTTP errors; the token is never
    part of the log or the cache key."""
    ck = _cache_key(endpoint, body)
    hit = kv().get(_CACHE_NS, ck, max_age_days=_CACHE_DAYS)
    if hit is not None:
        call_log.append({"endpoint": endpoint, "params": _summary(body), "returned": len(hit.get("data") or []),
                         "total": hit.get("total"), "seconds": 0.0, "http_status": 200, "cached": True})
        return hit, None
    tok = _token()
    if not tok:
        return None, "lens: LENS_API_TOKEN not set"
    t0 = time.time()
    await _gate(endpoint).wait()
    status, err, data = 0, None, None
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(f"{API_BASE}/{endpoint}/search", json=body,
                                  headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
        status = r.status_code
        if status == 200:
            data = r.json()
        elif status == 204:
            data = {"total": 0, "results": 0, "data": []}
        elif status == 429:
            err = (f"lens {endpoint}: 429 rate limit (remaining/min="
                   f"{r.headers.get('x-rate-limit-remaining-request-per-minute')}, "
                   f"remaining/month={r.headers.get('x-rate-limit-remaining-request-per-month')}, "
                   f"retry-after={r.headers.get('x-rate-limit-retry-after-seconds')}s)")
        elif status == 401:
            err = f"lens {endpoint}: 401 unauthorized (token missing/expired)"
        elif status == 404:
            data = {"total": 0, "results": 0, "data": []}   # docs: 404 = empty result for the query
        else:
            msg = ""
            try:
                msg = (r.json() or {}).get("message", "")
            except Exception:
                msg = r.text[:120]
            err = f"lens {endpoint}: HTTP {status} {msg}"[:300]
    except Exception as e:
        err = f"lens {endpoint}: {type(e).__name__}: {e}"[:300]
    call_log.append({"endpoint": endpoint, "params": _summary(body),
                     "returned": len((data or {}).get("data") or []), "total": (data or {}).get("total"),
                     "seconds": round(time.time() - t0, 2), "http_status": status, "cached": False,
                     **({"err": err} if err else {})})
    if data is not None and not err:
        kv().put(_CACHE_NS, ck, data)
    return data, err
