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


def _norm_doi(d: str) -> str:
    d = (d or "").strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d[4:] if d.startswith("doi:") else d


def _norm_oa(w: str) -> str:
    w = (w or "").strip().upper()
    w = re.sub(r"^HTTPS?://OPENALEX\.ORG/", "", w)
    return w


async def scholarly_by_ids(dois: list[str], oa_ids: list[str]) -> tuple[dict[str, dict], str | None]:
    """Papers -> the patents that cite them (Lens 'patent_citations').

    Terms queries on ids.doi / ids.openalex, batched so one request never
    asks for more than MAX_RECORDS['scholarly'] records.  Returns
    {key: {lens_id, title, patent_citations_count, patent_citations: [lens
    patent ids]}} keyed by the DOI / OpenAlex id as given (lower-cased DOI,
    upper-cased W-id); a paper matched by both ids appears under both keys.
    """
    dois_n = {_norm_doi(d): d for d in dois if d}
    oas_n = {_norm_oa(w): w for w in oa_ids if w}
    out: dict[str, dict] = {}
    first_err = None
    per = MAX_RECORDS["scholarly"]
    items = [("ids.doi", k) for k in dois_n] + [("ids.openalex", k) for k in oas_n]
    for i in range(0, len(items), per):
        chunk = items[i:i + per]
        d_chunk = [k for f, k in chunk if f == "ids.doi"]
        w_chunk = [k for f, k in chunk if f == "ids.openalex"]
        should = []
        if d_chunk:
            should.append({"terms": {"ids.doi": d_chunk}})
        if w_chunk:
            should.append({"terms": {"ids.openalex": w_chunk}})
        query = should[0] if len(should) == 1 else {"bool": {"should": should}}
        body = {"query": query, "size": per, "include": _SCHOLARLY_INCLUDE}
        data, err = await _post("scholarly", body)
        if err:
            first_err = first_err or err
            continue
        w_set, d_set = set(w_chunk), set(d_chunk)
        for w in data.get("data") or []:
            rec = {"lens_id": w.get("lens_id"), "title": w.get("title") or "",
                   "patent_citations_count": int(w.get("patent_citations_count") or 0),
                   "patent_citations": [pc.get("lens_id") for pc in (w.get("patent_citations") or []) if pc.get("lens_id")]}
            for ext in w.get("external_ids") or []:
                typ, val = (ext.get("type") or "").lower(), ext.get("value") or ""
                if typ == "doi" and _norm_doi(val) in d_set:
                    out[_norm_doi(val)] = rec
                elif typ == "openalex" and _norm_oa(val) in w_set:
                    out[_norm_oa(val)] = rec
                elif typ == "magid" and f"W{val}" in w_set:
                    # OpenAlex W-ids inherit MAG ids; Lens matches ids.openalex but only
                    # echoes the magid in external_ids
                    out[f"W{val}"] = rec
    return out, first_err


def _pub_num(doc_id: dict) -> str:
    j, n, k = (doc_id.get("jurisdiction") or "").upper(), str(doc_id.get("doc_number") or ""), (doc_id.get("kind") or "").upper()
    return f"{j}{n}{k}" if j and n else ""


def _to_candidate(p: dict, source: str) -> Candidate:
    pub = _pub_num(p)
    titles = ((p.get("biblio") or {}).get("invention_title") or [])
    title = ""
    for t in titles:
        if (t.get("lang") or "").lower() == "en":
            title = t.get("text") or ""
            break
    if not title and titles:
        title = titles[0].get("text") or ""
    cpc = []
    for c in (((p.get("biblio") or {}).get("classifications_cpc") or {}).get("classifications") or []):
        s = c.get("symbol")
        if s and s not in cpc:
            cpc.append(s)
    members = (((p.get("families") or {}).get("simple_family") or {}).get("members") or [])
    fam_pubs = sorted({_pub_num(m.get("document_id") or {}) for m in members} - {""})
    fam_lens = sorted({m.get("lens_id") for m in members if m.get("lens_id")})
    date = p.get("date_published") or ""
    return Candidate(
        title=title, url=f"https://www.lens.org/lens/patent/{p.get('lens_id')}" if p.get("lens_id") else "",
        pub_num=pub, match_type="Patent", year=date[:4], sources=[source],
        raw={"lens": {"lens_id": p.get("lens_id"), "date_published": date,
                      "publication_type": p.get("publication_type"),
                      "family": fam_pubs, "family_lens_ids": fam_lens,
                      "family_key": fam_lens[0] if fam_lens else p.get("lens_id"),
                      "cpc": cpc}})


async def patents_by_lens_ids(lens_ids: list[str], source: str = "lens_bridge") -> tuple[list[Candidate], str | None]:
    """Patent records for Lens patent ids (terms on lens_id), batched at
    MAX_RECORDS['patent'] per request."""
    ids = list(dict.fromkeys(i for i in lens_ids if i))
    out: list[Candidate] = []
    first_err = None
    per = MAX_RECORDS["patent"]
    for i in range(0, len(ids), per):
        body = {"query": {"terms": {"lens_id": ids[i:i + per]}}, "size": per, "include": _PATENT_INCLUDE}
        data, err = await _post("patent", body)
        if err:
            first_err = first_err or err
            continue
        out.extend(_to_candidate(p, source) for p in data.get("data") or [])
    return out, first_err


def _iso(yyyymmdd: str) -> str:
    s = re.sub(r"\D", "", yyyymmdd or "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else yyyymmdd


def search_body(query_terms: list[str], cpc: str | None, before: str | None, size: int) -> dict:
    """bool query: multi-word terms as match_phrase on title/abstract/claims,
    single words as match; at least one must hit; CPC prefix and
    date_published < before as filters."""
    should = []
    for t in query_terms:
        t = " ".join((t or "").split())
        if not t:
            continue
        for fld in ("title", "abstract", "claim"):
            should.append({"match_phrase" if " " in t else "match": {fld: t}})
    filt: list[dict] = []
    if cpc:
        sym = re.sub(r"[^A-Za-z0-9/]", "", cpc).upper()
        filt.append({"query_string": {"query": f"class_cpc.symbol:{sym.replace('/', chr(92) + '/')}*"}})
    if before:
        filt.append({"range": {"date_published": {"lt": _iso(before)}}})
    q: dict[str, Any] = {"bool": {"must": [{"bool": {"should": should, "minimum_should_match": 1}}]}}
    if filt:
        q["bool"]["filter"] = filt
    return {"query": q, "size": min(size, MAX_RECORDS["patent"]), "include": _PATENT_INCLUDE}


async def search_patents(query_terms: list[str], cpc: str | None = None, before: str | None = None,
                         size: int = 100) -> tuple[list[Candidate], str | None]:
    """Keyword search of the Patent API (one request, <= MAX_RECORDS['patent']
    records). `before` is YYYYMMDD; results carry the simple family."""
    terms = [t for t in query_terms if t and t.strip()]
    if not terms:
        return [], None
    data, err = await _post("patent", search_body(terms, cpc, before, size))
    if err:
        return [], err
    cands = [_to_candidate(p, "lens_search") for p in data.get("data") or []]
    total = data.get("total")
    for c in cands:
        c.raw["lens"]["total"] = total
    return cands, None
