"""BigQuery Patents Public Dataset recall channel.

Searches claim text in `patents-public-data.patents.publications`
for claim-level prior art matching. Free within GCP 1TB/month quota.
"""

import os
from patent_analyzer import metering
from patent_analyzer.recall.pool import Candidate


GC_PROJECT = os.getenv("GC_PROJECT", "aime-hello-world")
BQ_MAX_GIB = float(os.getenv("BQ_MAX_GIB_PER_QUERY", "20"))


# Wall-clock bound on every query. `job.result()` with no timeout waits forever:
# a slot-starved or stuck BigQuery job would otherwise pin the recall channel
# (and, before the channel budget in nodes/search.py, the whole job) open.
BQ_TIMEOUT_S = float(os.getenv("BQ_QUERY_TIMEOUT_S", "300"))


class BQBudgetExceeded(RuntimeError):
    pass


class BQTimeout(RuntimeError):
    pass


def _rows(job, timeout: float | None = None):
    """Wait for a query job, bounded. On timeout the job is cancelled — it is
    still billed for what it scanned, so leaving it running costs money as well
    as time — and the caller sees BQTimeout, which every call site already
    handles as "this channel produced nothing"."""
    import concurrent.futures
    try:
        return list(job.result(timeout=timeout if timeout is not None else BQ_TIMEOUT_S))
    except concurrent.futures.TimeoutError:
        try:
            job.cancel()
        except Exception:
            pass
        raise BQTimeout(f"BigQuery job {getattr(job, 'job_id', '?')} exceeded "
                        f"{timeout if timeout is not None else BQ_TIMEOUT_S:.0f}s; cancelled") from None


def guarded_query(client, sql: str, params=None, max_gib: float | None = None):
    """Dry-run first; refuse anything above max_gib and hard-cap billing.

    patents-public-data.patents.publications is not clustered: a point lookup
    that touches claims/description scans 300-1400 GiB (measured 2026-09-17,
    4.4 TiB billed in one evening). Every query in this module goes through
    here so a single call can never burn the monthly free tier again.
    """
    from google.cloud import bigquery
    cap = max_gib if max_gib is not None else BQ_MAX_GIB
    params = params or []
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, query_parameters=params))
    gib = dry.total_bytes_processed / 2 ** 30
    if gib > cap:
        raise BQBudgetExceeded(f"query would scan {gib:.1f} GiB > cap {cap} GiB")
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params, maximum_bytes_billed=int(cap * 2 ** 30) + 2 ** 20))
    rows = _rows(job)
    metering.count_bq(getattr(job, "total_bytes_billed", None) or gib * 2 ** 30)
    print(f"[BQ] scanned {gib:.2f} GiB, {len(rows)} rows")
    return rows


def capped_query(client, sql: str, params=None, max_gib: float = 10.0):
    """For SEARCH-indexed queries: the dry-run estimate ignores index pruning
    (reports the full column size), so skip it and rely on the hard billing
    cap — a job that exceeds maximum_bytes_billed fails and is not billed.
    Measured 2026-09-18 on amie_patents.abstracts: OR-of-terms 0.11 GiB,
    phrase OR 1.8 GiB, phrase AND 86 GiB. The cap itself is checked against
    the pre-execution estimate, so it must sit above the (inflated) estimate
    (~90 GiB for common tokens) — worst case ~$0.6 if the index is bypassed."""
    from google.cloud import bigquery
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params or [], maximum_bytes_billed=int(max_gib * 2 ** 30)))
    rows = _rows(job)
    metering.count_bq(job.total_bytes_billed)
    print(f"[BQ] billed {job.total_bytes_billed / 2 ** 30:.2f} GiB, {len(rows)} rows")
    return rows


async def search_claims(
    query_text: str,
    limit: int = 30,
    country_codes: list[str] | None = None,
) -> tuple[list[Candidate], str | None]:
    """Search BigQuery Patents for claims containing key terms from query_text.

    Uses SEARCH() for full-text matching on claims_localized text.
    Returns (candidates, error_string_or_None).
    """
    import asyncio
    from google.cloud import bigquery

    countries = country_codes or ["US", "EP", "WO", "CN", "JP", "KR"]
    country_filter = ", ".join(f"'{c}'" for c in countries)

    keywords = _extract_search_terms(query_text)
    if not keywords:
        return [], "No searchable terms extracted from claim"

    def _build_like(kws: list[str]) -> str:
        parts = []
        for kw in kws:
            safe = kw.lower().replace("'", "")
            parts.append(
                f"(LOWER(claims_localized[SAFE_OFFSET(0)].text) LIKE '%{safe}%'"
                f" OR LOWER(abstract_localized[SAFE_OFFSET(0)].text) LIKE '%{safe}%')"
            )
        return " AND ".join(parts)

    sql = f"""
    SELECT
      publication_number,
      title_localized[SAFE_OFFSET(0)].text AS title,
      abstract_localized[SAFE_OFFSET(0)].text AS abstract,
      claims_localized[SAFE_OFFSET(0)].text AS claims_text,
      country_code,
      filing_date,
      publication_date
    FROM `patents-public-data.patents.publications`
    WHERE country_code IN ({country_filter})
      AND (claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
           OR abstract_localized[SAFE_OFFSET(0)].text IS NOT NULL)
      AND {_build_like(keywords[:4])}
    LIMIT {limit}
    """

    try:
        client = bigquery.Client(project=GC_PROJECT)
        result = await asyncio.to_thread(guarded_query, client, sql)
        if not result and len(keywords) > 2:
            sql_fallback = f"""
            SELECT
              publication_number,
              title_localized[SAFE_OFFSET(0)].text AS title,
              abstract_localized[SAFE_OFFSET(0)].text AS abstract,
              claims_localized[SAFE_OFFSET(0)].text AS claims_text,
              country_code,
              filing_date,
              publication_date
            FROM `patents-public-data.patents.publications`
            WHERE country_code IN ({country_filter})
              AND (claims_localized[SAFE_OFFSET(0)].text IS NOT NULL
                   OR abstract_localized[SAFE_OFFSET(0)].text IS NOT NULL)
              AND {_build_like(keywords[:2])}
            LIMIT {limit}
            """
            # A second scan of the same table, on two keywords instead of four:
            # the recall is worse and it is billed again, so the ledger sees it.
            metering.incident("bigquery", metering.DEGRADED,
                              f"{len(keywords[:4])} keywords returned nothing; re-ran on {len(keywords[:2])}")
            result = await asyncio.to_thread(guarded_query, client, sql_fallback)
    except Exception as e:
        metering.incident("bigquery", metering.FAILED, f"{type(e).__name__}: {e}")
        return [], f"BigQuery error: {type(e).__name__}: {e}"

    candidates = []
    for row in result:
        pub_num = row.publication_number or ""
        title = row.title or pub_num
        abstract = row.abstract or ""
        claims = row.claims_text or ""

        c = Candidate(
            title=title,
            snippet=abstract[:500] if abstract else claims[:500],
            abstract=abstract,
            match_type="Patent",
            pub_num=pub_num,
            source_score=1.0,
            sources=["bigquery_patents"],
        )
        c.raw = {
            "bigquery": {
                "publication_number": pub_num,
                "claims_text": claims[:8000],
                "country_code": row.country_code,
                "filing_date": str(row.filing_date) if row.filing_date else "",
            }
        }
        candidates.append(c)

    return candidates, None


def _extract_search_terms(text: str, max_terms: int = 5) -> list[str]:
    """Extract key single words from text for SQL LIKE matching."""
    import re
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    stop = {
        "a", "an", "the", "of", "in", "on", "for", "and", "or", "by", "to",
        "with", "from", "is", "at", "as", "its", "via", "using", "based",
        "method", "system", "apparatus", "device", "comprising", "wherein",
        "step", "configured", "adapted", "claim", "claims", "said",
        "controlled", "powered", "delivery", "drug", "present", "novel",
        "designed", "proposed", "describes", "proposes", "conceptualized",
        "therapeutic", "function", "components", "enhanced", "described",
        "document", "invention", "provides", "includes", "related",
        "transient", "requires", "completing", "fully", "core", "body",
    }
    words = [w for w in text.split() if w not in stop and len(w) > 3]
    words = sorted(set(words), key=len, reverse=True)
    return words[:max_terms]


async def search_by_limitations(
    limitations: list[str],
    limit_per_limitation: int = 10,
) -> tuple[list[Candidate], str | None]:
    """Search for each claim limitation separately, then merge results."""
    all_candidates = []
    errors = []
    for lim in limitations[:10]:
        cands, err = await search_claims(lim, limit=limit_per_limitation)
        all_candidates.extend(cands)
        if err:
            errors.append(err)

    # Dedup by publication_number
    seen = set()
    deduped = []
    for c in all_candidates:
        if c.pub_num and c.pub_num not in seen:
            seen.add(c.pub_num)
            deduped.append(c)
        elif not c.pub_num:
            deduped.append(c)

    return deduped, "; ".join(errors) if errors else None


async def fetch_meta_light(pub_nums: list[str], max_gib: float = 15.0) -> dict[str, dict]:
    """title / family_id / priority_date only, for thousands of numbers at
    once. BigQuery bills the columns read: on amie_patents.pubs the
    (publication_number, family_id, priority_date, title) columns are 12.7 GiB
    for the whole table (dry-run 2026-09-18), whereas fetch_by_pub_nums
    also reads abstract + cpc_codes (~0.03 GiB per row). Same bucket
    pruning; keys are canonical."""
    import asyncio
    import re
    from google.cloud import bigquery

    norm = {re.sub(r"[\s\-/,.]", "", p.upper()): p for p in pub_nums if p}
    if not norm:
        return {}
    wanted = [_bq_form(p) for p in norm]
    client = bigquery.Client(project=GC_PROJECT)
    pubs_param = bigquery.ArrayQueryParameter("pubs", "STRING", wanted)

    def _run():
        b = client.query(
            "SELECT ARRAY(SELECT MOD(ABS(FARM_FINGERPRINT(p)), 4000) FROM UNNEST(@pubs) p) AS b",
            job_config=bigquery.QueryJobConfig(query_parameters=[pubs_param]))
        buckets = _rows(b, timeout=60)[0].b
        params = [pubs_param, bigquery.ArrayQueryParameter("buckets", "INT64", buckets)]
        return guarded_query(client, f"""
            SELECT publication_number, family_id, priority_date, title
            FROM `{GC_PROJECT}.amie_patents.pubs`
            WHERE bucket IN UNNEST(@buckets) AND publication_number IN UNNEST(@pubs)""", params,
            max_gib=max_gib)

    rows = await asyncio.to_thread(_run)
    return {_canon_pub(r.publication_number): {"family_id": r.family_id or "", "priority_date": str(r.priority_date or ""),
                                              "title": r.title or ""} for r in rows}


async def fetch_cited_by(pub_nums: list[str], max_gib: float = 8.0) -> dict[str, list[dict]]:
    """Forward citations from amie_patents.cited_by (copied once from
    google_patents_research.publications.cited_by, 21.6 GiB, bucketed on
    publication_number): pub → [{publication_number, category, filing_date}].
    Keys canonical. patent-search-pilot keeps forward edges at weight 1
    (cited 3 / family 2 / citing 1)."""
    import asyncio
    import re
    from google.cloud import bigquery

    norm = {re.sub(r"[\s\-/,.]", "", p.upper()): p for p in pub_nums if p}
    if not norm:
        return {}
    wanted = [_bq_form(p) for p in norm]
    client = bigquery.Client(project=GC_PROJECT)
    pubs_param = bigquery.ArrayQueryParameter("pubs", "STRING", wanted)

    def _run():
        b = client.query(
            "SELECT ARRAY(SELECT MOD(ABS(FARM_FINGERPRINT(p)), 4000) FROM UNNEST(@pubs) p) AS b",
            job_config=bigquery.QueryJobConfig(query_parameters=[pubs_param]))
        buckets = _rows(b, timeout=60)[0].b
        params = [pubs_param, bigquery.ArrayQueryParameter("buckets", "INT64", buckets)]
        return guarded_query(client, f"""
            SELECT publication_number, cited_by FROM `{GC_PROJECT}.amie_patents.cited_by`
            WHERE bucket IN UNNEST(@buckets) AND publication_number IN UNNEST(@pubs)""", params, max_gib=max_gib)

    rows = await asyncio.to_thread(_run)
    return {_canon_pub(r.publication_number): [{"publication_number": _canon_pub(x.get("publication_number") or ""),
                                                "category": x.get("category") or "", "filing_date": x.get("filing_date")}
                                               for x in (r.cited_by or [])] for r in rows}


async def fetch_similar(pub_nums: list[str], max_gib: float = 12.0) -> dict[str, list[str]]:
    """Google's own semantic neighbours: amie_patents.similar (copied once
    from google_patents_research.publications.similar — the embedding_v1
    nearest neighbours Google publishes, ~25 per patent; 117 GiB one-time,
    point lookups afterwards). pub → [neighbour pubs], keys canonical."""
    import asyncio
    import re
    from google.cloud import bigquery

    norm = {re.sub(r"[\s\-/,.]", "", p.upper()): p for p in pub_nums if p}
    if not norm:
        return {}
    wanted = [_bq_form(p) for p in norm]
    client = bigquery.Client(project=GC_PROJECT)
    pubs_param = bigquery.ArrayQueryParameter("pubs", "STRING", wanted)

    def _run():
        b = client.query(
            "SELECT ARRAY(SELECT MOD(ABS(FARM_FINGERPRINT(p)), 4000) FROM UNNEST(@pubs) p) AS b",
            job_config=bigquery.QueryJobConfig(query_parameters=[pubs_param]))
        buckets = _rows(b, timeout=60)[0].b
        params = [pubs_param, bigquery.ArrayQueryParameter("buckets", "INT64", buckets)]
        return guarded_query(client, f"""
            SELECT publication_number, similar FROM `{GC_PROJECT}.amie_patents.similar`
            WHERE bucket IN UNNEST(@buckets) AND publication_number IN UNNEST(@pubs)""", params, max_gib=max_gib)

    rows = await asyncio.to_thread(_run)
    return {_canon_pub(r.publication_number): [_canon_pub(x.get("publication_number") or "") for x in (r.similar or [])
                                               if x.get("publication_number")] for r in rows}


def _bq_form(p: str) -> str:
    import re
    m = re.match(r"^([A-Z]{2})((?:RE|PP|D|H|T)?\d+)([A-Z]\d?)?$", p)
    if not m:
        return p
    cc, digits, kind = m.group(1), m.group(2), m.group(3) or ""
    if cc == "US" and len(digits) == 11 and digits[4] == "0":
        digits = digits[:4] + digits[5:]
    return f"{cc}-{digits}-{kind}".rstrip("-")


def _canon_pub(p: str) -> str:
    import re
    p = re.sub(r"[\s\-]", "", p.upper())
    m = re.match(r"^US(\d{10})([A-Z]\d?)?$", p)
    if m and m.group(1)[:2] in ("19", "20"):
        return f"US{m.group(1)[:4]}0{m.group(1)[4:]}{m.group(2) or ''}"
    return p



async def fetch_by_pub_nums(pub_nums: list[str], with_claims: bool = True,
                            with_description: bool = False) -> dict[str, dict]:
    """Full text for known publication numbers from our own bucketed copy.

    amie_patents.pubs / amie_patents.claims are hash-partitioned on the
    publication number (4000 buckets) and clustered, so a lookup scans
    ~50 MiB instead of the 300-1400 GiB a point lookup costs on the public
    table (measured 2026-09-17). Buckets are computed by a zero-byte query
    so client and table agree on FARM_FINGERPRINT.

    Accepts any spelling ('US9075557B2', 'US-9075557-B2'); keys of the
    returned dict are the canonical no-separator form.
    """
    import asyncio
    import re
    from google.cloud import bigquery

    norm = {re.sub(r"[\s\-/,.]", "", p.upper()): p for p in pub_nums if p}
    if not norm:
        return {}

    # The module-level _bq_form / _canon_pub, not local copies. The copies that
    # used to live here had drifted: their number pattern was a bare `\d+`, so
    # every reissue, design and plant publication (US-RE41525-E, US-D712191-S,
    # US-PP20104-P2) was spelled back wrong and silently found nothing.
    # Measured on 300 publications drawn from amie_patents.claims itself:
    # 125 hits before, 300 after.
    _canon = _canon_pub
    wanted = [_bq_form(p) for p in norm]
    client = bigquery.Client(project=GC_PROJECT)
    pubs_param = bigquery.ArrayQueryParameter("pubs", "STRING", wanted)

    def _run():
        b = client.query(
            "SELECT ARRAY(SELECT MOD(ABS(FARM_FINGERPRINT(p)), 4000) FROM UNNEST(@pubs) p) AS b",
            job_config=bigquery.QueryJobConfig(query_parameters=[pubs_param]))
        buckets = _rows(b, timeout=60)[0].b
        params = [pubs_param, bigquery.ArrayQueryParameter("buckets", "INT64", buckets)]
        meta = guarded_query(client, f"""
            SELECT publication_number, family_id, country_code, priority_date, publication_date,
                   title, abstract, cpc_codes
            FROM `{GC_PROJECT}.amie_patents.pubs`
            WHERE bucket IN UNNEST(@buckets) AND publication_number IN UNNEST(@pubs)""", params,
            max_gib=2 + 0.03 * len(wanted))
        claims = {}
        if with_claims:
            for r in guarded_query(client, f"""
                SELECT publication_number, claims_text FROM `{GC_PROJECT}.amie_patents.claims`
                WHERE bucket IN UNNEST(@buckets) AND publication_number IN UNNEST(@pubs)""", params,
                max_gib=2 + 0.03 * len(wanted)):
                claims[r.publication_number] = r.claims_text or ""
        descs = {}
        if with_description:
            # descriptions is 510 GiB over the same 4000 buckets, so a bucket
            # costs 0.128 GiB against the 0.028 GiB a claims bucket costs
            # (measured: 60 publications = 7.6 GiB). Sharing the claims budget
            # is what made the first run of this fetch refuse every chunk.
            for r in guarded_query(client, f"""
                SELECT publication_number, description FROM `{GC_PROJECT}.amie_patents.descriptions`
                WHERE bucket IN UNNEST(@buckets) AND publication_number IN UNNEST(@pubs)""", params,
                max_gib=2 + DESC_GIB_PER_PUB * len(wanted)):
                descs[r.publication_number] = r.description or ""
        return meta, claims, descs

    meta, claims, descs = await asyncio.to_thread(_run)
    out = {}
    for r in meta:
        key = _canon(r.publication_number)
        out[key] = {
            "publication_number": key,
            "title": r.title or "", "abstract": r.abstract or "",
            "claims_text": claims.get(r.publication_number, ""),
            "description": descs.get(r.publication_number, ""),
            "cpc_codes": list(r.cpc_codes or []), "country_code": r.country_code or "",
            "priority_date": str(r.priority_date or ""), "publication_date": str(r.publication_date or ""),
            "family_id": r.family_id or "",
        }
    return out


async def search_abstracts(
    query_terms: list[str],
    limit: int = 30,
    before: str | None = None,
    country_codes: list[str] | None = None,
    max_gib: float = 30.0,
) -> tuple[list[Candidate], str | None]:
    """Ranked keyword recall over amie_patents.abstracts (title+abstract, 2000+).

    SEARCH() is only a filter and its index prunes at block level, so any
    query with ORDER BY reads 20-90 GiB (measured 2026-09-18; common tokens
    sit in every block). This function therefore (a) ranks matched rows by
    number of query terms present so results are usable, (b) prunes by
    priority-year partition when `before` is given, (c) is meant to be
    called ONCE per job. It is a paid channel (~$0.1-0.5/call), kept for
    evals; a local BM25 index is the free replacement.
    """
    import asyncio
    import re
    from google.cloud import bigquery

    words = []
    for t in query_terms:
        for w in re.findall(r"[a-z][a-z0-9\-]{3,}", (t or "").lower()):
            if w not in words:
                words.append(w)
    words = words[:10]
    if not words:
        return [], "no search terms"
    # A bare hyphen is a token the SEARCH() query parser rejects outright:
    # `ultra-wideband OR kuramoto` -> "400 Search query parser error: error at
    # position 8: token recognition error at: '-'", which killed the whole
    # channel for any invention whose terms contain a hyphenated word (measured
    # live 2026-09-18, and the same dry run succeeds once the term is quoted).
    # Backticks are SEARCH()'s own quoting, and they keep the phrase intact
    # instead of splitting it into two loose tokens.
    def _q(w: str) -> str:
        return f"`{w}`" if "-" in w else w

    search_expr = " OR ".join(_q(w) for w in words)
    score = " + ".join(f"IF(REGEXP_CONTAINS(t, r'\\b{re.escape(w)}'), 1, 0)" for w in words)
    filters = ["SEARCH((title, abstract), @q)"]
    params = [bigquery.ScalarQueryParameter("q", "STRING", search_expr),
              bigquery.ScalarQueryParameter("lim", "INT64", limit)]
    if before:
        filters.append("prio_year <= @py")
        filters.append("priority_date < @before")
        params.append(bigquery.ScalarQueryParameter("py", "INT64", int(str(before)[:4])))
        params.append(bigquery.ScalarQueryParameter("before", "INT64", int(before)))
    if country_codes:
        filters.append("country_code IN UNNEST(@cc)")
        params.append(bigquery.ArrayQueryParameter("cc", "STRING", country_codes))
    sql = f"""
    WITH m AS (
      SELECT publication_number, country_code, family_id, priority_date, title, abstract,
             LOWER(CONCAT(title, ' ', abstract)) AS t
      FROM `{GC_PROJECT}.amie_patents.abstracts`
      WHERE {' AND '.join(filters)})
    SELECT publication_number, country_code, family_id, priority_date, title, abstract, ({score}) AS score
    FROM m ORDER BY score DESC LIMIT @lim"""
    try:
        client = bigquery.Client(project=GC_PROJECT)
        rows = await asyncio.to_thread(capped_query, client, sql, params, max_gib)
    except Exception as e:
        return [], f"BigQuery error: {type(e).__name__}: {e}"
    out = []
    for r in rows:
        pub = r.publication_number.replace("-", "")
        out.append(Candidate(
            title=r.title or pub, snippet=(r.abstract or "")[:500], abstract=r.abstract or "",
            match_type="Patent", pub_num=pub, source_score=float(r.score), sources=["bigquery_patents"],
            year=str(r.priority_date or "")[:4],
            url=f"https://patents.google.com/patent/{pub}/en",
            raw={"bigquery": {"publication_number": pub, "family_id": r.family_id,
                              "country_code": r.country_code, "priority_date": str(r.priority_date or ""),
                              "term_hits": int(r.score)}},
        ))
    return out, None


async def fetch_citations(pub_nums: list[str]) -> dict[str, dict]:
    """Citation lists for known publications from amie_patents.citations
    (bucketed like pubs). Each entry: {family_id, priority_date,
    cits: [{cited, type, category, npl_text}]}.
    category: 'SEA' = search report / examiner, 'APP' = applicant (IDS),
    'PRS' = ?, 'UNKNOWN'; multi-valued as 'APP,APP'. type: EP-style
    relevance 'X' / 'Y' / 'A' (mostly empty for US). Verified 2026-09-18 on
    bucket 7: category APP 52.6k / SEA 17.5k / PRS 14.2k.
    Keys are canonical no-separator numbers."""
    import asyncio
    import re
    from google.cloud import bigquery

    norm = {re.sub(r"[\s\-/,.]", "", p.upper()): p for p in pub_nums if p}
    if not norm:
        return {}

    def _bq_form(p: str) -> str:
        m = re.match(r"^([A-Z]{2})(\d+)([A-Z]\d?)?$", p)
        if not m:
            return p
        cc, digits, kind = m.group(1), m.group(2), m.group(3) or ""
        if cc == "US" and len(digits) == 11 and digits[4] == "0":
            digits = digits[:4] + digits[5:]
        return f"{cc}-{digits}-{kind}".rstrip("-")

    wanted = [_bq_form(p) for p in norm]
    client = bigquery.Client(project=GC_PROJECT)
    pubs_param = bigquery.ArrayQueryParameter("pubs", "STRING", wanted)

    def _run():
        b = client.query("SELECT ARRAY(SELECT MOD(ABS(FARM_FINGERPRINT(p)), 4000) FROM UNNEST(@pubs) p) AS b",
                         job_config=bigquery.QueryJobConfig(query_parameters=[pubs_param]))
        buckets = _rows(b, timeout=60)[0].b
        params = [pubs_param, bigquery.ArrayQueryParameter("buckets", "INT64", buckets)]
        return guarded_query(client, f"""
            SELECT publication_number, family_id, priority_date, cits
            FROM `{GC_PROJECT}.amie_patents.citations`
            WHERE bucket IN UNNEST(@buckets) AND publication_number IN UNNEST(@pubs)""", params,
            max_gib=2 + 0.01 * len(wanted))

    rows = await asyncio.to_thread(_run)
    out = {}
    for r in rows:
        key = re.sub(r"[\s\-]", "", r.publication_number.upper())
        out[key] = {"family_id": r.family_id, "priority_date": str(r.priority_date or ""),
                    "cits": [{"cited": (x.get("cited") or "").replace("-", ""), "type": x.get("type") or "",
                              "category": x.get("category") or "", "npl_text": x.get("npl_text") or ""} for x in r.cits]}
    return out


async def fetch_families(family_ids: list[str]) -> dict[str, list[dict]]:
    """family_id -> [{publication_number, country_code, priority_date,
    publication_date}] from amie_patents.families (bucketed on family_id)."""
    import asyncio
    from google.cloud import bigquery

    fams = sorted({f for f in family_ids if f})
    if not fams:
        return {}
    client = bigquery.Client(project=GC_PROJECT)
    fam_param = bigquery.ArrayQueryParameter("fams", "STRING", fams)

    def _run():
        b = client.query("SELECT ARRAY(SELECT MOD(ABS(FARM_FINGERPRINT(f)), 4000) FROM UNNEST(@fams) f) AS b",
                         job_config=bigquery.QueryJobConfig(query_parameters=[fam_param]))
        buckets = _rows(b, timeout=60)[0].b
        params = [fam_param, bigquery.ArrayQueryParameter("buckets", "INT64", buckets)]
        return guarded_query(client, f"""
            SELECT family_id, members FROM `{GC_PROJECT}.amie_patents.families`
            WHERE bucket IN UNNEST(@buckets) AND family_id IN UNNEST(@fams)""", params,
            max_gib=2 + 0.01 * len(fams))

    rows = await asyncio.to_thread(_run)
    return {r.family_id: [{"publication_number": m.get("publication_number", "").replace("-", ""),
                           "country_code": m.get("country_code", ""),
                           "priority_date": str(m.get("priority_date") or ""),
                           "publication_date": str(m.get("publication_date") or "")} for m in r.members]
            for r in rows}


def _canon_oa_id(x: str) -> str:
    """'https://openalex.org/W123', 'w123', '123' -> 'W123'."""
    x = (x or "").strip().rsplit("/", 1)[-1].upper()
    if x.isdigit():
        x = "W" + x
    return x


def _pcs_point_lookup(client, table: str, key_col: str, keys: list[str], cols: str, max_gib: float):
    """Bucket-pruned IN lookup on an amie_patents.pcs_oa* table: bucket ids
    come from a zero-byte constant query so the dry-run estimate reflects
    partition pruning (same trick as fetch_by_pub_nums)."""
    from google.cloud import bigquery
    key_param = bigquery.ArrayQueryParameter("keys", "STRING", keys)
    b = client.query("SELECT ARRAY(SELECT MOD(ABS(FARM_FINGERPRINT(k)), 4000) FROM UNNEST(@keys) k) AS b",
                     job_config=bigquery.QueryJobConfig(query_parameters=[key_param]))
    buckets = _rows(b, timeout=60)[0].b
    params = [key_param, bigquery.ArrayQueryParameter("buckets", "INT64", buckets)]
    return guarded_query(client, f"""
        SELECT {cols} FROM `{GC_PROJECT}.amie_patents.{table}`
        WHERE bucket IN UNNEST(@buckets) AND {key_col} IN UNNEST(@keys)""", params, max_gib=max_gib)


async def fetch_citing_patents(oa_ids: list[str], max_gib: float = 5.0) -> dict[str, list[dict]]:
    """Paper -> USPTO patents citing it, from amie_patents.pcs_oa (Reliance on
    Science pcs_oa_uspto.csv, Zenodo 21493744, granted through 2025; bucketed
    on oa_id). Keys are canonical OpenAlex ids ('W123'); each entry:
    {patent_pub (canonical, e.g. 'US10494607B2'), reftype ('app' for
    99.998% of rows — the USPTO file carries no usable examiner flag),
    confscore (4-10), wherefound ('frontonly' / 'bodyonly' / 'both'),
    grant_year (NULL for grants before amie_patents.pubs coverage),
    family_id}. Papers with no citing patent are absent from the result."""
    import asyncio
    from google.cloud import bigquery

    keys = sorted({_canon_oa_id(x) for x in oa_ids if x and _canon_oa_id(x)})
    if not keys:
        return {}
    client = bigquery.Client(project=GC_PROJECT)
    rows = await asyncio.to_thread(
        _pcs_point_lookup, client, "pcs_oa", "oa_id", keys,
        "oa_id, patent_pub, reftype, confscore, wherefound, grant_year, family_id", max_gib)
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r.oa_id, []).append({
            "patent_pub": _canon_pub(r.patent_pub), "reftype": r.reftype or "",
            "confscore": int(r.confscore or 0), "wherefound": r.wherefound or "",
            "grant_year": int(r.grant_year) if r.grant_year else None, "family_id": r.family_id or ""})
    return out


async def fetch_cited_papers(patent_pubs: list[str], max_gib: float = 5.0) -> dict[str, list[dict]]:
    """Reverse bridge: USPTO patent -> OpenAlex papers it cites, from
    amie_patents.pcs_oa_by_patent (same rows as pcs_oa, bucketed on
    patent_pub). Keys are canonical publication numbers (_canon_pub);
    each entry: {oa_id, reftype, confscore, wherefound}. The file is
    USPTO-only but includes ~1M pre-grant publications (US-2015133390-A1),
    so application numbers do match. reftype is 'app' for 99.998% of rows
    (846 'exm' in 34.8M, counted 2026-09-18; the 2024 release is the same
    for its USPTO rows) — it does not separate examiner citations."""
    import asyncio
    import re
    from google.cloud import bigquery

    norm = sorted({re.sub(r"[\s\-/,.]", "", p.upper()) for p in patent_pubs if p})
    if not norm:
        return {}
    keys = [_bq_form(p) for p in norm]
    client = bigquery.Client(project=GC_PROJECT)
    rows = await asyncio.to_thread(
        _pcs_point_lookup, client, "pcs_oa_by_patent", "patent_pub", keys,
        "patent_pub, oa_id, reftype, confscore, wherefound", max_gib)
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(_canon_pub(r.patent_pub), []).append({
            "oa_id": r.oa_id, "reftype": r.reftype or "", "confscore": int(r.confscore or 0),
            "wherefound": r.wherefound or ""})
    return out


# ── Deep-read text source ──────────────────────────────────────────────────
#
# What we actually have. Measured on amie_patents, 2026-09-18:
#
#   pubs       150,176,247 rows / 130.7 GiB — bucket, publication_number,
#              family_id, country_code, priority_date, publication_date,
#              filing_date, kind_code, title, abstract, cpc_codes
#   claims      17,792,253 rows / 111.9 GiB — bucket, publication_number,
#              claims_text.  `SELECT SUBSTR(publication_number,1,2), COUNT(*)`
#              returns exactly one row: US.  17,792,253 of the 17,797,011 US
#              publications in `pubs` (99.97%); zero for CN/JP/EP/KR/WO/DE/…
#
# There is **no description / specification column in any table of the
# dataset**.  So the best text this pipeline can put in front of the evaluator,
# without leaving our own data, is:
#
#   US publication      -> title + abstract + claims_text + description
#   non-US publication  -> title + abstract only
#
# `descriptions` was built on 2026-09-18 from
# patents-public-data.patents.publications (1.0295 TiB scanned, $6.43 once) and
# truncated at 40,000 characters, which keeps 51.5% of the text at $10.91/month
# of logical storage instead of $21.19 for the full column.  The cut is not a
# loss of reachable evidence so much as a budget for the prompt: 40,000
# characters is already ~10k tokens per document beside the claims.
#
# It is US-only, and so is `claims`, because the public source is.  Measured
# 2026-09-18 on patents-public-data.patents.publications, claims_localized:
#
#   US 18,760,602 rows with claims (122.3 GB of text)
#   CN 0 · JP 0 · EP 0 · KR 0 · WO 0 · DE 0
#
# So "no specification for a non-US publication" is a fact about the data
# available, not a gap in this pipeline, and the report says
# "abstract + claims only" rather than leaving it blank
# (report_sections._EV_LABEL, graph.eval_subgraph read_as).
FULLTEXT_CHUNK = int(os.getenv("BQ_FULLTEXT_CHUNK", "300"))
# The description pass is chunked separately and smaller. A bucket of
# `descriptions` scans 0.128 GiB against 0.028 GiB for `claims` (measured:
# 60 publications = 7.6 GiB), so 300 publications would be ~38 GiB -- over the
# 30 GiB ceiling a single query is allowed. 150 is ~19 GiB.
DESC_CHUNK = int(os.getenv("BQ_DESC_CHUNK", "150"))
DESC_GIB_PER_PUB = float(os.getenv("BQ_DESC_GIB_PER_PUB", "0.15"))


def _is_patent_doc(doc: dict) -> bool:
    import re
    if doc.get("match_type") == "Patent":
        return True
    return bool(re.match(r"^[A-Z]{2}[-\s]?\d", (doc.get("pub_num") or "").upper()))


async def hydrate_full_text(docs: list[dict], chunk: int | None = None,
                            desc_chunk: int | None = None) -> dict:
    """Fill `claims_text` (and any missing abstract) on patent docs from our own
    bucketed copy, so the deep read does not depend on a PDF download.

    Chunked at `FULLTEXT_CHUNK` publications per query: 300 claims rows measured
    17.4 GiB, and the single-query ceiling is 30 GiB.  Every chunk still goes
    through `guarded_query`, so a chunk that would blow the budget is refused
    rather than billed.

    Returns counts, never raises: a chunk that fails leaves those docs as they
    were and is reported in `errors`, because the caller's fallback (PDF, then
    abstract, then nothing) is still valid.
    """
    chunk = chunk or FULLTEXT_CHUNK
    desc_chunk = desc_chunk or DESC_CHUNK
    stats = {"asked": 0, "chunks": 0, "desc_chunks": 0, "hit": 0,
             "with_claims": 0, "with_description": 0, "errors": []}
    want = [d for d in docs if _is_patent_doc(d)
            and (d.get("pub_num") or "").strip()
            and not (d.get("claims_text") or "").strip()]
    if not want:
        return stats
    by_pub: dict[str, list[dict]] = {}
    for d in want:
        by_pub.setdefault(_canon_pub(d["pub_num"]), []).append(d)
    pubs = list(by_pub)
    stats["asked"] = len(pubs)
    for i in range(0, len(pubs), chunk):
        batch = pubs[i:i + chunk]
        stats["chunks"] += 1
        try:
            got = await fetch_by_pub_nums(batch, with_claims=True)
        except Exception as exc:
            stats["errors"].append(f"claims: {type(exc).__name__}: {exc}"[:200])
            continue
        for key, row in got.items():
            for d in by_pub.get(key, []):
                stats["hit"] += 1
                claims = (row.get("claims_text") or "").strip()
                if claims:
                    d["claims_text"] = claims
                    stats["with_claims"] += 1
                if not (d.get("abstract") or "").strip() and row.get("abstract"):
                    d["abstract"] = row["abstract"]
                if not (d.get("title") or "").strip() and row.get("title"):
                    d["title"] = row["title"]

    # Second pass, smaller chunks and its own failure handling: the
    # specification is an improvement on the claims, never a precondition for
    # them, and the first run of this code lost every claim to a description
    # query that would not fit the shared budget.
    desc_want = [p for p, ds in by_pub.items()
                 if any((d.get("claims_text") or "").strip() for d in ds)
                 and not any((d.get("description") or "").strip() for d in ds)]
    for i in range(0, len(desc_want), desc_chunk):
        batch = desc_want[i:i + desc_chunk]
        stats["desc_chunks"] += 1
        try:
            got = await fetch_by_pub_nums(batch, with_claims=False, with_description=True)
        except Exception as exc:
            stats["errors"].append(f"description: {type(exc).__name__}: {exc}"[:200])
            continue
        for key, row in got.items():
            desc = (row.get("description") or "").strip()
            if not desc:
                continue
            for d in by_pub.get(key, []):
                d["description"] = desc
                stats["with_description"] += 1
    return stats
