"""BigQuery Patents Public Dataset recall channel.

Searches claim text in `patents-public-data.patents.publications`
for claim-level prior art matching. Free within GCP 1TB/month quota.
"""

import os
from patent_analyzer.recall.pool import Candidate


GC_PROJECT = os.getenv("GC_PROJECT", "aime-hello-world")
BQ_MAX_GIB = float(os.getenv("BQ_MAX_GIB_PER_QUERY", "20"))


class BQBudgetExceeded(RuntimeError):
    pass


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
    rows = list(job.result())
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
    rows = list(job.result())
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
            result = await asyncio.to_thread(guarded_query, client, sql_fallback)
    except Exception as e:
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


async def fetch_by_pub_nums(pub_nums: list[str], with_claims: bool = True) -> dict[str, dict]:
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

    def _bq_form(p: str) -> str:
        m = re.match(r"^([A-Z]{2})(\d+)([A-Z]\d?)?$", p)
        if not m:
            return p
        cc, digits, kind = m.group(1), m.group(2), m.group(3) or ""
        # BigQuery spells US pre-grant numbers year+6 digits (US-2012287933-A1),
        # USPTO/FiNE spell them year+7 with a leading zero (US20120287933A1)
        if cc == "US" and len(digits) == 11 and digits[4] == "0":
            digits = digits[:4] + digits[5:]
        return f"{cc}-{digits}-{kind}".rstrip("-")

    def _canon(p: str) -> str:
        p = re.sub(r"[\s\-]", "", p.upper())
        m = re.match(r"^US(\d{10})([A-Z]\d?)?$", p)
        if m and m.group(1)[:2] in ("19", "20"):
            return f"US{m.group(1)[:4]}0{m.group(1)[4:]}{m.group(2) or ''}"
        return p

    wanted = [_bq_form(p) for p in norm]
    client = bigquery.Client(project=GC_PROJECT)
    pubs_param = bigquery.ArrayQueryParameter("pubs", "STRING", wanted)

    def _run():
        b = client.query(
            "SELECT ARRAY(SELECT MOD(ABS(FARM_FINGERPRINT(p)), 4000) FROM UNNEST(@pubs) p) AS b",
            job_config=bigquery.QueryJobConfig(query_parameters=[pubs_param]))
        buckets = list(b.result())[0].b
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
        return meta, claims

    meta, claims = await asyncio.to_thread(_run)
    out = {}
    for r in meta:
        key = _canon(r.publication_number)
        out[key] = {
            "publication_number": key,
            "title": r.title or "", "abstract": r.abstract or "",
            "claims_text": claims.get(r.publication_number, ""),
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
    max_gib: float = 100.0,
) -> tuple[list[Candidate], str | None]:
    """Keyword recall over amie_patents.abstracts (title+abstract, 2000+)
    through its SEARCH index: only matching rows are scanned, so a query
    costs MBs rather than the 327 GiB the old LIKE scan cost.

    query_terms: OR-ed within, i.e. any term hits. Pass phrases in quotes.
    before: 'YYYYMMDD' priority-date cutoff (leakage control in evals).
    """
    import asyncio
    from google.cloud import bigquery

    terms = [t.strip() for t in query_terms if t and len(t.strip()) > 2][:8]
    if not terms:
        return [], "no search terms"
    search_expr = " OR ".join(f"`{t}`" if " " in t else t for t in terms)
    filters = ["SEARCH((title, abstract), @q)"]
    params = [bigquery.ScalarQueryParameter("q", "STRING", search_expr),
              bigquery.ScalarQueryParameter("lim", "INT64", limit)]
    if before:
        filters.append("priority_date < @before")
        params.append(bigquery.ScalarQueryParameter("before", "INT64", int(before)))
    if country_codes:
        filters.append("country_code IN UNNEST(@cc)")
        params.append(bigquery.ArrayQueryParameter("cc", "STRING", country_codes))
    sql = f"""
    SELECT publication_number, country_code, family_id, priority_date, title, abstract
    FROM `{GC_PROJECT}.amie_patents.abstracts`
    WHERE {' AND '.join(filters)}
    LIMIT @lim"""
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
            match_type="Patent", pub_num=pub, source_score=1.0, sources=["bigquery_patents"],
            year=str(r.priority_date or "")[:4],
            url=f"https://patents.google.com/patent/{pub}/en",
            raw={"bigquery": {"publication_number": pub, "family_id": r.family_id,
                              "country_code": r.country_code, "priority_date": str(r.priority_date or "")}},
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
        buckets = list(b.result())[0].b
        params = [pubs_param, bigquery.ArrayQueryParameter("buckets", "INT64", buckets)]
        return guarded_query(client, f"""
            SELECT publication_number, family_id, priority_date, cits
            FROM `{GC_PROJECT}.amie_patents.citations`
            WHERE bucket IN UNNEST(@buckets) AND publication_number IN UNNEST(@pubs)""", params,
            max_gib=2 + 0.01 * len(wanted))  # ~7 MiB per touched partition

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
        buckets = list(b.result())[0].b
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
