"""BigQuery Patents Public Dataset recall channel.

Searches claim text in `patents-public-data.patents.publications`
for claim-level prior art matching. Free within GCP 1TB/month quota.
"""

import os
from patent_analyzer.recall.pool import Candidate


GC_PROJECT = os.getenv("GC_PROJECT", "aime-hello-world")


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
        result = await asyncio.to_thread(
            lambda: list(client.query(sql).result())
        )
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
            result = await asyncio.to_thread(
                lambda: list(client.query(sql_fallback).result())
            )
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
