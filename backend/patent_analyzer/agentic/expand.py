"""Second hop from seed patents via our own BigQuery tables: examiner
(SEA) citations become new candidates; CPC subclass of the seeds feeds
the next query round. Point lookups only."""

from __future__ import annotations

import re
from collections import Counter

from ..recall.bigquery_patents import fetch_by_pub_nums, fetch_citations
from ..recall.pool import Candidate

MAX_CITED_PER_ROUND = 200


def _canon(p: str) -> str:
    return re.sub(r"[\s\-/,.]", "", (p or "").upper())


async def expand(seed_pubs: list[str], known: set[str], max_cited: int = MAX_CITED_PER_ROUND
                 ) -> tuple[list[Candidate], dict]:
    """Returns (new candidates from citations, info) where info has the
    seeds' family ids, CPC subclass counts and BQ stats."""
    seeds = [s for s in dict.fromkeys(_canon(p) for p in seed_pubs if p) if s]
    info = {"seeds": len(seeds), "cpc_subclasses": {}, "families": {}, "cited_total": 0, "cited_new": 0}
    if not seeds:
        return [], info
    meta = await fetch_by_pub_nums(seeds, with_claims=False)
    cpc = Counter()
    for k, m in meta.items():
        info["families"][k] = m.get("family_id", "")
        for c in m.get("cpc_codes") or []:
            cpc[c[:4]] += 1
    info["cpc_subclasses"] = dict(cpc.most_common(5))

    cits = await fetch_citations(seeds)
    cited = Counter()
    for s, c in cits.items():
        for x in c.get("cits", []):
            pub = _canon(x.get("cited", ""))
            if not pub or x.get("npl_text"):
                continue
            weight = 2 if "SEA" in (x.get("category") or "") else 1
            cited[pub] += weight
    info["cited_total"] = len(cited)
    new = [p for p, _ in cited.most_common() if p not in known and p not in meta][:max_cited]
    info["cited_new"] = len(new)
    if not new:
        return [], info
    cmeta = await fetch_by_pub_nums(new, with_claims=False)
    out = []
    for pub, m in cmeta.items():
        out.append(Candidate(
            title=m.get("title") or pub, snippet=(m.get("abstract") or "")[:500], abstract=m.get("abstract") or "",
            match_type="Patent", pub_num=pub, year=str(m.get("priority_date") or "")[:4],
            url=f"https://patents.google.com/patent/{pub}/en", source_score=float(cited[pub]),
            sources=["citation_graph"],
            raw={"bigquery": {"family_id": m.get("family_id", ""), "priority_date": m.get("priority_date", ""),
                              "cpc_codes": m.get("cpc_codes") or [], "cited_by_seeds": cited[pub]}},
        ))
    return out, info
