"""Second hop from seed patents via our own BigQuery tables: examiner
(SEA) citations become new candidates; CPC subclass of the seeds feeds
the next query round. Point lookups only."""

from __future__ import annotations

import re
from collections import Counter

from ..recall.bigquery_patents import fetch_by_pub_nums, fetch_citations, fetch_meta_light
from ..recall.pool import Candidate

MAX_CITED_PER_ROUND = 200
MAX_CITED_LIGHT = int(__import__("os").environ.get("EXPAND_MAX_CITED", "2000"))
FULL_META_HEAD = 200   # abstracts for the most-cited head; titles only beyond


def _canon(p: str) -> str:
    return re.sub(r"[\s\-/,.]", "", (p or "").upper())


def _after(meta: dict, cutoff: str | None) -> bool:
    """priority_date on/after the cutoff (YYYYMMDD): not prior art, never a seed."""
    d = str(meta.get("priority_date") or "").replace("-", "")[:8]
    return bool(cutoff and d and d >= cutoff)


async def expand(seed_pubs: list[str], known: set[str], max_cited: int = MAX_CITED_PER_ROUND,
                 before: str | None = None, light: bool = False) -> tuple[list[Candidate], dict]:
    """Returns (new candidates from citations, info) where info has the
    seeds' family ids, CPC subclass counts and BQ stats. Seeds and cited
    documents with priority_date >= `before` (YYYYMMDD) are dropped.
    `light`: the head (FULL_META_HEAD most-cited) gets title+abstract, the
    rest title/family/date only via the narrow-column lookup — h1b lost 3
    gold families at citation ranks 1414-1463 to the 200 cap."""
    seeds = [s for s in dict.fromkeys(_canon(p) for p in seed_pubs if p) if s]
    info = {"seeds": len(seeds), "cpc_subclasses": {}, "families": {}, "cited_total": 0, "cited_new": 0,
            "seeds_after_cutoff": []}
    if not seeds:
        return [], info
    # light: hundreds of seeds only need family/date (16 GiB vs 4 GiB measured on 501 seeds)
    meta = await (fetch_meta_light(seeds) if light else fetch_by_pub_nums(seeds, with_claims=False))
    info["seeds_after_cutoff"] = sorted(k for k, m in meta.items() if _after(m, before))
    meta = {k: m for k, m in meta.items() if k not in info["seeds_after_cutoff"]}
    seeds = [s for s in seeds if s not in info["seeds_after_cutoff"]]
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
    if light and len(new) > FULL_META_HEAD:
        cmeta = await fetch_by_pub_nums(new[:FULL_META_HEAD], with_claims=False)
        tail = await fetch_meta_light(new[FULL_META_HEAD:])
        for pub, m in tail.items():
            cmeta.setdefault(pub, m)
        info["cited_light"] = len(tail)
    else:
        cmeta = await fetch_by_pub_nums(new, with_claims=False)
    out = []
    for pub, m in cmeta.items():
        if _after(m, before):
            continue
        out.append(Candidate(
            title=m.get("title") or pub, snippet=(m.get("abstract") or "")[:500], abstract=m.get("abstract") or "",
            match_type="Patent", pub_num=pub, year=str(m.get("priority_date") or "")[:4],
            url=f"https://patents.google.com/patent/{pub}/en", source_score=float(cited[pub]),
            sources=["citation_graph"],
            raw={"bigquery": {"family_id": m.get("family_id", ""), "priority_date": m.get("priority_date", ""),
                              "cpc_codes": m.get("cpc_codes") or [], "cited_by_seeds": cited[pub]}},
        ))
    return out, info
