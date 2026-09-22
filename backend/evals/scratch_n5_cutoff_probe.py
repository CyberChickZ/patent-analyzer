#!/usr/bin/env python3
"""N5 probe: is the paper channel's 0 the cutoff filter, and does asking the
API for the date range instead bring the gold back?

paper_neighbourhood fetches S2/OpenAlex results and then drops everything
published after the application's priority date (_pre_cutoff). h1h logged
s2_search:* = 0 for every candidate of every case. This measures, per query we
actually issued: how much of the unfiltered top-100 survives the cutoff, and
where the examiner-cited NPL ranks once the date range is asked of the API.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import httpx

OUT = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature" / "n5_cutoff_probe.json"
CUTOFF = {"US20100036217A1": 2006, "US20120058475A1": 2006,
          "US20120144509A1": 2009, "US20150313878A1": 2012}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:120]


def _same(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a[:80], b[:80]).ratio() >= 0.9


async def s2_search(q: str, limit: int, year: str | None):
    from patent_analyzer.recall import semantic_scholar as ss
    params = {"query": q[:300], "limit": limit, "fields": ss.DEFAULT_FIELDS}
    if year:
        params["year"] = year
    async with httpx.AsyncClient() as client:
        data, err = await ss._get(client, f"{ss.API_BASE}/paper/search", params)
    return [(p.get("title") or "", p.get("year")) for p in ((data or {}).get("data") or [])], err


async def oa_search(q: str, limit: int, year: int | None):
    from patent_analyzer.recall import openalex as oa
    p = {"search": q[:300], "per_page": limit, "select": "id,title,publication_year"}
    if year:
        p["filter"] = f"publication_year:<{year + 1}"
    async with httpx.AsyncClient() as client:
        data, err = await oa._get(client, f"{oa.API_BASE}/works", oa._params(p))
    return [(w.get("title") or "", w.get("publication_year")) for w in ((data or {}).get("results") or [])], err


async def main():
    from common import load_env_yaml
    load_env_yaml()
    from scratch_n5_npl_autopsy import golds, our_queries

    gs = golds()
    rows = []
    for key, cut in CUTOFF.items():
        mine = [g for g in gs if g["key"] == key]
        for q in our_queries(key):
            raw, e1 = await s2_search(q, 100, None)
            filt, e2 = await s2_search(q, 100, f"-{cut}")
            ofilt, e3 = await oa_search(q, 100, cut)
            pre = sum(1 for _, y in raw if y and int(y) <= cut)
            r = {"key": key, "query": q, "s2_raw": len(raw), "s2_raw_pre_cutoff": pre,
                 "s2_filtered": len(filt), "oa_filtered": len(ofilt),
                 "err": [x for x in (e1, e2, e3) if x], "gold_found": []}
            for g in mine:
                nt = _norm(g["title"])
                for label, lst in (("s2_year", filt), ("oa_year", ofilt)):
                    for i, (t, _y) in enumerate(lst, 1):
                        if _same(nt, _norm(t)):
                            r["gold_found"].append({"title": g["title"][:60], "where": label, "rank": i})
                            break
            rows.append(r)
            print(f"{key} [{q[:52]:<52}] s2 {len(raw):>3} (pre-cutoff {pre:>3}) "
                  f"s2+year {len(filt):>3}  oa+year {len(ofilt):>3}  {r['gold_found'] or ''}")
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    tot = sum(r["s2_raw"] for r in rows)
    pre = sum(r["s2_raw_pre_cutoff"] for r in rows)
    print(f"\nunfiltered S2 results that survive the priority-date cutoff: {pre}/{tot} = {pre / max(tot, 1):.3f}")
    print(f"S2 asked for the date range directly: {sum(r['s2_filtered'] for r in rows)} results")
    print(f"OpenAlex asked for the date range directly: {sum(r['oa_filtered'] for r in rows)} results")
    hits = {(h['title'], h['where']) for r in rows for h in r["gold_found"]}
    print(f"gold NPL reachable by a date-ranged version of a query we already issue: {sorted(hits)}")


if __name__ == "__main__":
    asyncio.run(main())
