#!/usr/bin/env python3
"""N5 probe: one citation hop out of the named-query hits.

The named-facet queries reach 7 of the 17 examiner-cited NPL in the top-100 of
one channel; the rest are the field's own background papers ("Role of vacA and
the cagA locus…"), which no single keyword query puts in a top-100 because the
topic has tens of thousands of papers. On the patent side the same shape was
solved by the graph (19 of 34 h1h gold families first arrived through citation
expansion, H.md §H1.5). This asks whether the paper side behaves the same:
take the top hits of the named queries and pull their references.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

OUT = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature" / "n5_hop_probe.json"
OUT2 = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature" / "n5_hop_probe_anyyear.json"
CUTOFF = {"US20100036217A1": 2006, "US20120058475A1": 2006,
          "US20120144509A1": 2009, "US20150313878A1": 2012}
SEEDS_PER_CASE = 20
SEED_YEAR = os.environ.get("N5_SEED_YEAR", "1") == "1"
TOP_PER_QUERY = 5


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:120]


def _same(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a[:80], b[:80]).ratio() >= 0.9


async def main():
    from common import load_env_yaml
    load_env_yaml()
    import httpx

    from patent_analyzer.agentic.neighbourhood import named_queries
    from patent_analyzer.recall import semantic_scholar as ss
    from scratch_n5_npl_autopsy import golds, load_rec

    gs = golds()
    rows = []
    for key, cut in CUTOFF.items():
        rec = load_rec(key)
        by_cand: dict[str, list[dict]] = {}
        for e in rec.get("loop_elements") or []:
            by_cand.setdefault(e.get("candidate") or e["id"].split(".")[0], []).append(e)
        qs: list[str] = []
        for cid, es in list(by_cand.items())[:4]:
            for q in named_queries({"id": cid, "elements": es}):
                if q not in qs:
                    qs.append(q)
        seeds: list[tuple[str, str]] = []
        for q in qs:
            params = {"query": q[:300], "limit": 20, "fields": ss.DEFAULT_FIELDS}
            if SEED_YEAR:
                params["year"] = f"-{cut}"
            async with httpx.AsyncClient() as client:
                data, _ = await ss._get(client, f"{ss.API_BASE}/paper/search", params)
            for p in ((data or {}).get("data") or [])[:TOP_PER_QUERY]:
                if p.get("paperId") and all(p["paperId"] != s[0] for s in seeds):
                    seeds.append((p["paperId"], p.get("title") or ""))
            if len(seeds) >= SEEDS_PER_CASE:
                break
        seeds = seeds[:SEEDS_PER_CASE]
        hop: dict[str, str] = {}
        for pid, stitle in seeds:
            refs, _ = await ss.references_all(pid, 200)
            for c in refs:
                hop.setdefault(_norm(c.title), stitle)
        mine = [g for g in gs if g["key"] == key]
        for g in mine:
            nt = _norm(g["title"])
            via = hop.get(nt) or next((v for t, v in hop.items() if _same(nt, t)), "")
            rows.append({"key": key, "title": g["title"], "hop_hit": bool(via), "via": via[:70],
                         "seeds": len(seeds), "hop_size": len(hop)})
            print(f"{key:<18}{'HOP  ' if via else '  -  '}{g['title'][:58]:<60}{via[:40]}")
        print(f"   seeds={len(seeds)} refs={len(hop)}")
    (OUT if SEED_YEAR else OUT2).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    n = sum(1 for r in rows if r["hop_hit"])
    print(f"\nreachable by one reference hop out of the named-query hits: {n}/{len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
