#!/usr/bin/env python3
"""N5 probe: paper-side queries built from the NAMED facets instead of the
claim-language element bag.

The current neighbourhood.keyword_queries joins the `patent`/`thing` facets of
the first elements into one 7-word bag ("engineered bacterium bacterial
chromosome in-frame insertion recombinant"). That is patent prose; a paper is
titled with the name of the assay, gene, protein or reagent — which Phase 2
already extracted into the `named` facet and which the paper channel never
uses. This measures the gold rank for `named`-derived queries, blind: the
queries come only from loop_elements, never from the gold.
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

OUT = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature" / "n5_named_probe.json"
CUTOFF = {"US20100036217A1": 2006, "US20120058475A1": 2006,
          "US20120144509A1": 2009, "US20150313878A1": 2012}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:120]


def _same(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a[:80], b[:80]).ratio() >= 0.9


async def main():
    from common import load_env_yaml
    load_env_yaml()
    from patent_analyzer.agentic.neighbourhood import named_queries
    from scratch_n5_cutoff_probe import oa_search, s2_search
    from scratch_n5_npl_autopsy import golds, load_rec

    gs = golds()
    rows, best = [], {}
    for key, cut in CUTOFF.items():
        rec = load_rec(key)
        by_cand: dict[str, list[dict]] = {}
        for e in rec.get("loop_elements") or []:
            by_cand.setdefault(e.get("candidate") or e["id"].split(".")[0], []).append(e)
        cands = [{"id": cid, "elements": es} for cid, es in by_cand.items()]
        qs = []
        for c in cands[:4]:
            for q in named_queries(c):
                if q not in qs:
                    qs.append(q)
        mine = [g for g in gs if g["key"] == key]
        for q in qs:
            s2, e1 = await s2_search(q, 100, f"-{cut}")
            oa, e2 = await oa_search(q, 100, cut)
            found = []
            for g in mine:
                nt = _norm(g["title"])
                for label, lst in (("s2", s2), ("oa", oa)):
                    for i, (t, _y) in enumerate(lst, 1):
                        if _same(nt, _norm(t)):
                            found.append({"title": g["title"][:60], "where": label, "rank": i})
                            k = g["title"]
                            if k not in best or i < best[k]["rank"]:
                                best[k] = {"query": q, "where": label, "rank": i}
                            break
            rows.append({"key": key, "query": q, "s2": len(s2), "oa": len(oa),
                         "err": [x for x in (e1, e2) if x], "gold_found": found})
            print(f"{key} [{q[:44]:<44}] s2 {len(s2):>3} oa {len(oa):>3}  "
                  f"{[(f['where'], f['rank'], f['title'][:34]) for f in found] or ''}")
    OUT.write_text(json.dumps({"rows": rows, "best": best}, ensure_ascii=False, indent=1))
    print(f"\ngold NPL reachable by a named-facet query (top-100 of one channel): {len(best)}/{len(gs)}")
    for t, v in sorted(best.items()):
        print(f"  #{v['rank']:<4}{v['where']:<4}[{v['query'][:40]:<40}] {t[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
