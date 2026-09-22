#!/usr/bin/env python3
"""N5 after-measurement: re-run the real paper_neighbourhood for the four h1h
cases that have examiner-cited NPL, with the same inputs the h1h job gave it,
and count how many of the 17 gold papers come back.

Before = the whole h1h pool (a superset of what the neighbourhood produced):
3/17 by fuzzy title, 2/17 by the exact match L1-G used.
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

OUT = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature" / "n5_after.json"
CUTOFF = {"US20100036217A1": "20061011", "US20120058475A1": "20060920",
          "US20120144509A1": "20090603", "US20150313878A1": "20121211"}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:120]


def _same(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a[:80], b[:80]).ratio() >= 0.9


async def main():
    from common import load_env_yaml
    load_env_yaml()
    from patent_analyzer.agentic.neighbourhood import paper_neighbourhood
    from scratch_n5_npl_autopsy import golds, load_rec

    gs = golds()
    rows = []
    for key, cutoff in CUTOFF.items():
        rec = load_rec(key)
        # the job attaches facets to the elements (loop.attach_facets) before the
        # neighbourhood runs; the stored extraction predates that call, so splice the
        # facets the run actually had back in
        facets = {e["id"]: e.get("facets") for e in (rec.get("loop_elements") or [])}
        cands = rec["extraction"]["candidate_inventions"]
        for c in cands:
            for e in c.get("elements") or []:
                if facets.get(e["id"]):
                    e["facets"] = facets[e["id"]]
        title = (rec.get("events") and "") or ""
        # the job passed state["source_title"]; the arxiv channel logged it verbatim
        for ch in rec.get("channel_stats") or []:
            for e in ch.get("errors") or []:
                if e.get("query"):
                    title = e["query"]
        papers, info = await paper_neighbourhood(title, cands, cutoff=cutoff,
                                                 summary=rec.get("summary", ""))
        idx = [_norm(p.title) for p in papers]
        for g in [x for x in gs if x["key"] == key]:
            nt = _norm(g["title"])
            hit = any(_same(nt, t) for t in idx)
            rows.append({"key": key, "title": g["title"], "in_neighbourhood": hit})
            print(f"{key:<18}{'HIT ' if hit else '  - '}{g['title'][:64]}")
        print(f"   title={title[:60]!r} papers={len(papers)} sources={info['sources']}")
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    n = sum(1 for r in rows if r["in_neighbourhood"])
    print(f"\npaper channel alone, after: {n}/{len(rows)} = {n / len(rows):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
