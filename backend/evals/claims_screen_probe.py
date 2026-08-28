#!/usr/bin/env python3
"""Does reading claims rank a gold reference that abstracts cannot?

h1o showed the claims screen inheriting the abstract screen's ordering problem:
US20150126796A1 kept 5,434 documents worth reading, the two gold families sat
at rank 598 and 1,841 of that order, and the claims screen only reads the top
`PRUNE_STAGE3_IN` (400). This probe takes a window of that ordering that does
contain the gold, runs the real claims screen over it, and reports where the
gold lands afterwards — the number that decides whether raising the window is
worth the BigQuery and flash-lite cost.

    python3 evals/claims_screen_probe.py --key US20150126796A1 --tag h1o --window 2000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from funnel import RUN_DIR, _canon, gold_family_map  # noqa: E402


def _order(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda x: (-len(x.get("elements") or []), -float(x.get("cos") or 0)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--tag", default="h1o")
    ap.add_argument("--window", type=int, default=2000, help="how far down the stage-2 order to read claims")
    ap.add_argument("--keep", type=int, default=60)
    a = ap.parse_args()
    from common import load_env_yaml
    load_env_yaml()
    from patent_analyzer.agentic.prune import stage3_claims

    rec = json.loads((RUN_DIR / f"{a.key}_search_{a.tag}.json").read_text())
    gold = json.loads((RUN_DIR / "gold.json").read_text())[a.key]
    fam_of = await gold_family_map(gold)
    worth = _order([d for d in rec["funnel_docs"] if d.get("worth_reading")])
    window = worth[:a.window]
    in_window = [d["pub_num"] for d in window if _canon(d.get("pub_num") or "") in fam_of]
    print(f"{a.key}: {len(worth)} worth reading, window {len(window)}, gold in window: {in_window}")
    for g in in_window:
        print(f"  {g}: stage-2 order #{next(i + 1 for i, d in enumerate(worth) if d['pub_num'] == g)}")

    docs = [{"pub_num": d.get("pub_num"), "title": d.get("title"), "sources": d.get("sources"),
             "prune_cos": d.get("cos"), "prune_elements": d.get("elements")} for d in window]
    elements = rec.get("loop_elements") or []
    kept, stats = await stage3_claims(elements, docs, list(range(len(docs))), keep=a.keep, n_in=len(docs))
    print("claims screen:", stats)
    kept_pubs = [docs[i]["pub_num"] for i in kept]
    for g in in_window:
        i = next((j for j, d in enumerate(docs) if d["pub_num"] == g), None)
        d = docs[i] if i is not None else {}
        rank = kept_pubs.index(g) + 1 if g in kept_pubs else None
        print(f"  {g}: claims verdict={d.get('claims_worth_reading')} elements={d.get('claims_elements')} "
              f"reason={str(d.get('claims_reason'))[:70]} → rank after claims screen: {rank or f'outside top {a.keep}'}")
    out = RUN_DIR / f"{a.key}_claimsprobe_{a.tag}.json"
    out.write_text(json.dumps({"key": a.key, "tag": a.tag, "window": len(window), "stats": stats,
                               "gold_in_window": in_window, "kept": kept_pubs[:a.keep]}, indent=1))
    print("→", out)


if __name__ == "__main__":
    asyncio.run(main())
