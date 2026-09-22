#!/usr/bin/env python3
"""N7b: how far the re-ordered open-access chain actually gets.

Same two acceptance sets as N7 (`scratch_n7_fulltext_reach.py`), so the numbers
are comparable line for line:

  gold  the 17 examiner-cited NPL that resolved to a paper id
  job   the 104 papers the eight e4/h1h runs delivered

What changed underneath: the chain now asks repositories and the APIs meant to
be called *before* the publisher's own copy, the HTTP client speaks Chrome's TLS
fingerprint, and acquisition is a walk down a plan rather than one URL.

The two numbers stay separate and both are printed:

  fulltext_tier      how far *resolution* got - a URL was found
  fulltext_download  what actually came back - %PDF bytes, or API full text

Per-source success is reported against the number of documents where that source
was *tried*, not against the number where it produced a link.

    python3 evals/scratch_n7b_reach.py --set gold
    python3 evals/scratch_n7b_reach.py --set both --limit 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from patent_analyzer import fulltext as ft  # noqa: E402
from scratch_n7_fulltext_reach import gold_docs, job_docs  # noqa: E402

GOLD_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature"
OUT = GOLD_DIR / "n7b_reach.json"


async def run(docs: list[dict], max_steps: int) -> list[dict]:
    rows = []
    for i, doc in enumerate(docs, 1):
        p = await ft.resolve(doc)
        got = await ft.acquire(doc, p, max_steps=max_steps)
        rows.append({
            "title": (doc.get("title") or "")[:70], "doi": p["doi"],
            "fulltext_tier": p["fulltext_tier"], "resolved_url": p["fulltext_url"],
            "resolve_source": p.get("fulltext_source", ""),
            "plan": [s["source"] for s in p.get("fulltext_plan") or []],
            "fulltext_download": got["fulltext_download"],
            "download_source": got["fulltext_source"],
            "bytes": len(got["pdf"] or b""), "chars": len(got["text"]),
            "attempts": got["attempts"], "detail": got["fulltext_detail"][:200],
        })
        print(f"  [{i}/{len(docs)}] {rows[-1]['fulltext_download']:<9} "
              f"{rows[-1]['download_source'] or '-':<18} {rows[-1]['doi'][:48]}", flush=True)
    return rows


def report(name: str, rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n=== {name}: {n} documents ===")

    print("\nfulltext_tier — how far RESOLUTION got (a URL was found):")
    tiers = Counter(r["fulltext_tier"] for r in rows)
    for t in ft.TIERS:
        print(f"  {t:<16}{tiers.get(t, 0):>4}{(100 * tiers.get(t, 0) / n if n else 0):>7.0f}%")

    print("\nfulltext_download — what actually CAME BACK:")
    dls = Counter(r["fulltext_download"] for r in rows)
    for k in ("ok", "ok_text", "failed", "no_url"):
        print(f"  {k:<16}{dls.get(k, 0):>4}{(100 * dls.get(k, 0) / n if n else 0):>7.0f}%")
    read = dls.get("ok", 0) + dls.get("ok_text", 0)
    print(f"  {'READ (either)':<16}{read:>4}{(100 * read / n if n else 0):>7.0f}%")

    # A tier with no credential was never asked, so it must not appear in the
    # denominator — reporting "Wiley TDM: 0/15" would read as "Wiley refused"
    # when the truth is that nobody applied for the token yet.
    tried: Counter = Counter()
    won: Counter = Counter()
    skipped: Counter = Counter()
    for r in rows:
        for a in r["attempts"]:
            if str(a.get("detail", "")).startswith("skipped:"):
                skipped[a["source"]] += 1
                continue
            tried[a["source"]] += 1
            if a["ok"]:
                won[a["source"]] += 1
    print("\nper-source, denominator = documents where the source was actually TRIED:")
    print(f"  {'source':<18}{'tried':>7}{'got it':>8}{'rate':>8}")
    for s in sorted(tried, key=lambda s: -won[s]):
        print(f"  {s:<18}{tried[s]:>7}{won[s]:>8}{(100 * won[s] / tried[s]):>7.0f}%")
    for s in sorted(skipped):
        print(f"  {s:<18}{'-':>7}{'-':>8}{'n/a':>8}  skipped on {skipped[s]} docs "
              f"(no credential — never asked, not refused)")

    why: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for a in r["attempts"]:
            if not a["ok"]:
                why[a["source"]][str(a["detail"])[:70]] += 1
    print("\nwhy each source came back empty (top reasons):")
    for s in sorted(why):
        for reason, c in why[s].most_common(2):
            print(f"  {s:<18}{c:>4}  {reason}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="gold", choices=["gold", "job", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=14)
    args = ap.parse_args()

    out = {}
    for name in (["gold", "job"] if args.which == "both" else [args.which]):
        docs = gold_docs() if name == "gold" else job_docs()
        if args.limit:
            docs = docs[:args.limit]
        print(f"\n--- {name}: {len(docs)} documents ---", flush=True)
        rows = asyncio.run(run(docs, args.max_steps))
        report(name, rows)
        out[name] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev.update(out)
    OUT.write_text(json.dumps(prev, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
