#!/usr/bin/env python3
"""N7: how far does open-access full text actually get us.

Two sets, reported separately because they fail for different reasons:

  gold   the examiner-cited NPL that gold_nature.py --stage npl could resolve
         to a paper id (17 distinct DOIs over the h1h run). This is the set we
         are blind on: paper-side reach was .118 against .607 on patents.
  job    the papers a real job delivered (`ranked`, match_type != Patent, over
         the eight e4/h1h runs). This is what the pipeline would hand Phase 4.

Per document it runs patent_analyzer.fulltext.resolve and then *actually
fetches* the URL, because a resolved URL is not a PDF: Unpaywall's best
location is often a repository landing page. Success means bytes that start
with %PDF, not a 200.

    python3 evals/scratch_n7_fulltext_reach.py --set gold
    python3 evals/scratch_n7_fulltext_reach.py --set job --limit 40
    python3 evals/scratch_n7_fulltext_reach.py --set gold --no-fetch

Nothing here touches a paywall. See patent_analyzer/fulltext.py for why there
is no proxied tier.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import certifi

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from patent_analyzer import fulltext as ft  # noqa: E402

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "e4"
GOLD_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature"
OUT = GOLD_DIR / "n7_fulltext_reach.json"
TAG = "h1h"

_SSL = ssl.create_default_context(cafile=certifi.where())
_UA = "patent-analyzer/0.4 (https://github.com/CyberChickZ/patent-analyzer)"
_HOST_COOLDOWN_S = 1.0
_last_hit: dict[str, float] = {}


def gold_docs() -> list[dict]:
    """The resolved examiner-cited NPL, deduplicated by DOI."""
    per_key = json.loads((GOLD_DIR / f"npl_{TAG}.json").read_text())
    seen, docs = set(), []
    for rows in per_key.values():
        for r in rows:
            res = r.get("resolved") or {}
            doi = (res.get("doi") or "").lower()
            if not doi or doi in seen:
                continue
            seen.add(doi)
            docs.append({"pub_num": doi, "title": res.get("title") or r.get("title") or "",
                         "year": res.get("year", "")})
    return docs


def job_docs() -> list[dict]:
    """Papers a real job delivered, over every h1h run, deduplicated."""
    seen, docs = set(), []
    for p in sorted(RUN_DIR.glob(f"*_search_{TAG}.json")):
        rec = json.loads(p.read_text())
        for d in rec.get("ranked") or []:
            if d.get("match_type") == "Patent":
                continue
            k = (d.get("pub_num") or "").lower() or (d.get("title") or "").lower()
            if not k or k in seen:
                continue
            seen.add(k)
            docs.append(d)
    return docs


def fetch(url: str, timeout: float = 30.0) -> tuple[bool, str, int]:
    """(is_pdf, detail, bytes). One polite request per host per second."""
    host = urllib.parse.urlparse(url).netloc
    gap = _HOST_COOLDOWN_S - (time.time() - _last_hit.get(host, 0))
    if gap > 0:
        time.sleep(gap)
    _last_hit[host] = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
            head = r.read(8)
            ctype = (r.headers.get("Content-Type") or "").split(";")[0]
            if head[:4] != b"%PDF":
                return False, f"not a PDF ({ctype or 'no content-type'})", 0
            rest = r.read()
            return True, ctype or "application/pdf", len(head) + len(rest)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}", 0
    except Exception as exc:
        return False, f"{type(exc).__name__}", 0


async def run(docs: list[dict], do_fetch: bool) -> list[dict]:
    rows = []
    patches = await ft.resolve_many(docs)
    for doc, p in zip(docs, patches):
        row = {"title": (doc.get("title") or "")[:80], "doi": p["doi"],
               "tier": p["fulltext_tier"], "url": p["fulltext_url"],
               "landing_page": p["landing_page"], "detail": p["fulltext_detail"],
               "pdf_ok": None, "fetch_detail": "", "bytes": 0}
        if do_fetch and p["fulltext_url"]:
            ok, detail, n = await asyncio.to_thread(fetch, p["fulltext_url"])
            row["pdf_ok"], row["fetch_detail"], row["bytes"] = ok, detail, n
        # Europe PMC is the last open-access stop for whatever the fetch missed.
        row["epmc_chars"] = 0
        if do_fetch and not row["pdf_ok"] and p["doi"]:
            text, err = await ft.europepmc_text(p["doi"])
            row["epmc_chars"] = len(text)
            if not text:
                row["epmc_detail"] = err or ""
        rows.append(row)
    return rows


def report(name: str, rows: list[dict], do_fetch: bool) -> None:
    total = len(rows)
    tiers = Counter(r["tier"] for r in rows)
    print(f"\n=== {name}: {total} documents ===")
    print(f"{'tier':<16}{'n':>5}{'share':>8}" + (f"{'PDF fetched':>14}" if do_fetch else ""))
    for t in ft.TIERS:
        n = tiers.get(t, 0)
        line = f"{t:<16}{n:>5}{(100 * n / total if total else 0):>7.0f}%"
        if do_fetch:
            ok = sum(1 for r in rows if r["tier"] == t and r["pdf_ok"])
            line += f"{(f'{ok}/{n}' if n else '-'):>14}"
        print(line)
    if do_fetch:
        ok = sum(1 for r in rows if r["pdf_ok"])
        epmc = sum(1 for r in rows if r.get("epmc_chars"))
        read = sum(1 for r in rows if r["pdf_ok"] or r.get("epmc_chars"))
        print(f"{'TOTAL PDF':<16}{ok:>5}{(100 * ok / total if total else 0):>7.0f}%")
        print(f"{'+ Europe PMC':<16}{epmc:>5}{(100 * epmc / total if total else 0):>7.0f}%  "
              f"(full text through the Europe PMC API, no PDF)")
        print(f"{'READ (either)':<16}{read:>5}{(100 * read / total if total else 0):>7.0f}%")

    # The manual-download list is only useful if it has a page to open.
    miss = [r for r in rows if not r["pdf_ok"] and not r.get("epmc_chars")] if do_fetch \
        else [r for r in rows if r["tier"] == "abstract_only"]
    with_lp = sum(1 for r in miss if r["landing_page"])
    if miss:
        print(f"\nstill unread: {with_lp}/{len(miss)} have a landing page for the manual list "
              f"({100 * with_lp / len(miss):.0f}%)")
        why = Counter(r["detail"] for r in miss)
        for k, v in why.most_common():
            print(f"  {v:>3}  {k}")
    if do_fetch:
        bad = [r for r in rows if r["url"] and not r["pdf_ok"]]
        if bad:
            print(f"\nresolved but not a PDF ({len(bad)}):")
            for r in bad[:15]:
                print(f"  [{r['tier']}] {r['fetch_detail']:<28} {r['url'][:78]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="which", default="gold", choices=["gold", "job", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    do_fetch = not args.no_fetch

    out = {}
    for name in (["gold", "job"] if args.which == "both" else [args.which]):
        docs = gold_docs() if name == "gold" else job_docs()
        if args.limit:
            docs = docs[:args.limit]
        rows = asyncio.run(run(docs, do_fetch))
        report(name, rows, do_fetch)
        out[name] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev.update(out)
    OUT.write_text(json.dumps(prev, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
