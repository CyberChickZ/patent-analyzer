#!/usr/bin/env python3
"""
Fetch real abstracts from Semantic Scholar API (free, no key required).
Merges abstracts back into results.json.

Usage:
    python3 fetch_abstracts.py --results results.json [--limit 50]
"""

import argparse
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import ssl
import certifi
import sys

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def search_paper(title: str) -> dict | None:
    """Search Semantic Scholar by title, return first match with abstract."""
    params = urllib.parse.urlencode({
        "query": title,
        "limit": "1",
        "fields": "title,abstract,url,year,authors",
    })
    url = f"{API_URL}?{params}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "patent-analyzer/0.1")
        with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
        papers = data.get("data", [])
        if papers and papers[0].get("abstract"):
            return papers[0]
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--limit", type=int, default=50, help="Max papers to fetch")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    scoring = results.get("evaluation", {}).get("scoring_report", [])

    # Only fetch for docs with overlap, prioritized by score
    candidates = [m for m in scoring if (m.get("similarity_score", 0) or 0) > 0]
    if not candidates:
        candidates = scoring[:args.limit]
    else:
        candidates = candidates[:args.limit]

    print(f"Fetching abstracts for {len(candidates)} papers...")
    fetched = 0

    for i, doc in enumerate(candidates):
        title = doc.get("title", "")
        if not title:
            continue
        if doc.get("abstract"):  # already has one
            continue

        print(f"  [{i+1}/{len(candidates)}] {title[:60]}...")
        result = search_paper(title)

        if result and result.get("abstract"):
            doc["abstract"] = result["abstract"]
            doc["s2_url"] = result.get("url", "")
            doc["year"] = result.get("year", "")
            authors = result.get("authors", [])
            if authors:
                doc["authors"] = ", ".join(a.get("name", "") for a in authors[:3])
                if len(authors) > 3:
                    doc["authors"] += f" et al. ({len(authors)} authors)"
            fetched += 1
            print(f"    -> Got abstract ({len(result['abstract'])} chars)")
        else:
            print(f"    -> No abstract found")

        # Rate limit: 100 requests per 5 minutes = 1 per 3 seconds
        time.sleep(3)

    # Save back
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {fetched}/{len(candidates)} abstracts fetched")


if __name__ == "__main__":
    main()
