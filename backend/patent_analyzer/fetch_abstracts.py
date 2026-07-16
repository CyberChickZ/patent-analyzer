#!/usr/bin/env python3
"""
Fetch real abstracts from OpenAlex API (free, no key required, generous rate limit).
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

API_URL = "https://api.openalex.org/works"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
MAILTO = "patent-analyzer@example.com"  # polite pool gets higher rate limit


def reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    words = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[k] for k in sorted(words.keys()))


def search_paper(title: str) -> dict | None:
    """Search OpenAlex by title, return first match with abstract."""
    params = urllib.parse.urlencode({
        "filter": f"title.search:{title}",
        "per_page": "1",
        "mailto": MAILTO,
    })
    url = f"{API_URL}?{params}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "patent-analyzer/0.1")
        with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
        results = data.get("results", [])
        if results:
            w = results[0]
            abstract = reconstruct_abstract(w.get("abstract_inverted_index", {}))
            if abstract:
                authors = [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])[:3]]
                author_str = ", ".join(a for a in authors if a)
                if len(w.get("authorships", [])) > 3:
                    author_str += f" et al."
                return {
                    "abstract": abstract,
                    "year": w.get("publication_year"),
                    "authors": author_str,
                    "doi": w.get("doi", ""),
                    "oa_url": w.get("id", ""),
                }
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

    # Prioritize docs with overlap, then by score
    candidates = sorted(scoring, key=lambda m: m.get("similarity_score", 0) or 0, reverse=True)[:args.limit]

    print(f"Fetching abstracts for {len(candidates)} papers via OpenAlex...")
    fetched = 0
    skipped = 0

    for i, doc in enumerate(candidates):
        title = doc.get("title", "")
        if not title:
            continue
        if doc.get("abstract"):
            skipped += 1
            continue

        print(f"  [{i+1}/{len(candidates)}] {title[:60]}...", end=" ", flush=True)
        result = search_paper(title)

        if result and result.get("abstract"):
            doc["abstract"] = result["abstract"]
            if result.get("authors"):
                doc["authors"] = result["authors"]
            if result.get("year"):
                doc["year"] = result["year"]
            if result.get("doi"):
                doc["doi"] = result["doi"]
            fetched += 1
            print(f"OK ({len(result['abstract'])} chars)")
        else:
            print("not found")

        # Polite pool: 10 requests/second is fine with mailto
        time.sleep(0.15)

    # Save back
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {fetched} fetched, {skipped} already had abstract, {len(candidates)-fetched-skipped} not found")


if __name__ == "__main__":
    main()
