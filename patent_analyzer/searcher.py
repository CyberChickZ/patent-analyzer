#!/usr/bin/env python3
"""
SerpAPI search helper for patent-analyze skill.
Features:
- Incremental log file (--log-file)
- Incremental result saving after each group
- Network retry with exponential backoff
- Resume from partial results if output file exists
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import ssl
import certifi
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SERPAPI_URL = "https://serpapi.com/search.json"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
MAX_PAGES_PATENT = 1
MAX_PAGES_PAPER = 5
TIMEOUT_S = 60
MAX_RETRIES = 4
RETRY_BACKOFF = [10, 30, 60, 90]  # was [2,5,10] — too short to recover from SerpAPI 429


def log(msg: str, log_file: str | None = None):
    """Print and optionally append to log file."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def serpapi_search(
    engine: str,
    query: str,
    api_key: str,
    log_file: str | None = None,
    max_pages: int = 1,
    num_per_page: int = 100,
) -> tuple[list[dict[str, Any]], str | None]:
    """Perform a paginated SerpAPI search with retry.

    Returns (results, error) where error is None on success (even if 0 results)
    or a short human-readable string when something went wrong (HTTP error,
    network failure, SerpAPI error message, etc.).
    """
    all_results: list[dict[str, Any]] = []
    error: str | None = None

    if not api_key:
        return all_results, "SERPAPI_KEY not set"

    for page in range(1, max_pages + 1):
        # Google Patents requires num >= 10
        actual_num = max(10, num_per_page)
        params = {
            "engine": engine,
            "q": query,
            "api_key": api_key,
            "num": str(actual_num),
            "page": str(page),
        }
        url = f"{SERPAPI_URL}?{urllib.parse.urlencode(params)}"

        data = None
        last_exc: str | None = None
        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=SSL_CTX) as resp:
                    data = json.loads(resp.read().decode())
                break  # success
            except urllib.error.HTTPError as e:
                last_exc = f"HTTP {e.code}"
                log(f"  [HTTP {e.code}] attempt {attempt+1}/{MAX_RETRIES}: {query[:60]}...", log_file)
                if e.code == 401:
                    log("  [FATAL] Invalid API key", log_file)
                    return all_results, "HTTP 401 (invalid SerpAPI key or quota exhausted)"
                if e.code == 429:
                    return all_results, "HTTP 429 (rate limited by SerpAPI)"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
            except Exception as e:
                last_exc = f"{type(e).__name__}: {e}"
                log(f"  [NET ERROR] attempt {attempt+1}/{MAX_RETRIES}: {last_exc}", log_file)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])

        if data is None:
            log(f"  [SKIP] All retries failed for: {query[:60]}...", log_file)
            return all_results, f"network failure after {MAX_RETRIES} retries: {last_exc or 'unknown'}"

        if "error" in data:
            err_msg = str(data.get("error", "unknown SerpAPI error"))
            log(f"  [SERPAPI] {err_msg}", log_file)
            return all_results, f"SerpAPI error: {err_msg[:200]}"

        organic = data.get("organic_results", [])
        if not organic:
            log(f"  [EMPTY] No organic results for: {query[:60]}...", log_file)
            # Genuine 0 results — not an error, just no matches
            break

        total = data.get("search_information", {}).get("total_results", 0)
        log(f"  [OK] page {page}: {len(organic)} results (total: {total})", log_file)

        for item in organic:
            result: dict[str, Any] = {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
            }

            if engine == "google_patents":
                result["pub_num"] = item.get("publication_number", "")
                result["pdf_link"] = item.get("pdf", "")
                result["match_type"] = "Patent"
                result["url"] = item.get("patent_link", "") or item.get("link", "")
                result["patent_link"] = item.get("patent_link", "")
                result["filing_date"] = item.get("filing_date", "")
                result["grant_date"] = item.get("grant_date", "")
                result["priority_date"] = item.get("priority_date", "")
                result["publication_date"] = item.get("publication_date", "")
                result["inventor"] = item.get("inventor", "")
                result["assignee"] = item.get("assignee", "")
                result["country_status"] = item.get("country_status", {})
            else:
                result["pub_num"] = item.get("result_id", "")
                result["match_type"] = "Paper"
                result["url"] = item.get("link", "")
                result["authors"] = ", ".join(a.get("name", "") for a in item.get("publication_info", {}).get("authors", []))
                result["year"] = item.get("publication_info", {}).get("summary", "").split(",")[-1].strip() if item.get("publication_info") else ""
                pdf_links = []
                for resource in item.get("resources", []):
                    if resource.get("file_format", "").upper() == "PDF":
                        link = resource.get("link", "")
                        if link.startswith("http"):
                            pdf_links.append(link)
                result["pdf_link"] = pdf_links

            all_results.append(result)

        if len(organic) < num_per_page:
            break

        time.sleep(0.3)

    return all_results, error


def save_incremental(results: dict, output_path: str):
    """Save current results incrementally."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def download_pdf(url: str, output_dir: Path, filename: str, log_file: str | None = None) -> str | None:
    """Try to download a PDF with retry."""
    if not url or not url.startswith("http"):
        return None

    output_path = output_dir / filename
    if output_path.exists():
        return str(output_path)

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "Mozilla/5.0 (patent-analyze)")
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                content = resp.read(8192)
                if b"%PDF" not in content[:8] and "application/pdf" not in (resp.headers.get("Content-Type") or ""):
                    return None
                rest = resp.read()
                with open(output_path, "wb") as f:
                    f.write(content)
                    f.write(rest)
            return str(output_path)
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
            else:
                log(f"  [DL FAIL] {filename}: {e}", log_file)
    return None


def main():
    parser = argparse.ArgumentParser(description="SerpAPI patent/paper search")
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--serpapi-key", default=None)
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--papers-dir", default=None)
    parser.add_argument("--log-file", default=None, help="Path to incremental log file")
    args = parser.parse_args()

    lf = args.log_file
    api_key = args.serpapi_key or os.environ.get("SERPAPI_KEY", "")
    if not api_key:
        log("[FATAL] No SerpAPI key. Set SERPAPI_KEY env or pass --serpapi-key", lf)
        sys.exit(1)

    log(f"Starting search. Key: ...{api_key[-8:]}", lf)

    with open(args.queries_file) as f:
        queries_data = json.load(f)

    groups = queries_data.get("groups", [])

    # Try to resume from existing output
    results: dict[str, Any] = {"groups": [], "all_patents": [], "all_papers": []}
    completed_groups: set[str] = set()
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                existing = json.load(f)
            results = existing
            completed_groups = {g["group_id"] for g in existing.get("groups", [])}
            log(f"Resuming: {len(completed_groups)} groups already done", lf)
        except Exception:
            pass

    seen_patents: set[str] = {p["pub_num"] for p in results["all_patents"] if p.get("pub_num")}
    seen_papers: set[str] = {p.get("pub_num") or p.get("title", "") for p in results["all_papers"]}

    for group in groups:
        group_id = group.get("group_id", "?")
        if group_id in completed_groups:
            log(f"[{group_id}] Already done, skipping", lf)
            continue

        label = group.get("label", "")
        patent_queries = group.get("patent_queries", [])
        paper_queries = group.get("paper_queries", [])

        log(f"\n{'='*50}", lf)
        log(f"[{group_id}] {label}", lf)
        log(f"{'='*50}", lf)

        group_result: dict[str, Any] = {
            "group_id": group_id,
            "label": label,
            "patent_query": "",
            "patent_matches_found": 0,
            "paper_query": "",
            "paper_matches_found": 0,
        }

        # Patent searches
        for qi, query in enumerate(patent_queries):
            log(f"[{group_id}] Patent Q{qi+1}/{len(patent_queries)}: {query[:80]}...", lf)
            matches, err = serpapi_search("google_patents", query, api_key, lf, MAX_PAGES_PATENT, 100)
            if err:
                log(f"  [ERR] {err}", lf)
            new = 0
            for m in matches:
                if m["pub_num"] and m["pub_num"] not in seen_patents:
                    seen_patents.add(m["pub_num"])
                    results["all_patents"].append(m)
                    new += 1
            if matches:
                group_result["patent_query"] = query
                group_result["patent_matches_found"] += new
            log(f"  → {len(matches)} raw, {new} new unique", lf)
            time.sleep(0.5)

        # Paper searches
        for qi, query in enumerate(paper_queries):
            log(f"[{group_id}] Scholar Q{qi+1}/{len(paper_queries)}: {query[:80]}...", lf)
            matches, err = serpapi_search("google_scholar", query, api_key, lf, MAX_PAGES_PAPER, 20)
            if err:
                log(f"  [ERR] {err}", lf)
            new = 0
            for m in matches:
                key = m.get("pub_num") or m.get("title", "")
                if key and key not in seen_papers:
                    seen_papers.add(key)
                    results["all_papers"].append(m)
                    new += 1
            if matches:
                group_result["paper_query"] = query
                group_result["paper_matches_found"] += new
            log(f"  → {len(matches)} raw, {new} new unique", lf)
            time.sleep(0.5)

        results["groups"].append(group_result)

        # Incremental save after each group
        results["summary"] = {
            "total_patents": len(results["all_patents"]),
            "total_papers": len(results["all_papers"]),
            "total_unique": len(results["all_patents"]) + len(results["all_papers"]),
        }
        save_incremental(results, args.output)
        log(f"[{group_id}] Done. Running total: {results['summary']['total_patents']} patents, {results['summary']['total_papers']} papers. Saved.", lf)

    # Download PDFs
    if args.download_pdfs and args.papers_dir:
        papers_dir = Path(args.papers_dir)
        papers_dir.mkdir(parents=True, exist_ok=True)
        all_docs = results["all_patents"] + results["all_papers"]
        log(f"\nDownloading PDFs for {len(all_docs)} documents...", lf)
        downloaded = 0
        for doc in all_docs:
            if doc.get("local_pdf"):
                continue
            links = doc.get("pdf_link", "")
            if isinstance(links, str) and links:
                links_list = [links]
            elif isinstance(links, list):
                links_list = links
            else:
                continue
            import re as _re
            _tc = _re.sub(r'[^\w\s-]', '', doc.get("title", "")).strip()[:80]
            fname = _re.sub(r'\s+', '_', _tc) or doc.get("pub_num", "x").replace("/", "_")
            for i, link in enumerate(links_list):
                suffix = f"_{i}" if i > 0 else ""
                local = download_pdf(link, papers_dir, f"{fname}{suffix}.pdf", lf)
                if local:
                    doc["local_pdf"] = local
                    downloaded += 1
                    break
        log(f"Downloaded {downloaded} PDFs", lf)
        save_incremental(results, args.output)

    # Final summary
    tp = len(results["all_patents"])
    ts = len(results["all_papers"])
    log(f"\n{'='*60}", lf)
    log(f"SEARCH COMPLETE: {tp} unique patents, {ts} unique papers", lf)
    log(f"{'='*60}", lf)
    log(f"Results saved to: {args.output}", lf)


if __name__ == "__main__":
    main()
