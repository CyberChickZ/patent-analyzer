#!/usr/bin/env python3
"""
Semantic search using paper embeddings.

Two modes:
1. LOCAL EMBEDDING: Use sentence-transformers to embed target paper + all candidates,
   rank by cosine similarity. Replaces keyword-based prefilter.py with semantic ranking.

2. S2 RECOMMENDATIONS: Use Semantic Scholar's SPECTER-based recommendation API
   to find similar papers given seed paper IDs from initial search results.

Usage:
    # Mode 1: Semantic re-ranking of existing search results
    python3 semantic_search.py rerank \
        --target-text "paper abstract or summary" \
        --search-results phase3_search.json \
        --output semantic_top.json \
        --limit 30

    # Mode 2: Find new papers via S2 recommendations from top seeds
    python3 semantic_search.py recommend \
        --seed-titles "3DMatch,Universal Correspondence Network" \
        --output s2_recommendations.json \
        --limit 50

    # Mode 3: Combined — rerank + augment with recommendations
    python3 semantic_search.py full \
        --phase1 phase1.json \
        --search-results phase3_search.json \
        --output semantic_results.json \
        --limit 30
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
import numpy as np
from pathlib import Path
from typing import Any

SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# ─── LOCAL EMBEDDING ────────────────────────────────────────────────

_model = None

def get_model():
    """Load sentence-transformers model (lazy, cached)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: fast, good quality, 384-dim
        print("Loading embedding model (first time may download ~80MB)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into vectors."""
    model = get_model()
    return model.encode(texts, show_progress_bar=len(texts) > 50, normalize_embeddings=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b. Returns 1D array."""
    # Since embeddings are normalized, dot product = cosine similarity
    return np.dot(b, a)


def rerank_by_embedding(target_text: str, documents: list[dict], limit: int = 30) -> list[dict]:
    """
    Rerank documents by semantic similarity to target text.
    Each document should have 'title' and optionally 'snippet'/'abstract'.
    Returns top `limit` documents sorted by similarity, with 'semantic_score' added.
    """
    # Build document texts
    doc_texts = []
    for doc in documents:
        parts = [doc.get("title", "")]
        abstract = doc.get("abstract", "")
        snippet = doc.get("snippet", "")
        if abstract:
            parts.append(abstract)
        elif snippet:
            parts.append(snippet)
        doc_texts.append(" ".join(parts))

    if not doc_texts:
        return []

    print(f"Embedding {len(doc_texts)} documents...")
    # Embed target
    target_vec = embed_texts([target_text])[0]
    # Embed all documents
    doc_vecs = embed_texts(doc_texts)

    # Compute similarities
    similarities = cosine_similarity(target_vec, doc_vecs)

    # Sort by similarity
    ranked_indices = np.argsort(similarities)[::-1][:limit]

    results = []
    for idx in ranked_indices:
        doc = documents[idx].copy()
        doc["semantic_score"] = float(similarities[idx])
        results.append(doc)

    return results


# ─── S2 RECOMMENDATIONS ────────────────────────────────────────────

def s2_search_paper(title: str) -> str | None:
    """Find a paper on Semantic Scholar by title, return paperId."""
    q = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=1&fields=paperId,title"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "patent-analyzer/0.1")
        resp = urllib.request.urlopen(req, timeout=15, context=SSL_CTX)
        data = json.loads(resp.read())
        papers = data.get("data", [])
        if papers:
            return papers[0].get("paperId")
    except Exception as e:
        print(f"  [S2 ERROR] {e}", file=sys.stderr)
    return None


def s2_recommend(paper_id: str, limit: int = 20) -> list[dict]:
    """Get recommendations for a paper from S2's SPECTER embeddings."""
    url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}?fields=title,abstract,year,externalIds&limit={limit}"
    try:
        resp = urllib.request.urlopen(url, timeout=15, context=SSL_CTX)
        data = json.loads(resp.read())
        return data.get("recommendedPapers", [])
    except Exception as e:
        print(f"  [S2 REC ERROR] {e}", file=sys.stderr)
    return []


def find_recommendations(seed_titles: list[str], limit: int = 50) -> list[dict]:
    """Find recommended papers from a list of seed paper titles."""
    all_recs = []
    seen = set()

    for title in seed_titles:
        print(f"  Finding seed: {title[:60]}...")
        pid = s2_search_paper(title)
        if not pid:
            print(f"    -> not found on S2")
            continue
        time.sleep(1.5)  # Rate limit

        print(f"    -> paperId={pid[:15]}..., getting recommendations...")
        recs = s2_recommend(pid, limit=20)
        time.sleep(1.5)

        for r in recs:
            rid = r.get("paperId", "")
            if rid and rid not in seen:
                seen.add(rid)
                all_recs.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("abstract", ""),
                    "abstract": r.get("abstract", ""),
                    "year": r.get("year"),
                    "match_type": "Paper",
                    "pub_num": rid,
                    "source": f"s2_rec_from_{title[:30]}",
                })

        print(f"    -> {len(recs)} recommendations")

    print(f"Total unique recommendations: {len(all_recs)}")
    return all_recs[:limit]


# ─── COMMANDS ───────────────────────────────────────────────────────

def cmd_rerank(args):
    """Rerank existing search results by semantic similarity."""
    search = load_json(args.search_results)
    all_docs = search.get("all_patents", []) + search.get("all_papers", [])

    target = args.target_text
    if not target and args.phase1:
        phase1 = load_json(args.phase1)
        target = phase1.get("summary", "")

    if not target:
        print("Error: provide --target-text or --phase1")
        sys.exit(1)

    ranked = rerank_by_embedding(target, all_docs, limit=args.limit)

    result = {
        "method": "semantic_embedding",
        "model": "all-MiniLM-L6-v2",
        "total_candidates": len(all_docs),
        "selected": len(ranked),
        "documents": ranked,
    }
    save_json(result, args.output)

    print(f"\nTop {min(10, len(ranked))} by semantic similarity:")
    for i, doc in enumerate(ranked[:10]):
        ss = doc["semantic_score"]
        print(f"  {i+1}. [{ss:.3f}] {doc.get('title','')[:60]}")


def cmd_recommend(args):
    """Find new papers via S2 SPECTER recommendations."""
    seeds = [s.strip() for s in args.seed_titles.split(",") if s.strip()]
    if not seeds:
        print("Error: provide --seed-titles (comma-separated)")
        sys.exit(1)

    recs = find_recommendations(seeds, limit=args.limit)
    save_json({"method": "s2_recommendations", "papers": recs}, args.output)


def cmd_full(args):
    """Full semantic pipeline: rerank existing + augment with S2 recommendations."""
    search = load_json(args.search_results)
    phase1 = load_json(args.phase1)
    target = phase1.get("summary", "")

    all_docs = search.get("all_patents", []) + search.get("all_papers", [])
    print(f"=== Step 1: Semantic reranking of {len(all_docs)} documents ===")
    ranked = rerank_by_embedding(target, all_docs, limit=args.limit)

    # Use top 3 papers as seeds for S2 recommendations
    top_seeds = [d["title"] for d in ranked[:3] if d.get("title")]
    new_papers = []
    if top_seeds:
        print(f"\n=== Step 2: S2 recommendations from top {len(top_seeds)} seeds ===")
        try:
            new_papers = find_recommendations(top_seeds, limit=30)
        except Exception as e:
            print(f"  S2 recommendations failed: {e}")

    # Merge: add new papers not already in ranked
    existing_titles = {d.get("title", "").lower().strip() for d in ranked}
    added = 0
    for p in new_papers:
        t = p.get("title", "").lower().strip()
        if t and t not in existing_titles:
            existing_titles.add(t)
            ranked.append(p)
            added += 1

    # Re-embed and re-rank the combined set
    if added > 0:
        print(f"\n=== Step 3: Re-ranking combined set ({len(ranked)} docs) ===")
        ranked = rerank_by_embedding(target, ranked, limit=args.limit)

    result = {
        "method": "semantic_full",
        "model": "all-MiniLM-L6-v2",
        "total_candidates": len(all_docs) + len(new_papers),
        "new_from_s2": added,
        "selected": len(ranked),
        "documents": ranked,
    }
    save_json(result, args.output)

    print(f"\nFinal: {len(ranked)} documents ({added} new from S2)")
    print(f"\nTop 10 by semantic similarity:")
    for i, doc in enumerate(ranked[:10]):
        ss = doc.get("semantic_score", 0)
        src = "(S2)" if doc.get("source", "").startswith("s2_") else ""
        print(f"  {i+1}. [{ss:.3f}] {doc.get('title','')[:60]} {src}")


# ─── UTIL ───────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def main():
    parser = argparse.ArgumentParser(description="Semantic paper search via embeddings")
    sub = parser.add_subparsers(dest="command")

    p_rerank = sub.add_parser("rerank", help="Rerank search results by semantic similarity")
    p_rerank.add_argument("--target-text", default=None, help="Target paper text")
    p_rerank.add_argument("--phase1", default=None, help="phase1.json (uses summary as target)")
    p_rerank.add_argument("--search-results", required=True)
    p_rerank.add_argument("--output", required=True)
    p_rerank.add_argument("--limit", type=int, default=30)

    p_rec = sub.add_parser("recommend", help="Find papers via S2 recommendations")
    p_rec.add_argument("--seed-titles", required=True, help="Comma-separated seed paper titles")
    p_rec.add_argument("--output", required=True)
    p_rec.add_argument("--limit", type=int, default=50)

    p_full = sub.add_parser("full", help="Full: rerank + S2 recommendations")
    p_full.add_argument("--phase1", required=True)
    p_full.add_argument("--search-results", required=True)
    p_full.add_argument("--output", required=True)
    p_full.add_argument("--limit", type=int, default=30)

    args = parser.parse_args()
    if args.command == "rerank":
        cmd_rerank(args)
    elif args.command == "recommend":
        cmd_recommend(args)
    elif args.command == "full":
        cmd_full(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
