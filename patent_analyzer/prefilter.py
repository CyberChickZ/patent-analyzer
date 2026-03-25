#!/usr/bin/env python3
"""
Deterministic pre-filtering: rank search results by keyword overlap with checklist.
No LLM needed — pure text matching.

Usage:
    python3 prefilter.py \
        --search-results phase3_search.json \
        --checklist-file phase2.json \
        --output top200.json \
        --limit 200
"""

import argparse
import json
import re
import sys
from collections import Counter


def tokenize(text: str) -> set[str]:
    """Extract lowercase tokens, removing stop words."""
    stops = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "each", "all", "any", "few",
        "more", "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "when", "where",
        "how", "what", "which", "who", "whom", "this", "that", "these",
        "those", "it", "its", "we", "our", "they", "their", "them", "he",
        "she", "his", "her", "i", "me", "my", "you", "your", "using", "based",
        "method", "system", "apparatus", "device", "comprising", "includes",
        "including", "wherein", "according", "provided", "present", "invention",
    }
    words = set(re.findall(r'[a-z][a-z0-9]+', text.lower()))
    return words - stops


def bigrams(tokens: set[str]) -> set[str]:
    """Generate bigrams from sorted token list for phrase matching."""
    sorted_toks = sorted(tokens)
    bgs = set()
    for i in range(len(sorted_toks) - 1):
        bgs.add(f"{sorted_toks[i]}_{sorted_toks[i+1]}")
    return bgs


def score_document(doc: dict, checklist_tokens: set[str], checklist_bigrams: set[str], key_phrases: set[str]) -> float:
    """Score a document by keyword overlap with checklist. Returns 0-1."""
    title = doc.get("title", "")
    snippet = doc.get("snippet", "")
    text = f"{title} {snippet}"

    doc_tokens = tokenize(text)
    doc_bigrams = bigrams(doc_tokens)

    # Unigram overlap
    if not checklist_tokens:
        return 0.0
    uni_overlap = len(doc_tokens & checklist_tokens) / len(checklist_tokens)

    # Bigram overlap
    bi_overlap = 0.0
    if checklist_bigrams:
        bi_overlap = len(doc_bigrams & checklist_bigrams) / len(checklist_bigrams)

    # Key phrase exact match bonus
    phrase_bonus = 0.0
    text_lower = text.lower()
    for phrase in key_phrases:
        if phrase in text_lower:
            phrase_bonus += 0.05

    # Weighted combination
    score = 0.5 * uni_overlap + 0.3 * bi_overlap + min(0.2, phrase_bonus)
    return min(1.0, score)


def main():
    parser = argparse.ArgumentParser(description="Pre-filter search results by keyword relevance")
    parser.add_argument("--search-results", required=True)
    parser.add_argument("--checklist-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    with open(args.search_results) as f:
        search = json.load(f)

    with open(args.checklist_file) as f:
        phase2 = json.load(f)

    checklist = phase2.get("checklist", [])
    ucd = phase2.get("ucd", "")

    # Build keyword sets from checklist + UCD
    all_text = " ".join(checklist) + " " + ucd
    cl_tokens = tokenize(all_text)
    cl_bigrams = bigrams(cl_tokens)

    # Key technical phrases to look for
    key_phrases = {
        "dense descriptor", "dense correspondence", "contrastive loss",
        "self-supervised", "pixel correspondence", "3d reconstruction",
        "object mask", "change detection", "domain randomization",
        "hard negative", "cross-object", "multi-object", "descriptor space",
        "robotic grasping", "robotic manipulation", "visual descriptor",
        "class generalization", "instance specific", "nearest neighbor",
        "descriptor matching", "contrastive learning", "feature matching",
        "object representation", "visual representation", "descriptor learning",
        "tsdf", "volumetric fusion", "rgbd", "depth image",
    }

    # Score all documents
    all_docs = search.get("all_patents", []) + search.get("all_papers", [])
    print(f"Total documents: {len(all_docs)}")

    scored = []
    for doc in all_docs:
        s = score_document(doc, cl_tokens, cl_bigrams, key_phrases)
        doc["relevance_score"] = round(s, 4)
        scored.append(doc)

    # Sort by relevance score descending
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Take top N
    top = scored[:args.limit]

    # Stats
    types = Counter(d["match_type"] for d in top)
    avg_score = sum(d["relevance_score"] for d in top) / len(top) if top else 0

    result = {
        "total_candidates": len(all_docs),
        "selected": len(top),
        "avg_relevance": round(avg_score, 4),
        "type_breakdown": dict(types),
        "documents": top,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Selected top {len(top)}: {types.get('Patent', 0)} patents, {types.get('Paper', 0)} papers")
    print(f"Avg relevance: {avg_score:.4f}")
    print(f"Score range: {top[0]['relevance_score']:.4f} - {top[-1]['relevance_score']:.4f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
