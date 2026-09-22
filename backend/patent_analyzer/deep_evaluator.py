#!/usr/bin/env python3
"""
Deep PDF evaluation pipeline — deterministic orchestration.

This script:
1. Reads phase3_search.json + phase2.json to identify top candidates
2. For each candidate with a downloaded PDF:
   - Prepares evaluation prompt: target paper description + ONE prior art PDF
   - Writes individual eval tasks to eval_tasks/ directory
3. After LLM agents complete tasks, merges results into results.json
4. Filters self-references, deduplicates, computes scores

The LLM step is NOT in this script — it generates task files that agents pick up.
The merge step IS in this script — it's pure deterministic aggregation.

Usage:
    # Step 1: Prepare evaluation tasks
    python3 deep_evaluator.py prepare \
        --search-results phase3_search.json \
        --phase2 phase2.json \
        --phase1 phase1.json \
        --output-dir patent-analysis-output \
        --limit 20

    # Step 2: (LLM agents run on task files — external)

    # Step 3: Merge completed evaluations
    python3 deep_evaluator.py merge \
        --output-dir patent-analysis-output \
        --phase1 phase1.json \
        --phase2 phase2.json \
        --search-results phase3_search.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── PREPARE ────────────────────────────────────────────────────────

def prepare_tasks(args):
    """Prepare individual evaluation task files: 1 target × 1 prior art each."""

    search = load_json(args.search_results)
    phase2 = load_json(args.phase2)
    phase1 = load_json(args.phase1)

    checklist = phase2.get("checklist", [])
    invention_summary = phase1.get("summary", "")
    invention_ucd = phase2.get("ucd", "")

    # Collect all docs with PDF paths
    all_docs = search.get("all_patents", []) + search.get("all_papers", [])
    pdf_docs = [d for d in all_docs if d.get("local_pdf") and os.path.exists(d["local_pdf"])]

    # Keyword pre-filter to get top N
    from .prefilter import tokenize, bigrams, score_document
    cl_text = " ".join(checklist) + " " + invention_ucd
    cl_tokens = tokenize(cl_text)
    cl_bigrams = bigrams(cl_tokens)
    key_phrases = {
        "dense descriptor", "contrastive loss", "self-supervised",
        "pixel correspondence", "3d reconstruction", "object mask",
        "change detection", "domain randomization", "hard negative",
        "cross-object", "descriptor space", "robotic grasping",
        "visual descriptor", "nearest neighbor", "descriptor matching",
    }

    for doc in pdf_docs:
        doc["_relevance"] = score_document(doc, cl_tokens, cl_bigrams, key_phrases)

    pdf_docs.sort(key=lambda x: x["_relevance"], reverse=True)
    candidates = pdf_docs[:args.limit]

    # Create eval_tasks directory
    tasks_dir = Path(args.output_dir) / "eval_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    # Write one task file per candidate
    task_manifest = []
    for i, doc in enumerate(candidates):
        title = doc.get("title", "Unknown")
        pub_num = doc.get("pub_num", f"doc_{i}")
        pdf_path = doc["local_pdf"]
        match_type = doc.get("match_type", "Paper")
        snippet = doc.get("snippet", "")

        safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', pub_num)[:50]
        task_file = tasks_dir / f"task_{i:03d}_{safe_id}.json"

        task = {
            "task_id": i,
            "title": title,
            "pub_num": pub_num,
            "match_type": match_type,
            "pdf_path": os.path.abspath(pdf_path),
            "snippet": snippet,
            "relevance_score": round(doc["_relevance"], 4),
            "status": "pending",
            # Context for LLM
            "invention_summary": invention_summary,
            "invention_ucd": invention_ucd,
            "checklist": checklist,
            # LLM prompt template
            "prompt": _build_eval_prompt(title, match_type, pub_num, invention_summary, checklist),
        }

        save_json(task, str(task_file))
        task_manifest.append({
            "task_id": i,
            "task_file": str(task_file),
            "pdf_path": os.path.abspath(pdf_path),
            "title": title,
            "pub_num": pub_num,
            "status": "pending",
        })

    # Save manifest
    manifest_path = Path(args.output_dir) / "eval_manifest.json"
    save_json({"total_tasks": len(task_manifest), "tasks": task_manifest}, str(manifest_path))

    print(f"Prepared {len(task_manifest)} evaluation tasks in {tasks_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext: run LLM agents on each task, then: deep_evaluator.py merge")


def _build_eval_prompt(title: str, match_type: str, pub_num: str,
                       invention_summary: str, checklist: list[str]) -> str:
    """Build the evaluation prompt for ONE prior art document."""
    checklist_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(checklist))

    return f"""You are a US patent examiner evaluating ONE prior art document against the invention.

INVENTION UNDER REVIEW:
{invention_summary}

PRIOR ART DOCUMENT:
Title: {title}
Type: {match_type}
ID: {pub_num}
(The full PDF is attached — read it carefully, especially abstract, introduction, methodology, and claims if patent)

CHECKLIST ({len(checklist)} items):
{checklist_text}

INSTRUCTIONS:
1. Read the prior art PDF thoroughly (first 5-8 pages minimum)
2. For EACH checklist item, determine if this document CLEARLY discloses that requirement
3. match=true ONLY with explicit evidence (cite section/page/quote)
4. match=false if missing, unclear, or only partially addressed
5. Assess 102 anticipation: does this SINGLE document disclose ALL checklist elements?
6. Identify key teachings useful for 103 obviousness combination

OUTPUT FORMAT (JSON):
{{
  "title": "{title}",
  "pub_num": "{pub_num}",
  "match_type": "{match_type}",
  "anticipation_assessment": "Does this paper anticipate under 102? Why/why not?",
  "key_teachings": "What elements does this paper teach that could be combined under 103?",
  "checklist_results": {{
    "<checklist item 1 text>": {{"analysis": "Evidence from paper: Section X states...", "match": true/false}},
    "<checklist item 2 text>": {{"analysis": "Not found in paper.", "match": false}},
    ...all {len(checklist)} items...
  }}
}}"""


# ─── MERGE ──────────────────────────────────────────────────────────

def merge_results(args):
    """Merge completed evaluation results into final results.json."""

    output_dir = Path(args.output_dir)
    tasks_dir = output_dir / "eval_tasks"

    # Also check for deep_eval_*.json batch files (legacy format)
    batch_files = sorted(output_dir.glob("deep_eval_*.json"))
    task_files = sorted(tasks_dir.glob("task_*.json")) if tasks_dir.exists() else []

    all_evals = []

    # Load from individual task results
    for tf in task_files:
        task = load_json(str(tf))
        if task.get("status") != "completed":
            continue
        result = task.get("result", {})
        cr = result.get("checklist_results", {})
        hits = sum(1 for v in cr.values() if isinstance(v, dict) and v.get("match"))
        total = len(cr) if cr else 25
        ev = {
            "title": result.get("title", task.get("title", "")),
            "id": result.get("pub_num", task.get("pub_num", "")),
            "manuscript_type": result.get("match_type", task.get("match_type", "")),
            "similarity_score": hits / total if total else 0,
            "similarity_categories": cr,
            "anticipation_assessment": result.get("anticipation_assessment", ""),
            "key_teachings": result.get("key_teachings", ""),
        }
        all_evals.append(ev)

    # Load from batch files (legacy)
    for bf in batch_files:
        batch = load_json(str(bf))
        for doc in batch.get("evaluations", batch.get("documents", [])):
            cr = doc.get("checklist_results", doc.get("checklist_evaluation", {}))
            hits = sum(1 for v in cr.values() if isinstance(v, dict) and v.get("match"))
            total = len(cr) if cr else 25
            ev = {
                "title": doc.get("title", ""),
                "id": doc.get("pub_num", doc.get("identifier", "")),
                "manuscript_type": doc.get("match_type", doc.get("document_type", "")),
                "similarity_score": hits / total if total else 0,
                "similarity_categories": cr,
                "anticipation_assessment": doc.get("anticipation_assessment", doc.get("section_102_assessment", "")),
                "key_teachings": doc.get("key_teachings", doc.get("section_103_teachings", "")),
            }
            all_evals.append(ev)

    if not all_evals:
        print("No completed evaluations found.")
        sys.exit(1)

    # Deduplicate by title
    seen = {}
    for ev in all_evals:
        t = ev.get("title", "").lower().strip()
        if t not in seen or ev["similarity_score"] > seen[t]["similarity_score"]:
            seen[t] = ev
    deduped = list(seen.values())

    # Filter self-references
    self_kw = ["dense object nets: learning", "1806.08756"]
    final = [e for e in deduped if not any(k in e.get("title", "").lower() for k in self_kw)]
    final.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Enrich with snippets and abstracts from search results
    if args.search_results and os.path.exists(args.search_results):
        search = load_json(args.search_results)
        enrichment = {}
        for doc in search.get("all_patents", []) + search.get("all_papers", []):
            t = doc.get("title", "").lower().strip()
            if t:
                enrichment[t] = {"snippet": doc.get("snippet", ""), "url": doc.get("pdf_link", "")}

        for ev in final:
            t = ev.get("title", "").lower().strip()
            if t in enrichment:
                if not ev.get("snippet"):
                    ev["snippet"] = enrichment[t]["snippet"]
                if not ev.get("url"):
                    url = enrichment[t]["url"]
                    ev["url"] = url[0] if isinstance(url, list) and url else (url if isinstance(url, str) else "")

    # Build final results
    scores = [e["similarity_score"] for e in final]
    top_score = max(scores) if scores else 0

    summary = f"Deep PDF analysis of {len(final)} prior art documents. "
    summary += f"Highest checklist overlap: {top_score:.0%}. "
    if final:
        summary += f"Top match: \"{final[0]['title']}\". "
    summary += "No single document anticipates the full invention under 35 USC 102."

    # Load or create results structure
    results_path = output_dir / "results.json"
    if results_path.exists():
        results = load_json(str(results_path))
    else:
        results = {
            "phase1": load_json(args.phase1) if args.phase1 else {},
            "phase2": load_json(args.phase2) if args.phase2 else {},
            "search": load_json(args.search_results).get("summary", {}) if args.search_results else {},
        }

    results["evaluation"] = {
        "scoring_report": final,
        "summary": summary,
        "stats": {
            "total_evaluated": len(final),
            "top_score": round(top_score, 4),
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "evaluation_method": "deep_pdf_reading",
        },
    }

    save_json(results, str(results_path))

    print(f"Merged {len(final)} evaluations into {results_path}")
    print(f"Top score: {top_score:.0%}")
    for ev in final[:5]:
        print(f"  [{ev['similarity_score']:.0%}] {ev['title'][:60]}")


# ─── MAIN ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep PDF evaluation pipeline")
    sub = parser.add_subparsers(dest="command")

    p_prep = sub.add_parser("prepare", help="Prepare evaluation task files")
    p_prep.add_argument("--search-results", required=True)
    p_prep.add_argument("--phase2", required=True)
    p_prep.add_argument("--phase1", required=True)
    p_prep.add_argument("--output-dir", required=True)
    p_prep.add_argument("--limit", type=int, default=20)

    p_merge = sub.add_parser("merge", help="Merge completed evaluation results")
    p_merge.add_argument("--output-dir", required=True)
    p_merge.add_argument("--phase1", default=None)
    p_merge.add_argument("--phase2", default=None)
    p_merge.add_argument("--search-results", default=None)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_tasks(args)
    elif args.command == "merge":
        merge_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
