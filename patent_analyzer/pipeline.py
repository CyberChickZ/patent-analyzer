"""
Main pipeline orchestrator.

Shows which steps are deterministic (FIXED) vs LLM-dependent (LLM).

Phase 1: IDCA
  [LLM] Invention detection → parse first word (FIXED routing)
  [LLM] Document type classification → parse first word (FIXED routing)
  [LLM] Invention summary
  [LLM] Invention category → parse first word (FIXED routing)

Phase 2: Decomposition
  [LLM] Unstructured decomposition
  [LLM] Checklist generation (→ JSON parse is FIXED)
  [LLM] Delegation planning (→ JSON parse is FIXED)
  [FIXED] Query generation from delegation output

Phase 3: Search
  [FIXED] SerpAPI search with retry + incremental save
  [FIXED] Deduplication by pub_num/title
  [FIXED] PDF download with retry
  [FIXED] Pre-filtering by keyword relevance (top N selection)

Phase 4: Evaluation
  [LLM] Per-document checklist evaluation
  [FIXED] Score computation (matches / total)
  [FIXED] Risk classification (threshold-based)
  [FIXED] Aggregation + sorting

Phase 5: Report
  [LLM] Overall summary generation
  [FIXED] HTML report generation from JSON data
  [FIXED] Markdown export (client-side JS)
"""

import json
from pathlib import Path

from .config import EVAL_LIMIT, OUTPUT_DIR
from .query_builder import build_all_queries
from .prefilter import score_document, tokenize, bigrams
from .scorer import aggregate_evaluations, merge_into_final_results, classify_risk


def run_phase3_fixed(delegation: dict, api_key: str, output_dir: str = OUTPUT_DIR) -> dict:
    """
    Phase 3: All deterministic.
    1. Build queries from delegation (FIXED)
    2. Run SerpAPI search (FIXED)
    3. Pre-filter top N (FIXED)
    """
    queries = build_all_queries(delegation)

    # Save queries
    queries_path = Path(output_dir) / "queries.json"
    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)

    # Search would be run via searcher.py CLI
    # Pre-filter would be run via prefilter.py CLI
    return queries


def run_phase4_fixed(eval_batches: list[dict]) -> dict:
    """
    Phase 4 aggregation: All deterministic.
    Takes LLM evaluation outputs and computes final scores.
    """
    return aggregate_evaluations(eval_batches)


def run_phase5_fixed(phase1: dict, phase2: dict, search: dict, evaluation: dict) -> dict:
    """
    Phase 5 data assembly: All deterministic.
    Report HTML generation is also deterministic (generate_report.py).
    """
    return merge_into_final_results(phase1, phase2, search, evaluation)
