"""
Patent Analyzer — Automated patent novelty analysis pipeline.

Deterministic steps are fixed code (no LLM):
  - Query generation, SerpAPI search, deduplication, PDF download
  - Keyword-based pre-filtering, scoring aggregation
  - HTML report generation

LLM-dependent steps use prompt templates:
  - Invention detection (IDCA)
  - Summarization, checklist generation, delegation planning
  - Per-document checklist evaluation
"""

__version__ = "0.1.0"
