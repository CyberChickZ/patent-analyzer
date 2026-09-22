"""
Patent Analysis Pipeline — complete flow.

All steps are either FIXED (deterministic code) or LLM (prompt template).
Nothing runs ad-hoc in the terminal — every step is a script.

═══════════════════════════════════════════════════════════════
PHASE 1: IDCA (Invention Detection & Classification)
═══════════════════════════════════════════════════════════════
  [LLM]   Read paper → detect invention status (present/implied/absent)
  [FIXED] Parse first word → route (absent → END)
  [LLM]   Classify document type, summarize invention, categorize
  [FIXED] Parse first word → save phase1.json

═══════════════════════════════════════════════════════════════
PHASE 2: Decomposition & Planning
═══════════════════════════════════════════════════════════════
  [LLM]   Unstructured decomposition (UCD)
  [LLM]   Checklist generation (20-30 items)
  [LLM]   Delegation planning (atoms + search groups)
  [FIXED] query_builder.py → generate queries.json from delegation

═══════════════════════════════════════════════════════════════
PHASE 3: Search & Download
═══════════════════════════════════════════════════════════════
  [FIXED] searcher.py → SerpAPI search with retry + incremental save
  [FIXED] searcher.py → PDF download with retry
  [FIXED] fetch_abstracts.py → OpenAlex API for real abstracts
  [FIXED] prefilter.py → keyword relevance ranking → top N

═══════════════════════════════════════════════════════════════
PHASE 4: Deep Evaluation (read full PDFs)
═══════════════════════════════════════════════════════════════
  [FIXED] deep_evaluator.py prepare → create eval_tasks/*.json
          Each task = invention description + ONE prior art PDF + checklist
          NOT batched — 1 target × 1 prior art per task

  [LLM]   For each task: read target description + read prior art PDF
          → evaluate each checklist item with evidence citations
          → assess 102 anticipation + 103 key teachings
          → save result into task file

  [FIXED] deep_evaluator.py merge → aggregate all task results
          → deduplicate, filter self-refs, compute scores
          → enrich with snippets/abstracts
          → save results.json

═══════════════════════════════════════════════════════════════
PHASE 5: Report Generation
═══════════════════════════════════════════════════════════════
  [FIXED] report_generator.py → static HTML from results.json

═══════════════════════════════════════════════════════════════

CLI Commands (all fixed code):

  # Phase 3
  patent-search --queries-file queries.json --output search.json --log-file search.log
  patent-prefilter --search-results search.json --checklist-file phase2.json --output top200.json

  # Phase 3b: abstracts
  python -m patent_analyzer.fetch_abstracts --results results.json --limit 50

  # Phase 4
  python -m patent_analyzer.deep_evaluator prepare \\
      --search-results search.json --phase1 phase1.json --phase2 phase2.json \\
      --output-dir ./output --limit 20

  # (LLM agents run on eval_tasks/*.json — each reads 1 PDF)

  python -m patent_analyzer.deep_evaluator merge \\
      --output-dir ./output --search-results search.json

  # Phase 5
  patent-report --input results.json --output report.html
"""
