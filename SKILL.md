---
name: patent-analyze
description: >
  Analyze a paper/patent PDF for novelty under US patent law (35 USC 102/103).
  5-phase pipeline: detect + summarize (LLM w/ self-check retry), decompose +
  plan, multi-channel parallel recall (5 sources) + semantic rerank + self-
  citation filter, deep eval with source+candidate PDFs side-by-side, HTML
  report with feedback form.
---

# Patent Analysis Pipeline

**This file documents the LIVE Cloud Run service** (`app/main.py:run_pipeline`).
An older CLI flow with `eval_tasks/*.json` and manual checkpoints existed in
earlier versions; that flow is gone. The entry point for everything now is
`run_pipeline(job_id)` — REST `POST /analyze` and A2A `message/send` both call it.

## Invariants

- **Exactly 6 LLM calls per job** (+ per-doc deep eval). Tokens go to PDF analysis, not routing.
- **Every LLM step is wrapped in `try_step`**: exceptions become `failure_reason` events
  (LLM-summarized human-readable cause), not silent retries.
- **Every LLM call is captured** (system + user + response + thinking) in `state.json`
  so the timeline can replay exactly what Gemini saw.
- **Defensive guards against empty inputs**: when recall pool is empty, pipeline halts
  with `failed_recall` rather than letting the summarizer hallucinate fake prior art.

## Phase 1 — Detect & Summarize

- PyMuPDF extracts **page 1 only** (abstract + intro; the LLM needs the authors'
  self-description, not experimental tables). Pages 1–10 are kept for downstream.
- Source identity: `source_title` via metadata → fallback to first non-boilerplate
  line. Header boilerplate (`"Published as a conference paper at..."`, `"Under review"`,
  `"Proceedings of"`, `ICLR`/`NeurIPS`/`ICML`/`CVPR` short lines, etc.) is explicitly
  skipped. `source_arxiv_id` and `source_doi` extracted via regex from first 2 pages.
- Deterministic `detect_invention` / `classify_document` / `classify_category`
  (keyword rules — not LLM).
- `summarize_invention` (LLM, thinking budget 4096) → `self_check` LLM.
  Self-check failures emit `quality_warning`; no retry here (summary is downstream-tolerant).

## Phase 2 — Decompose & Plan

- `decompose_invention` (LLM, thinking 4096) — 8–20 atomic elements.
- `generate_checklist` (LLM, thinking 4096) — 20–30 testable items.
- `plan_delegation` (LLM, thinking 4096) — atoms × search groups with anchor
  and expansion terms per group.
- **`plan_delegation` self-check is retry-enabled** (`try_step(retry_with_feedback=True)`):
  if the reviewer flags a hallucinated atom, the plan is regenerated once with
  `feedback={issues, suggestion, previous}` injected into the prompt. Rationale: one
  fake atom pollutes all 5 search groups downstream.

## Phase 3 — Multi-channel Recall (parallel)

Five channels via `asyncio.gather`:

| Channel | Query used | Rate-limit strategy |
|---|---|---|
| `serpapi_patents` | per-group `patent_queries[:2]` | shared SerpAPI `Semaphore(1)` + 1.5s cooldown |
| `serpapi_scholar` | per-group `paper_queries[:2]` | same shared lock |
| `semantic_scholar` | short query (title / first sentence) | anonymous 100/5min; key = `SEMANTIC_SCHOLAR_KEY` |
| `openalex` | short query | key = `OPENALEX_KEY` |
| `arxiv` | short query | Atom API |

- **`recall_query_short`** (title / first non-trivial sentence, ≤200 chars) is used
  by SS / OpenAlex / arXiv. Long narrative queries return 0 from keyword-oriented APIs;
  SerpAPI channels use the structured group queries instead.
- Per-query failures that didn't kill the channel emit one compact `query_partial`
  event (not `channel_error`) — no timeline spam.
- **Pool dedupes** by stable identity (DOI / arXiv ID / patent number / title hash)
  with a √N cross-channel consensus boost.
- **If `pool = 0`, halt**: write `recall_failure: true` to `results.json`, set
  `job.status = failed_recall`, skip Phase 3b/4/5 entirely. This is the hard guard
  that prevents the earlier regression where `generate_overall_summary` hallucinated
  three fake US patent numbers when handed an empty matches list.

## Phase 3b — Semantic Rerank + Self-Citation Filter

- **Self-citation filter (3 layers)**, in order:
  1. arXiv ID exact match (source vs candidate) — strongest signal when both are tagged
  2. DOI exact match — same
  3. Title token similarity ≥ 0.75 — fallback when identity isn't known AND source_title
     is not header boilerplate
- `sentence-transformers` (`all-MiniLM-L6-v2`) reranks the filtered pool against the
  invention fingerprint. Top 30 proceed to PDF download.
- Download tries each candidate's `pdf_link`; 403 / paywall failures are expected.
- Candidates are usable for Phase 4 if **(PDF downloaded) OR (abstract/snippet ≥ 120 chars)**.
  Abstract-fallback candidates are tagged `eval_source="abstract"`.

## Phase 4 — Deep Evaluation

- `evaluate_batch(docs, max_concurrent=2, source_pdf_path=..., source_title=...)`.
  `max_concurrent=2` is intentional — prevents Gemini rate-limit contention with Phase 3.
- For each candidate:
  - **PDF path** (primary): `call_llm_with_pdfs([source_pdf, candidate_pdf])`. Gemini
    sees both files natively (multi-modal part bytes, no text extraction, no truncation).
    Per file: <18 MB → inline bytes; ≥18 MB → fallback to text of first 20 pages with
    a `[fallback_text ...]` prefix so the LLM knows.
  - **Abstract path** (fallback): `evaluate_single_document_text(abstract)`. Prompt
    explicitly tells the LLM the full PDF is unavailable and to only match on evidence
    the abstract states.
- Prompt requires `is_source_duplicate: true/false` at the top of the JSON output. If
  `true`, the pipeline rewrites all `match` fields to `false` (so a bad-recall candidate
  cannot silently become the top "prior art" against itself).
- **Post-eval filters**, in order:
  - **Selfmatch backstop**: drop if (`score ≥ 95%` AND `title_overlap ≥ 60%`) OR `title_overlap ≥ 80%`.
    Only fires when `source_title` is non-boilerplate. Emits `selfmatch_dropped` event.
  - **Zero-match filter**: drop any doc where all checklist items are `match=false`.
    Noise, not prior art.

## Phase 5 — Report

- `generate_overall_summary(summary, hit_matches[:10])` — LLM writes a 5-section novelty
  assessment ("What you invented", "What's already out there", "What appears new",
  "Honest assessment", "Suggested next steps"). **Hard assertion**: function raises
  `ValueError` if `top_matches=[]` (belt-and-suspenders beyond the Phase 3 halt).
- `generate_html(results)` renders an interactive report:
  - Expandable cards per candidate with matched/unmatched checklist drill-down
  - `abstract only` orange badge for candidates evaluated without PDF
  - Feedback form at the bottom (star rating + pre-set issue tags + freeform comment)
  - Feedback POSTs to `/feedback/{job_id}` → saved to `feedback.json` in GCS for
    offline analysis
- All artifacts uploaded to `gs://aime-hello-world-amie-uswest1/patent-analyzer/jobs/{job_id}/`:
  `state.json`, `results.json`, `report.html`, `feedback.json` (if submitted),
  `upload/{filename}` (if signed-URL flow).

## Evolve mode (`?evolve=true`)

Opt-in per request. Adds a full-context reviewer between phases. In Phase 3 the
reviewer can request `do_more` → fans out via Semantic Scholar recommendations +
references + citations from the top pooled hit (1-hop citation graph, bounded to
one expansion round). In Phase 4 it evaluates in elastic batches of 5 (cap 30)
with `elastic_stop` available if top hits are clearly irrelevant.

## Where to change what

| Change | File |
|---|---|
| Add/remove an LLM call | `app/llm.py` + wire it in `app/main.py:run_pipeline` via `try_step` |
| Add/remove a recall channel | `patent_analyzer/recall/<channel>.py` + register in `channel_specs` in `run_pipeline` |
| Adjust rate limiting | SerpAPI: `serpapi_lock` and `SERPAPI_COOLDOWN` in `run_pipeline`; HTTP retry backoff in `patent_analyzer/searcher.py` |
| Change self-citation thresholds | Phase 3b token-similarity: `0.75` in `run_pipeline`; Phase 4 post-eval: `0.95 / 0.60 / 0.80` |
| Change report UI | `patent_analyzer/report_generator.py` (CSS + HTML + JS all inline) |
| Change what gets saved | `results` dict in `run_pipeline` right before Phase 5 |

## Rules for adding new LLM steps

1. Must go through `call_llm` / `call_llm_with_pdfs` so `_emit` captures it.
2. Must set `llm_label(phase, label)` beforehand so the event gets attributed correctly.
3. If the step's output feeds a downstream step, wrap in `try_step` with
   `validator_input=...` so self-check runs.
4. If a self-check failure would poison downstream (e.g. hallucinated plan atom),
   set `retry_with_feedback=True` and accept a `feedback=None` kwarg in the step.
5. If the step can crash due to empty/unusable input, add a hard `ValueError` at
   the top of the function. Callers must guard, not the LLM.
