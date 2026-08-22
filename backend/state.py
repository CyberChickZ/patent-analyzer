"""GraphState for the patent analysis pipeline.

Design principles:
- Only structured metadata in state (no raw LLM prompts/responses)
- File references as GCS URIs, not local paths
- Annotated reducers for fields written by parallel nodes
- hitl_pending field ready for P2 (HITL interrupt/resume)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_dicts(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class Event(TypedDict, total=False):
    ts: str
    phase: str
    kind: str
    message: str
    payload: dict


class HitlPending(TypedDict, total=False):
    """Populated when pipeline is waiting for human input."""
    type: str          # "checklist_review" | "search_review"
    prompt: str
    options: list[str]
    data: dict


class PhaseResult(TypedDict, total=False):
    status: str   # "running" | "completed" | "error"
    data: dict


class GraphState(TypedDict, total=False):
    # ── Job identity ──
    job_id: str
    input_gcs_uri: str        # gs://bucket/path/to/upload.pdf
    input_local_path: str     # transient: local copy of PDF for current instance
    output_dir: str           # local output directory

    # ── Config (set once at invocation) ──
    hitl_enabled: bool
    evolve: bool              # self-review mode
    notify_email: str

    # ── Pipeline status ──
    status: str               # "pending" | "running" | "waiting_for_hitl" | "completed" | "error"
    phase: str                # current phase: "phase1" .. "phase5"
    error: str

    # ── Phase 1: IDCA outputs ──
    source_title: str
    source_arxiv_id: str
    source_doi: str
    document_text: str        # text layer read by extraction / self_check: Doc JSON rendered with
                              # [S<path>.P<n>] markers (adapters.docjson), else fitz / plain text
    doc_json: dict | None     # IDCA Gemini transcription {title, abstract, sections[{heading, level,
                              # paragraphs}], figures[{label, caption}], equations[{label, latex}],
                              # references_count}; None when the call failed
    doc_json_stats: dict      # {sections, paragraphs, figures, equations, references_count, chars,
                              # source: gemini|none, fallback_paragraphs}
    status_determination: str # "Present" | "Implied" | "Absent"
    doc_type: str
    input_mode: str
    category: str
    fields_map: list[str]
    cpc_subclass: str
    source_citation: str
    publication_date: str
    summary: str              # invention summary (400-800 words)
    reasoning: str
    personas: dict[str, str]  # domain-specific persona strings

    # ── Phase 2: SSR outputs ──
    innovation_axes: list[dict]
    technology_choices: list[dict]
    patent_types: list[str]
    checklist: list[dict]          # final reviewed checklist
    delegation: dict               # search query plan {groups: [...]}

    # ── Phase 3: Search outputs ──
    search_results: Annotated[list[dict], operator.add]  # reducer: parallel append
    search_stats: dict             # {total_queries, total_results, ...}
    ranked_candidates: list[dict]  # after rerank + dedup + download

    # ── Phase 4: Evaluation outputs ──
    eval_results: Annotated[list[dict], operator.add]    # reducer: parallel append
    scoring_report: list[dict]     # scored + sorted
    eval_stats: dict               # {quote_stats: {...}, evaluated}
    extraction: dict               # Phase 2 extraction schema (candidate_inventions, ...)
    combination_analysis: str
    overall_summary: str
    novelty_score: float
    risk_level: str
    adjudication: dict             # rule verdict (patent_analyzer.adjudicate) + claim_chart once the draft node ran

    # ── Phase 4b: Draft claims (nodes/draft.py) ──
    draft_claims: dict             # {candidate_id, strategy, claims[{no, form, depends_on, preamble, limitations[{lid, text,
                                   #  basis[{element_id, evidence_quote, evidence_loc}], origin, coverage, flags}]}],
                                   #  avoidance, definiteness, recheck, llm_calls}

    # ── Phase 5: Report ──
    report_html_gcs: str      # gs:// URI of generated HTML report
    report_md_gcs: str
    results_json_gcs: str

    # ── HITL (P2 ready, activated in P2+) ──
    hitl_pending: HitlPending | None
    hitl_history: list[dict]  # [{phase, user_input, timestamp, changes}]
    hitl_response: dict  # user's HITL choice/comment from frontend submit
    pause_after: list[str]    # phases whose gate interrupts: idca | extract | search | evaluate | draft
    paused_at: str            # phase the graph is currently paused after ("" when running)
    user_edits: Annotated[list[dict], operator.add]   # {phase, kind, id/pub_num, op, before, after, ts}
    prompt_versions: Annotated[dict[str, int], _merge_dicts]   # prompt name -> registry version used

    # ── Observability (not checkpointed — side channel) ──
    events: Annotated[list[Event], operator.add]
    phase_results: dict[str, PhaseResult]
