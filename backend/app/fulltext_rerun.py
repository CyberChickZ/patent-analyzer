"""Re-read the uploaded full texts alone, then re-adjudicate and re-report.

The unit of work is a reference, not a phase. When a reviewer supplies the PDF
for two papers the run could only see an abstract of, the right amount of work
is two deep reads — not the 25 the Phase 4 fan-out would redo, of documents
whose text has not changed.

The determination is *not* left alone, though. A second reference that turns out
to cover three more elements can move a run from "no blocking reference" to a
§103 flag, so everything downstream of the deep read is recomputed from scratch:
`reduce_eval` (scores, coverage, `adjudicate`) and then the report node, which
rewrites results.json, report.html and report.md and re-uploads all three.

What is reused from the existing HITL rerun (app/hitl.py) is its machinery —
the serial job queue, the job record, the event feed — not its mechanism. The
graph replay in `_run_langgraph_pipeline` needs a live checkpoint and a job that
is paused or failed; a finished job is neither, and on the default in-memory
checkpointer its thread is gone the moment the instance recycles. Everything
here is rebuilt from results.json instead, which is the artefact that survives.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

# The re-read is one or a handful of documents, and several agents share the
# Vertex quota. Two at a time, not the fan-out's four.
_RERUN_CONCURRENCY = 2


def _event(job: dict, phase: str, kind: str, message: str) -> None:
    job.setdefault("events", []).append(
        {"ts": datetime.now(timezone.utc).isoformat(), "phase": phase, "kind": kind, "message": message})


def _load_json(job: dict, name: str) -> dict:
    """A per-job artefact, local disk first and then the bucket."""
    from app.main import GCS_BUCKET, GCS_PREFIX, _get_gcs

    job_id = job.get("id", "")
    path = Path(job.get("output_dir", f"/tmp/outputs/{job_id}")) / name
    if path.exists():
        return json.loads(path.read_text())
    try:
        blob = _get_gcs().bucket(GCS_BUCKET).blob(f"{GCS_PREFIX}{job_id}/{name}")
        if blob.exists():
            return json.loads(blob.download_as_bytes())
    except Exception:
        pass
    return {}


def state_from_results(results: dict, job: dict, search_stats: dict) -> dict:
    """A GraphState good enough for `reduce_eval` and `report_node`.

    results.json is the inverse of what the report node writes, so this is that
    mapping run backwards. Only the keys those two read are filled; `doc_json`
    is left out on purpose, so the transcription already written beside the job
    is not rewritten from a record that does not carry it.
    """
    p1 = results.get("phase1") or {}
    p2 = results.get("phase2") or {}
    ev = results.get("evaluation") or {}
    job_id = job.get("id") or results.get("job_id", "")
    return {
        "job_id": job_id,
        "output_dir": job.get("output_dir", f"/tmp/outputs/{job_id}"),
        "input_local_path": job.get("input_path") or results.get("source_filename", ""),
        "source_title": results.get("source_title", ""),
        "status_determination": p1.get("status_determination", ""),
        "doc_type": p1.get("doc_type", ""),
        "input_mode": p1.get("input_mode", ""),
        "fields_map": p1.get("fields_map", []),
        "cpc_subclass": p1.get("cpc_subclass", ""),
        "source_citation": p1.get("source_citation", ""),
        "publication_date": p1.get("publication_date", ""),
        "summary": p1.get("summary", ""),
        "category": p1.get("invention_type", ""),
        "reasoning": p1.get("reasoning", ""),
        "doc_json_stats": p1.get("doc_json_stats", {}),
        "checklist": p2.get("checklist", []),
        "innovation_axes": p2.get("innovation_axes", []),
        "technology_choices": p2.get("technology_choices", []),
        "delegation": p2.get("delegation", {}),
        "search_stats": search_stats,
        "extraction": results.get("extraction", {}),
        "draft_claims": results.get("draft_claims", {}),
        "user_edits": results.get("user_edits", []),
        "prompt_versions": results.get("prompt_versions", {}),
        "phase_metrics": job.get("phase_metrics") or {},
        "overall_summary": ev.get("summary", ""),
        "combination_analysis": ev.get("combination_analysis", ""),
        # A re-read is not a new run, and the reviewer is sitting in front of the
        # page that triggered it. No second email.
        "notify_email": "",
    }


def _doc_for(row: dict, upload: dict) -> dict:
    """The candidate dict the deep read expects, rebuilt for one reference with
    the uploaded PDF standing in for the download that never worked."""
    return {
        "title": row.get("title", ""),
        "pub_num": row.get("pub_num", ""),
        "match_type": row.get("match_type") or "Paper",
        "doi": row.get("doi", ""),
        "landing_page": row.get("landing_page", ""),
        "local_pdf": upload.get("path", ""),
        "fulltext_tier": "user_upload",
        "fulltext_detail": (f"uploaded by hand on {str(upload.get('uploaded_at', ''))[:19]}"
                            + (f" ({upload['filename']})" if upload.get("filename") else "")),
        "fulltext_url": upload.get("gcs_uri", ""),
        "fulltext_download": "ok",
    }


async def rerun_evidence(job: dict) -> None:
    """Deep-read the staged uploads, recompute the verdict, rewrite the report.

    Called from `_run_langgraph_pipeline` when the job carries
    `_rerun_evidence`, so it runs on the same serial worker as a normal job and
    its progress shows up on the same `/events/{job_id}` feed.
    """
    from app.main import _refresh_accounting, _save_job
    from graph.eval_subgraph import _eval_one, reduce_eval
    from nodes.report import report_node
    from patent_analyzer import fulltext_gap as gap
    from patent_analyzer import metering

    job_id = job.get("id", "")
    refs = list(job.pop("_rerun_evidence", []) or [])
    uploads = job.get("fulltext_uploads") or {}
    job["status"] = "running"
    job["phase"] = "phase4"
    job["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    _save_job(job)

    metering.start_run(job_id)
    results = _load_json(job, "results.json")
    if not results:
        job["status"] = "error"
        job["error"] = "Evidence re-run: this job has no results.json to build on"
        _save_job(job)
        return
    # funnel.json holds the full search_stats; results.json only the slimmed
    # copy. Either works — the report node slims whatever it is given — but the
    # full one keeps the new report identical to the old outside the re-read.
    search_stats = (_load_json(job, "funnel.json") or {}).get("search_stats") \
        or ((results.get("search") or {}).get("summary") or {})

    rows = {r["ref_id"]: r for r in gap.gap_rows(results, uploads)}
    todo = [(r, rows[r], uploads[r]) for r in refs if r in rows and r in uploads]
    if not todo:
        job["status"] = "completed"
        job["error"] = ""
        _event(job, "phase4", "warn", "Evidence re-run: nothing to read (no uploaded PDF matched a reference)")
        _save_job(job)
        return

    _event(job, "phase4", "start", f"Evidence re-run: deep-reading {len(todo)} uploaded full text(s) — "
                                  + "; ".join((r["title"] or r["ref_id"])[:70] for _, r, _ in todo))

    checklist = (results.get("phase2") or {}).get("checklist") or []
    summary = (results.get("phase1") or {}).get("summary") or ""
    src = job.get("input_path") or ""
    source_pdf = src if src.endswith(".pdf") and Path(src).exists() else None

    sem = asyncio.Semaphore(_RERUN_CONCURRENCY)

    async def _one(row: dict, upload: dict) -> dict:
        async with sem:
            out = await _eval_one({"summary": summary, "checklist": checklist,
                                   "doc": _doc_for(row, upload), "source_pdf_path": source_pdf,
                                   "source_title": results.get("source_title", "")})
        return out["eval_results"][0]

    fresh = await asyncio.gather(*[_one(row, up) for _, row, up in todo], return_exceptions=True)

    scoring_report = list((results.get("evaluation") or {}).get("scoring_report") or [])
    by_ref = {gap.ref_id(r): i for i, r in enumerate(scoring_report)}
    read_ok, failed = [], []
    for (ref, row, up), res in zip(todo, fresh):
        if isinstance(res, BaseException):
            failed.append(ref)
            _event(job, "phase4", "warn", f"Evidence re-run failed for {row.get('title') or ref}: "
                                          f"{type(res).__name__}: {res}")
            continue
        if res.get("source") not in ("pdf", "full_text"):
            # The upload was read and produced nothing — say so instead of
            # letting it look like a successful re-read of a covered reference.
            failed.append(ref)
            _event(job, "phase4", "warn", f"Uploaded PDF for {row.get('title') or ref} produced no "
                                          f"evidence ({res.get('source')}: {res.get('no_content_reason', '')})")
        else:
            read_ok.append(ref)
            _event(job, "phase4", "info", f"Read {row.get('title') or ref} from the uploaded PDF "
                                          f"({len(res.get('checklist_results') or {})} criteria answered)")
        idx = by_ref.get(ref)
        if idx is None:
            scoring_report.append(res)
        else:
            scoring_report[idx] = res
        uploads.get(ref, {})["reread"] = True

    metering.mark("evidence_rerun")

    # Everything downstream of the deep read is recomputed, not patched: one
    # more covered element can change the §102/§103 determination, and a verdict
    # carried over from the previous run would be a verdict about different
    # evidence.
    n_delivered = int(((results.get("eval_stats") or {}).get("read_gap") or {}).get("delivered") or 0)
    reduced = await reduce_eval({
        "eval_results": scoring_report,
        "checklist": checklist,
        # _read_gap reads only the *length* of this list (graph/eval_subgraph),
        # so the delivered count from the original run is restored as a list of
        # that length rather than re-deriving a pool this process does not hold.
        "ranked_candidates": [{}] * (n_delivered or len(scoring_report)),
    })

    before = (results.get("adjudication") or {}).get("label_text") or (results.get("adjudication") or {}).get("label", "")
    after = (reduced.get("adjudication") or {}).get("label_text") or (reduced.get("adjudication") or {}).get("label", "")
    for evt in reduced.get("events", []):
        job.setdefault("events", []).append(evt)
    _event(job, "phase4", "info" if before == after else "warn",
           f"Determination after the re-read: {after or '—'}"
           + ("" if before == after else f" (was: {before or '—'})"))

    state = state_from_results(results, job, search_stats)
    state.update({k: reduced[k] for k in ("scoring_report", "novelty_score", "risk_level",
                                          "adjudication", "eval_stats", "overall_summary",
                                          "combination_analysis") if k in reduced})
    job["phase"] = "phase5"
    _save_job(job)
    patch = await report_node(state)
    for evt in patch.get("events", []):
        job.setdefault("events", []).append(evt)

    job.setdefault("phases", {})["phase4"] = {
        "evaluated": len(reduced.get("scoring_report") or []),
        "top_score": round((reduced.get("scoring_report") or [{}])[0].get("similarity_score", 0), 4)}
    job.setdefault("fulltext_reruns", []).append({
        "at": datetime.now(timezone.utc).isoformat(),
        "refs": refs, "read": read_ok, "failed": failed,
        "determination_before": before, "determination_after": after})
    job["fulltext_uploads"] = uploads
    job["status"] = "completed"
    job["phase"] = "phase5"
    job["error"] = ""
    _refresh_accounting(job)
    job["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    _save_job(job)
