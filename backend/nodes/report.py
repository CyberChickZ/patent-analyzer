"""Phase 5: Generate HTML/Markdown report, upload to GCS, send email."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from patent_analyzer.funnel import slim_search_stats
from state import GraphState

GCS_TIMEOUT_S = float(os.environ.get("GCS_TIMEOUT_S", "60"))


def _save_to_gcs(job_id: str, blob_suffix: str, content: str, content_type: str) -> str:
    """Best-effort upload, bounded. Returns "" on success or the error, so the
    caller can say so — a report that only exists on this instance's disk used
    to look identical to one safely in the bucket."""
    try:
        from google.cloud import storage
        bucket_name = os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1")
        prefix = os.environ.get("GCS_PREFIX", "patent-analyzer/jobs/")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{prefix}{job_id}/{blob_suffix}")
        blob.upload_from_string(content, content_type=content_type, timeout=GCS_TIMEOUT_S)
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[:300]


def _used_prompt_versions() -> dict:
    try:
        from app import prompts
        return prompts.used_versions()
    except Exception:
        return {}


async def _determination(state: GraphState, checklist: list, scoring_report: list, _event) -> dict:
    """Rule verdict from Phase 4 (+ claim chart) and, for a §103 verdict only, one LLM call that
    writes the reasoning from the rule output. The label is never touched here."""
    adj = state.get("adjudication") or (state.get("eval_stats") or {}).get("adjudication") or {}
    if not adj or not adj.get("n_elements"):
        return adj or {}
    from patent_analyzer.adjudicate import claim_chart
    adj = dict(adj)
    if not adj.get("claim_chart"):            # the draft node already built it from the same evidence
        adj["claim_chart"] = claim_chart(adj, checklist, scoring_report)
    adj["obviousness_explanation"] = ""
    if adj.get("label") == "103":
        try:
            from app.llm import explain_obviousness
            _event("llm", "Writing the §103 reasoning from the rule output (one call)")
            adj["obviousness_explanation"] = await explain_obviousness(
                adj, adj["claim_chart"], state.get("summary", ""), scoring_report)
        except Exception as e:
            _event("warn", f"§103 explanation skipped: {e}")
    _event("info", f"Determination: {adj.get('label')} ({adj.get('basis', '')}) — chart columns "
                   f"{[d.get('pub_num') or d.get('title') for d in adj['claim_chart']['docs']]}")
    return adj


async def report_node(state: GraphState) -> dict:
    """Phase 5: compile results, generate report, upload."""
    from patent_analyzer.report_generator import generate_html, generate_markdown
    from patent_analyzer.scorer import classify_risk

    job_id = state["job_id"]
    output_dir = state.get("output_dir", f"/tmp/outputs/{job_id}")
    job_dir = Path(output_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    events = []

    def _event(kind: str, message: str):
        events.append({"ts": datetime.now(timezone.utc).isoformat(),
                        "phase": "phase5", "kind": kind, "message": message})

    _event("start", "Generating report")

    scoring_report = state.get("scoring_report", [])
    top_score = scoring_report[0].get("similarity_score", 0) if scoring_report else 0
    checklist = state.get("checklist", [])

    adjudication = await _determination(state, checklist, scoring_report, _event)
    draft = state.get("draft_claims") or {}

    # Close the last phase's books (there is no gate after report) and merge what
    # the gates recorded in state with what this process measured — after a HITL
    # resume on a restarted server, only one of the two has the early phases.
    from patent_analyzer import metering
    metering.mark("report")
    phase_metrics = {**(state.get("phase_metrics") or {}), **metering.phases()}
    cost = metering.report(phase_metrics)
    ledger = metering.ledger(phase_metrics)
    _event("info", f"Run cost estimate: ${cost['totals']['cost_usd']:.4f} over "
                   f"{cost['totals']['llm_calls']} LLM calls and "
                   f"{cost['totals']['external_calls']} external calls")
    if most := ledger.get("most_expensive"):
        _event("info", f"Most expensive phase: {most['phase']} at ${most['cost_usd']:.4f} "
                       f"({most['share_of_total']:.0%} of the run), driven by {most['driver']}")
    if ledger["totals"]["failures"] or ledger["totals"]["degradations"]:
        _event("warn", f"{ledger['totals']['failures']} failed and "
                       f"{ledger['totals']['degradations']} degraded calls — see results.json `ledger`")

    results = {
        "job_id": job_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_filename": os.path.basename(state.get("input_local_path", "")),
        "source_title": state.get("source_title", ""),
        "phase1": {
            "status_determination": state.get("status_determination", ""),
            "has_innovation": state.get("status_determination", "") != "Absent",
            "doc_mode": state.get("doc_type", ""),
            "doc_type": state.get("doc_type", ""),
            "input_mode": state.get("input_mode", "academic_paper"),
            "fields_map": state.get("fields_map", []),
            "cpc_subclass": state.get("cpc_subclass", ""),
            "source_citation": state.get("source_citation", ""),
            "publication_date": state.get("publication_date", ""),
            "summary": state.get("summary", ""),
            "invention_type": state.get("category", ""),
            "reasoning": state.get("reasoning", ""),
            "doc_json_stats": state.get("doc_json_stats", {}),
        },
        "phase2": {
            "checklist": state.get("checklist", []),
            "innovation_axes": state.get("innovation_axes", []),
            "technology_choices": state.get("technology_choices", []),
            "delegation": state.get("delegation", {}),
        },
        "search": {
            "summary": slim_search_stats(state.get("search_stats", {}), job_id),
        },
        "extraction": state.get("extraction", {}),
        "eval_stats": state.get("eval_stats", {}),
        "user_edits": state.get("user_edits", []),           # reviewer changes at the phase gates
        "prompt_versions": {**_used_prompt_versions(), **(state.get("prompt_versions") or {})},
        "cost": cost,                                        # per-phase time / tokens / $ (patent_analyzer.metering)
        "ledger": ledger,                                    # the same spend as flat rows + most_expensive + incidents
        "adjudication": adjudication,
        "draft_claims": draft,
        "evaluation": {
            "scoring_report": scoring_report,
            "summary": state.get("overall_summary", ""),
            "combination_analysis": state.get("combination_analysis", ""),
            "stats": {
                "total_evaluated": len(scoring_report),
                "top_score": round(top_score, 4),
                "risk_level": classify_risk(top_score),
            },
        },
    }

    # the full funnel, beside results.json rather than inside it
    funnel = {"job_id": job_id, "generated_at": results["generated_at"], "search_stats": state.get("search_stats", {})}
    funnel_str = json.dumps(funnel, ensure_ascii=False, default=str)
    (job_dir / "funnel.json").write_text(funnel_str)

    results_str = json.dumps(results, indent=2, ensure_ascii=False, default=str)
    (job_dir / "results.json").write_text(results_str)
    _event("info", f"results.json {len(results_str) / 1e6:.2f} MB, funnel.json {len(funnel_str) / 1e6:.2f} MB")
    upload_errors = {}
    if err := _save_to_gcs(job_id, "funnel.json", funnel_str, "application/json"):
        upload_errors["funnel.json"] = err
    if err := _save_to_gcs(job_id, "results.json", results_str, "application/json"):
        upload_errors["results.json"] = err
    if state.get("doc_json"):
        # the Gemini transcription the text layer was rendered from (for coverage checks against the PDF)
        (job_dir / "doc_json.json").write_text(json.dumps(state["doc_json"], indent=1, ensure_ascii=False))
    if state.get("user_edits"):
        # reviewer changes alone, in the evals' element/doc vocabulary (future training / agreement data)
        edits_str = json.dumps(state["user_edits"], indent=2, ensure_ascii=False, default=str)
        (job_dir / "user_edits.json").write_text(edits_str)
        if err := _save_to_gcs(job_id, "user_edits.json", edits_str, "application/json"):
            upload_errors["user_edits.json"] = err

    from patent_analyzer.report_sections import inject_html, inject_md
    read_gap = (state.get("eval_stats") or {}).get("read_gap") or {}
    if read_gap.get("shortfall"):
        _event("warn", read_gap.get("headline", "") + " — " +
                       "; ".join(f"{k}: {v}" for k, v in (read_gap.get("reasons") or {}).items()))
    # generate_html/markdown render the determination themselves (top of the report); inject_* add the other sections
    # Decided before the report is written, because it is the half that can be:
    # whether a copy was asked for and whether this server can send one at all.
    # The transport result cannot be in the file — the file is the attachment.
    from app import email_notify
    email_plan = email_notify.plan_for(state.get("notify_email", ""))
    if email_plan["requested"] and not email_plan["enabled"]:
        _event("email_failed", f"No email will be sent to {email_plan['to']}: "
                               f"{email_notify.NOT_CONFIGURED}")

    html = inject_html(generate_html(results), results["extraction"], results["search"]["summary"],
                       scoring_report, checklist, adjudication=adjudication, draft=draft, cost=cost,
                       read_gap=read_gap, email=email_plan)
    (job_dir / "report.html").write_text(html, encoding="utf-8")
    if err := _save_to_gcs(job_id, "report.html", html, "text/html"):
        upload_errors["report.html"] = err

    md = inject_md(generate_markdown(results), results["extraction"], results["search"]["summary"],
                   scoring_report, checklist, adjudication=adjudication, draft=draft, cost=cost,
                   read_gap=read_gap, email=email_plan)
    (job_dir / "report.md").write_text(md, encoding="utf-8")
    if err := _save_to_gcs(job_id, "report.md", md, "text/markdown"):
        upload_errors["report.md"] = err

    gcs_bucket = os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1")
    gcs_prefix = os.environ.get("GCS_PREFIX", "patent-analyzer/jobs/")
    if upload_errors:
        _event("gcs_failed", f"{len(upload_errors)} artefact(s) could not be uploaded to "
                             f"gs://{gcs_bucket}/{gcs_prefix}{job_id}/ — they exist only on this "
                             f"instance's disk ({job_dir}): "
                             + "; ".join(f"{k}: {v}" for k, v in upload_errors.items()))
    else:
        _event("done", f"Report uploaded to gs://{gcs_bucket}/{gcs_prefix}{job_id}/")

    # Email notification
    notify_email = state.get("notify_email", "")
    email_status = {"requested": bool(notify_email), "to": notify_email,
                    "sent": False, "error": ""}
    if notify_email:
        if not email_plan["enabled"]:
            # already reported above, before the report was written
            email_status["error"] = email_notify.NOT_CONFIGURED
        else:
            try:
                subject = f"Patent Analysis Complete: {state.get('source_title', 'report')}"
                err = email_notify.send_report(notify_email, subject, md, job_id=job_id)
                if err:
                    email_status["error"] = err
                    _event("email_failed", f"Email to {notify_email}: {err}")
                else:
                    email_status["sent"] = True
                    _event("email_sent", f"Report emailed to {notify_email}")
            except Exception as e:
                email_status["error"] = f"{type(e).__name__}: {e}"
                _event("email_failed", email_status["error"])

    return {
        "phase": "phase5",
        "status": "completed",
        # on the job record, which is in GCS and is what /status returns — the
        # report file was written before the mail went out and cannot say
        "email_status": email_status,
        "report_html_gcs": f"gs://{gcs_bucket}/{gcs_prefix}{job_id}/report.html",
        "report_md_gcs": f"gs://{gcs_bucket}/{gcs_prefix}{job_id}/report.md",
        "results_json_gcs": f"gs://{gcs_bucket}/{gcs_prefix}{job_id}/results.json",
        "events": events,
        "phase_results": {"phase5": {"status": "completed", "data": {"done": True}}},
    }
