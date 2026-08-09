"""Phase 5: Generate HTML/Markdown report, upload to GCS, send email."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from state import GraphState


def _save_to_gcs(job_id: str, blob_suffix: str, content: str, content_type: str):
    try:
        from google.cloud import storage
        bucket_name = os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1")
        prefix = os.environ.get("GCS_PREFIX", "patent-analyzer/jobs/")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{prefix}{job_id}/{blob_suffix}")
        blob.upload_from_string(content, content_type=content_type)
    except Exception:
        pass


def _used_prompt_versions() -> dict:
    try:
        from app import prompts
        return prompts.used_versions()
    except Exception:
        return {}


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
        },
        "phase2": {
            "checklist": state.get("checklist", []),
            "innovation_axes": state.get("innovation_axes", []),
            "technology_choices": state.get("technology_choices", []),
            "delegation": state.get("delegation", {}),
        },
        "search": {
            "summary": state.get("search_stats", {}),
        },
        "extraction": state.get("extraction", {}),
        "eval_stats": state.get("eval_stats", {}),
        "user_edits": state.get("user_edits", []),           # reviewer changes at the phase gates
        "prompt_versions": {**_used_prompt_versions(), **(state.get("prompt_versions") or {})},
        "adjudication": state.get("adjudication") or (state.get("eval_stats") or {}).get("adjudication") or {},
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

    results_str = json.dumps(results, indent=2, ensure_ascii=False, default=str)
    (job_dir / "results.json").write_text(results_str)
    _save_to_gcs(job_id, "results.json", results_str, "application/json")
    if state.get("user_edits"):
        # reviewer changes alone, in the evals' element/doc vocabulary (future training / agreement data)
        edits_str = json.dumps(state["user_edits"], indent=2, ensure_ascii=False, default=str)
        (job_dir / "user_edits.json").write_text(edits_str)
        _save_to_gcs(job_id, "user_edits.json", edits_str, "application/json")

    from patent_analyzer.report_sections import inject_html, inject_md
    html = inject_html(generate_html(results), results["extraction"], results["search"]["summary"],
                       scoring_report, state.get("checklist", []), results["adjudication"])
    (job_dir / "report.html").write_text(html, encoding="utf-8")
    _save_to_gcs(job_id, "report.html", html, "text/html")

    md = inject_md(generate_markdown(results), results["extraction"], results["search"]["summary"],
                   scoring_report, state.get("checklist", []), results["adjudication"])
    (job_dir / "report.md").write_text(md, encoding="utf-8")
    _save_to_gcs(job_id, "report.md", md, "text/markdown")

    gcs_bucket = os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1")
    gcs_prefix = os.environ.get("GCS_PREFIX", "patent-analyzer/jobs/")
    _event("done", f"Report uploaded to gs://{gcs_bucket}/{gcs_prefix}{job_id}/")

    # Email notification
    notify_email = state.get("notify_email", "")
    if notify_email:
        try:
            from app.email_notify import send_report
            subject = f"Patent Analysis Complete: {state.get('source_title', 'report')}"
            err = send_report(notify_email, subject, md, job_id=job_id)
            if err:
                _event("email_failed", f"Email to {notify_email}: {err}")
            else:
                _event("email_sent", f"Report emailed to {notify_email}")
        except Exception as e:
            _event("email_failed", str(e))

    return {
        "phase": "phase5",
        "status": "completed",
        "report_html_gcs": f"gs://{gcs_bucket}/{gcs_prefix}{job_id}/report.html",
        "report_md_gcs": f"gs://{gcs_bucket}/{gcs_prefix}{job_id}/report.md",
        "results_json_gcs": f"gs://{gcs_bucket}/{gcs_prefix}{job_id}/results.json",
        "events": events,
        "phase_results": {"phase5": {"status": "completed", "data": {"done": True}}},
    }
