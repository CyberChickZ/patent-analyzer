"""
Patent Analyzer — FastAPI backend with A2A protocol support.

LLM: Google GenAI (Vertex AI) — Gemini
Search: SerpAPI + OpenAlex + sentence-transformers
Protocol: A2A JSON-RPC for agent-to-agent integration

Deploy: gcloud run deploy (see deploy/)
"""

import asyncio
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Patent Analyzer", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_BASE = Path(os.getenv("OUTPUT_DIR", "/tmp/outputs"))
OUTPUT_BASE.mkdir(exist_ok=True)

GCS_BUCKET = os.getenv("GCS_BUCKET", "aime-hello-world-amie-uswest1")
GCS_PREFIX = os.getenv("GCS_PREFIX", "patent-analyzer/jobs/")

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Job store — in-memory cache + GCS persistence
jobs: dict[str, dict] = {}

_gcs_client = None


def _get_gcs():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def _save_job(job: dict):
    """Persist job state to GCS."""
    # Local disk (for same-instance reads)
    job_dir = Path(job["output_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    state_path = job_dir / "state.json"
    state_data = json.dumps(job, ensure_ascii=False, default=str)
    state_path.write_text(state_data)
    # GCS (persistent across instances/deploys)
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job['id']}/state.json")
        blob.upload_from_string(state_data, content_type="application/json")
    except Exception:
        pass  # best-effort; local disk is the primary during pipeline run


def _save_report_to_gcs(job_id: str, report_html: str):
    """Upload report.html to GCS."""
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/report.html")
        blob.upload_from_string(report_html, content_type="text/html")
    except Exception:
        pass


def _save_results_to_gcs(job_id: str, results_json: str):
    """Upload results.json to GCS."""
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/results.json")
        blob.upload_from_string(results_json, content_type="application/json")
    except Exception:
        pass


def _load_job(job_id: str) -> dict | None:
    """Try to load job state from local disk, then GCS."""
    # Local disk first
    state_path = OUTPUT_BASE / job_id / "state.json"
    if state_path.exists():
        job = json.loads(state_path.read_text())
        jobs[job_id] = job
        return job
    # GCS fallback
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/state.json")
        if blob.exists():
            job = json.loads(blob.download_as_text())
            jobs[job_id] = job
            return job
    except Exception:
        pass
    return None


def _list_jobs_from_gcs() -> list[dict]:
    """List all jobs from GCS."""
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blobs = bucket.list_blobs(prefix=GCS_PREFIX)
        for blob in blobs:
            if blob.name.endswith("/state.json"):
                job_id = blob.name.replace(GCS_PREFIX, "").split("/")[0]
                if job_id and job_id not in jobs:
                    try:
                        job = json.loads(blob.download_as_text())
                        jobs[job_id] = job
                    except Exception:
                        pass
    except Exception:
        pass
    return list(jobs.values())


def _get_job(job_id: str) -> dict | None:
    """Get job from memory, disk, or GCS."""
    if job_id in jobs:
        return jobs[job_id]
    return _load_job(job_id)


# ─── REST API ──────────────────────────────────────────────────

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "version": "0.3.0"}


@app.post("/analyze")
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    evolve: bool = False,
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True)

    input_path = job_dir / file.filename
    content = await file.read()
    input_path.write_bytes(content)

    job = {
        "id": job_id,
        "status": "queued",
        "filename": file.filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "starting",
        "phases": {},
        "output_dir": str(job_dir),
        "input_path": str(input_path),
        "evolve": bool(evolve),
    }
    jobs[job_id] = job
    _save_job(job)

    background_tasks.add_task(run_pipeline, job_id)
    return {"job_id": job_id, "status": "queued", "evolve": bool(evolve)}


@app.get("/upload-url")
async def get_upload_url(filename: str, content_type: str = "application/pdf"):
    """Generate a signed PUT URL so the client can upload directly to GCS, bypassing 32MB Cloud Run limit."""
    from datetime import timedelta
    from google.auth import default
    from google.auth.transport.requests import Request as AuthRequest

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    job_id = str(uuid.uuid4())[:8]
    safe_name = re.sub(r"[^\w.-]", "_", filename)
    object_key = f"{GCS_PREFIX}{job_id}/upload/{safe_name}"

    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(object_key)
        # Use service-account credentials for signing (required on Cloud Run)
        creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(AuthRequest())
        signed_url = blob.generate_signed_url(
            expiration=timedelta(hours=1),
            method="PUT",
            version="v4",
            service_account_email="amie-backend-sa@aime-hello-world.iam.gserviceaccount.com",
            access_token=creds.token,
            content_type=content_type,
        )
    except Exception as e:
        raise HTTPException(500, f"Signed URL generation failed: {e}")

    return {
        "job_id": job_id,
        "signed_url": signed_url,
        "gcs_uri": f"gs://{GCS_BUCKET}/{object_key}",
        "object_key": object_key,
        "content_type": content_type,
        "headers": {"Content-Type": content_type},
    }


@app.post("/analyze-gcs")
async def start_analysis_from_gcs(
    background_tasks: BackgroundTasks,
    payload: dict,
    evolve: bool = False,
):
    """Start analysis with a file already uploaded to GCS via signed URL."""
    job_id = payload.get("job_id") or str(uuid.uuid4())[:8]
    gcs_uri = payload.get("gcs_uri")
    filename = payload.get("filename", "upload.pdf")
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        raise HTTPException(400, "Missing/invalid gcs_uri")

    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / filename

    # Download from GCS to local
    try:
        path_part = gcs_uri[len("gs://"):]
        bucket_name, _, object_key = path_part.partition("/")
        bucket = _get_gcs().bucket(bucket_name)
        blob = bucket.blob(object_key)
        blob.download_to_filename(str(input_path))
    except Exception as e:
        raise HTTPException(500, f"Failed to download from GCS: {e}")

    job = {
        "id": job_id,
        "status": "queued",
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "starting",
        "phases": {},
        "output_dir": str(job_dir),
        "input_path": str(input_path),
        "gcs_uri": gcs_uri,
        "evolve": bool(evolve or payload.get("evolve")),
    }
    jobs[job_id] = job
    _save_job(job)

    background_tasks.add_task(run_pipeline, job_id)
    return {"job_id": job_id, "status": "queued", "evolve": job["evolve"]}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Detect zombie jobs: running but no heartbeat for >15 min
    if job.get("status") == "running":
        last_hb = job.get("last_heartbeat") or job.get("created_at")
        if last_hb:
            try:
                from datetime import datetime as _dt
                hb_time = _dt.fromisoformat(last_hb.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - hb_time).total_seconds()
                if age > 900:  # 15 minutes
                    job["status"] = "error"
                    job["error"] = f"Pipeline appears stuck (no heartbeat for {int(age)}s). Likely killed by Cloud Run instance shutdown."
                    _save_job(job)
            except Exception:
                pass
    return job


@app.get("/events/{job_id}")
async def get_events(job_id: str, since: int = 0):
    """Get events for a job, optionally since a given index."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    events = job.get("events", [])
    return {
        "events": events[since:],
        "total": len(events),
        "status": job.get("status"),
        "phase": job.get("phase"),
    }


@app.get("/report/{job_id}")
async def get_report(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Try local first, then GCS
    report_path = Path(job.get("output_dir", f"/tmp/outputs/{job_id}")) / "report.html"
    if report_path.exists():
        return FileResponse(str(report_path), media_type="text/html")
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/report.html")
        if blob.exists():
            from fastapi.responses import Response
            return Response(content=blob.download_as_bytes(), media_type="text/html")
    except Exception:
        pass
    raise HTTPException(404, "Report not generated yet")


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    results_path = Path(job.get("output_dir", f"/tmp/outputs/{job_id}")) / "results.json"
    if results_path.exists():
        return FileResponse(str(results_path), media_type="application/json")
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/results.json")
        if blob.exists():
            from fastapi.responses import Response
            return Response(content=blob.download_as_bytes(), media_type="application/json")
    except Exception:
        pass
    raise HTTPException(404, "Results not ready")


@app.get("/jobs")
async def list_jobs():
    _list_jobs_from_gcs()
    return [{"id": j["id"], "status": j["status"], "phase": j["phase"], "filename": j["filename"], "created_at": j.get("created_at")}
            for j in jobs.values()]


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from memory, local disk, and GCS."""
    # Remove from memory
    jobs.pop(job_id, None)
    # Remove local disk
    local_dir = OUTPUT_BASE / job_id
    if local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)
    # Remove from GCS
    deleted_blobs = 0
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blobs = list(bucket.list_blobs(prefix=f"{GCS_PREFIX}{job_id}/"))
        for blob in blobs:
            blob.delete()
            deleted_blobs += 1
    except Exception as e:
        return {"deleted": True, "gcs_error": str(e), "local": True}
    return {"deleted": True, "blobs_removed": deleted_blobs, "local": True}


@app.post("/jobs/cleanup")
async def cleanup_old_jobs(days: int = 7):
    """Delete all jobs older than N days from GCS and memory."""
    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    _list_jobs_from_gcs()  # Make sure jobs is populated
    deleted = []
    for job_id in list(jobs.keys()):
        job = jobs[job_id]
        created_at = job.get("created_at", "")
        try:
            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            if ts < cutoff:
                await delete_job(job_id)
                deleted.append(job_id)
        except Exception:
            pass
    return {"deleted_count": len(deleted), "deleted_ids": deleted}


# ─── A2A Protocol ──────────────────────────────────────────────

A2A_AGENT_CARD = {
    "name": "Patent Analyzer",
    "description": "AI-powered patent novelty analysis. Upload a paper/patent PDF and get a comprehensive prior art search and novelty assessment report.",
    "url": "",
    "version": "0.3.0",
    "defaultInputModes": ["application/pdf", "text/plain", "application/json"],
    "defaultOutputModes": ["application/json", "text/plain"],
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    },
    "skills": [
        {
            "id": "patent_analyze",
            "name": "Analyze patent novelty",
            "description": "Upload a PDF and run full novelty analysis pipeline: invention detection, decomposition, prior art search, deep evaluation, and report generation.",
            "tags": ["patent", "novelty", "analysis", "prior-art"],
            "examples": ["Send a PDF file part to start analysis"],
        },
        {
            "id": "patent_status",
            "name": "Get analysis status",
            "description": "Poll the status of a running or completed patent analysis job.",
            "tags": ["patent", "status", "polling"],
            "examples": ["Send metadata.task_id to get job status"],
        },
    ],
    "supportsAuthenticatedExtendedCard": False,
}


def _build_agent_card(request: Request) -> dict:
    base_url = os.getenv("A2A_PUBLIC_URL") or str(request.base_url).rstrip("/")
    if base_url.startswith("http://") and "run.app" in base_url:
        base_url = base_url.replace("http://", "https://", 1)
    card = dict(A2A_AGENT_CARD)
    card["url"] = f"{base_url}/a2a"
    return card


def _job_to_a2a_task(job: dict) -> dict:
    status_map = {
        "queued": "submitted",
        "running": "working",
        "completed": "completed",
        "error": "failed",
    }
    job_status = job.get("status", "queued")
    # Determine which phase we're in for running jobs
    if job_status == "running":
        phase_label = job.get("phase", "running")
    else:
        phase_label = job_status

    task = {
        "id": job["id"],
        "contextId": job["id"],
        "status": {
            "state": status_map.get(job_status, "unknown"),
            "timestamp": job.get("created_at", datetime.now(timezone.utc).isoformat()),
            "message": {
                "messageId": f"{job['id']}-status",
                "role": "agent",
                "parts": [{"text": f"Phase: {phase_label}, Status: {job_status}"}],
            },
        },
        "metadata": {
            "phase": job.get("phase"),
            "phases": job.get("phases", {}),
            "filename": job.get("filename"),
        },
    }

    if job_status == "completed":
        task["metadata"]["report_url"] = f"/report/{job['id']}"
        task["metadata"]["results_url"] = f"/results/{job['id']}"

    if job_status == "error":
        task["metadata"]["error"] = job.get("error")

    return task


@app.get("/.well-known/agent-card.json")
async def well_known_agent_card(request: Request):
    return JSONResponse(_build_agent_card(request))


@app.get("/agent-card.json")
async def agent_card(request: Request):
    return JSONResponse(_build_agent_card(request))


@app.post("/a2a")
async def a2a_jsonrpc(request: Request, background_tasks: BackgroundTasks):
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32600, "message": "Invalid JSON body"},
        }, status_code=400)

    rpc_id = raw.get("id") if isinstance(raw, dict) else None
    method = raw.get("method") if isinstance(raw, dict) else None
    params = raw.get("params", {}) if isinstance(raw, dict) else {}

    # agent.getCard
    if method == "agent.getCard":
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": _build_agent_card(request),
        })

    # tasks.get / tasks/get
    if method in ("tasks.get", "tasks/get"):
        task_id = params.get("taskId") or params.get("id") or params.get("task_id")
        job = _get_job(task_id) if task_id else None
        if not job:
            return JSONResponse({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32001, "message": f"Task not found: {task_id}"},
            }, status_code=404)
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": _job_to_a2a_task(job),
        })

    # message/send — the main entry point
    if method in ("message/send", "tasks.create"):
        return await _handle_a2a_send(raw, rpc_id, params, request, background_tasks)

    return JSONResponse({
        "jsonrpc": "2.0", "id": rpc_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }, status_code=404)


async def _handle_a2a_send(
    raw: dict, rpc_id: Any, params: dict,
    request: Request, background_tasks: BackgroundTasks,
):
    import base64
    import binascii

    # Determine skill
    metadata = params.get("metadata", {})
    message = params.get("message", {})
    msg_meta = message.get("metadata", {})
    skill_id = metadata.get("skill_id") or msg_meta.get("skill_id") or "patent_analyze"

    # Handle patent_status
    if skill_id == "patent_status":
        task_id = metadata.get("task_id") or msg_meta.get("task_id")
        job = _get_job(task_id) if task_id else None
        if not job:
            return JSONResponse({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32001, "message": f"Task not found: {task_id}"},
            }, status_code=404)
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": _job_to_a2a_task(job),
        })

    # Handle patent_analyze — extract PDF from file parts
    pdf_bytes = None
    filename = "upload.pdf"

    for part in message.get("parts", []):
        file_obj = part.get("file")
        if not file_obj:
            continue
        fname = file_obj.get("name", "upload.pdf")
        mime = (file_obj.get("mimeType") or "application/pdf").lower()
        is_pdf = fname.lower().endswith(".pdf") or mime == "application/pdf"
        if not is_pdf:
            continue
        b64 = file_obj.get("bytes")
        if b64:
            try:
                pdf_bytes = base64.b64decode(b64, validate=True)
                filename = fname
                break
            except (binascii.Error, ValueError):
                continue

    if pdf_bytes is None:
        # Check for text-based invocation (e.g. from n8n with just a URL or instructions)
        text_parts = [p.get("text", "") for p in message.get("parts", []) if p.get("text")]
        if not text_parts:
            return JSONResponse({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32602, "message": "No PDF file part found. Send a FilePart with PDF bytes."},
            }, status_code=400)
        # For now, return helpful error. Future: support URL-based PDF fetching.
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "error": {"code": -32602, "message": "Text-only invocation not yet supported. Please send a PDF file part."},
        }, status_code=400)

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True)
    input_path = job_dir / filename
    input_path.write_bytes(pdf_bytes)

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "starting",
        "phases": {},
        "output_dir": str(job_dir),
        "input_path": str(input_path),
    }

    background_tasks.add_task(run_pipeline, job_id)

    return JSONResponse({
        "jsonrpc": "2.0", "id": rpc_id,
        "result": _job_to_a2a_task(jobs[job_id]),
    })


# ─── Pipeline ───────────────────────────────────────────────────

async def run_pipeline(job_id: str):
    """Full async pipeline — all phases."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from app.llm import (
        detect_invention, classify_document, summarize_invention,
        classify_category, decompose_invention, generate_checklist,
        plan_delegation, evaluate_batch, generate_overall_summary,
        self_check, review_phase_output,
    )
    from patent_analyzer.query_builder import build_all_queries
    from patent_analyzer.scorer import compute_total_score, classify_risk

    job = jobs[job_id]
    job_dir = Path(job["output_dir"])
    input_path = job["input_path"]
    job.setdefault("events", [])
    job["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    evolve = bool(job.get("evolve", False))

    def update(phase: str, status: str, data: dict | None = None):
        job["phase"] = phase
        job["status"] = status
        job["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
        if data:
            job["phases"][phase] = data
        _save_job(job)

    def heartbeat():
        """Cheap heartbeat — only updates timestamp, no event, no GCS write to disk only."""
        job["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
        # Local disk only — fast
        try:
            state_path = Path(job["output_dir"]) / "state.json"
            state_path.write_text(json.dumps(job, ensure_ascii=False, default=str))
        except Exception:
            pass

    def event(phase: str, kind: str, message: str, payload: dict | None = None):
        """Record a granular event so the user can see what's happening live."""
        evt = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "kind": kind,
            "message": message,
        }
        if payload:
            evt["payload"] = payload
        job["events"].append(evt)
        # Keep only last 500 events to avoid bloat
        if len(job["events"]) > 500:
            job["events"] = job["events"][-500:]
        job["last_heartbeat"] = evt["ts"]
        _save_job(job)

    # Install LLM hook so every call_llm() records an event with full prompt+response
    _current_llm_label = {"phase": "phase1", "label": "LLM call"}

    def llm_hook(system: str, user: str, response: str, thoughts: str = ""):
        phase = _current_llm_label["phase"]
        label = _current_llm_label["label"]
        payload = {
            "system": system,
            "user": user,
            "response": response,
        }
        if thoughts:
            payload["thoughts"] = thoughts
        suffix = f" — {len(response)} chars" + (f" (+{len(thoughts)} thought)" if thoughts else "")
        event(phase, "llm_response", f"{label}{suffix}", payload)

    from app.llm import set_llm_hook as _set_hook
    _set_hook(llm_hook)

    def llm_label(phase: str, label: str):
        _current_llm_label["phase"] = phase
        _current_llm_label["label"] = label

    async def maybe_review(phase: str, phase_name: str, task_desc: str,
                           original_input: str, output_text: str,
                           extra_context: str = "") -> dict | None:
        """Run a full-context review if evolve mode is on. Emits an event with
        the review JSON. Returns the review dict or None."""
        if not evolve:
            return None
        try:
            llm_label(phase, f"Review: {phase_name}")
            review = await review_phase_output(
                phase_name=phase_name,
                task_description=task_desc,
                original_input=original_input,
                output_to_review=output_text,
                extra_context=extra_context,
            )
        except Exception as e:
            event(phase, "review_failed", f"{phase_name} review crashed: {e}")
            return None
        kind = "review_pass" if review.get("good_enough") else "review_warning"
        msg = (review.get("what_works") or "")[:120] if review.get("good_enough") \
            else (review.get("what_doesnt") or "")[:120]
        event(phase, kind, f"{phase_name}: {msg}", {"review": review})
        return review

    async def try_step(label: str, phase: str, fn, validator_input: str | None = None):
        """
        Run a single LLM step. No retry. On failure, attach a human-readable
        failure_reason to the timeline (via summarize_failure) and return None.
        On success, optionally run self_check as a *quality signal only* — failures
        emit a quality_warning event but the result is still returned as-is.
        """
        llm_label(phase, label)
        try:
            result = await fn()
        except Exception as e:
            import traceback
            raw = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
            try:
                from app.llm import summarize_failure
                reason = await summarize_failure(label, raw, validator_input or "")
            except Exception as inner:
                reason = f"{type(e).__name__}: {e} (failure summarizer also failed: {inner})"
            event(phase, "failure_reason", f"{label}: {reason[:200]}", {
                "failure_reason": reason,
                "raw_error": raw[:600],
            })
            return None

        if validator_input is not None and result is not None:
            try:
                llm_label(phase, f"Self-check: {label}")
                check = await self_check(label, validator_input, str(result))
                if not check.get("ok"):
                    issues = ", ".join(check.get("issues", []))
                    event(phase, "quality_warning", f"{label}: {issues}", {"check": check})
                    # No retry — the warning is the signal. Result is returned anyway.
            except Exception:
                pass
        return result

    try:
        # ── Phase 1: IDCA ──
        update("phase1", "running")
        event("phase1", "start", "Reading PDF and detecting invention")

        source_title = ""
        if input_path.endswith(".pdf"):
            import fitz
            doc = fitz.open(input_path)
            paper_text = "\n".join(page.get_text() for page in doc[:10])
            # Extract source title from PDF metadata first, fallback to first non-trivial line
            meta_title = (doc.metadata or {}).get("title", "") or ""
            if meta_title and len(meta_title) > 5:
                source_title = meta_title.strip()
            else:
                # Heuristic: first non-empty line longer than 15 chars on first page
                first_page = doc[0].get_text() if len(doc) > 0 else ""
                for line in first_page.split("\n"):
                    line = line.strip()
                    if len(line) > 15 and not line.lower().startswith(("abstract", "copyright", "©")):
                        source_title = line
                        break
            doc.close()
            event("phase1", "info", f"Extracted {len(paper_text)} chars from PDF (first 10 pages)")
            if source_title:
                event("phase1", "info", f"Source title: {source_title[:120]}")
        else:
            paper_text = Path(input_path).read_text()

        detection = detect_invention(paper_text)
        event("phase1", "info", f"Invention detection (heuristic): {detection['status']}")
        if detection["status"] == "absent":
            # Old behavior was to stop the pipeline here, but the heuristic is
            # a keyword scan that misses any paper avoiding the standard
            # "we propose / novel method / apparatus" phrasing. Modern CV/ML
            # papers often phrase things differently and trigger a false absent.
            # The downstream LLM call (summarize_invention with thinking) can
            # judge invention presence far better than a keyword scan, so we
            # log a quality_warning and continue. If there is genuinely no
            # invention, summarize_invention's self_check will flag it.
            event(
                "phase1",
                "quality_warning",
                "Heuristic returned 'absent' (no invention keywords matched in first 10k chars). "
                "Proceeding anyway — letting the LLM decide.",
                {"raw": detection.get("raw", "")},
            )
            detection["status"] = "implied"  # downgraded but pipeline continues

        doc_type = classify_document(paper_text, os.path.basename(input_path))
        event("phase1", "info", f"Document type: {doc_type}")

        summary = await try_step(
            "Summarizing invention",
            phase="phase1",
            fn=lambda: summarize_invention(paper_text),
            validator_input=paper_text[:15000],  # match what summarize/decompose actually saw
        )
        if not summary:
            summary = "(invention summary unavailable — see failure_reason in events)"

        await maybe_review(
            phase="phase1",
            phase_name="invention summary",
            task_desc="Extract WHAT the invention is in 200-400 words from the paper text. "
                      "Must include core technical mechanisms (not just background or experiments). "
                      "Must use the paper's own terminology. Must not invent details.",
            original_input=paper_text[:15000],
            output_text=summary,
        )

        category = classify_category(summary)
        event("phase1", "info", f"Category: {category['invention_type']} — {category['reasoning']}")

        phase1 = {
            "status": detection["status"],
            "doc_mode": doc_type,
            "summary": summary,
            "invention_type": category["invention_type"],
            "reasoning": category["reasoning"],
        }
        save_json(phase1, job_dir / "phase1.json")
        update("phase1", "completed", phase1)

        # ── Phase 2: Decomposition ──
        update("phase2", "running")

        ucd = await try_step(
            "Decomposing invention",
            phase="phase2",
            fn=lambda: decompose_invention(paper_text),
            validator_input=paper_text[:15000],  # match what summarize/decompose actually saw
        )
        if not ucd:
            ucd = "(decomposition unavailable)"

        checklist = await try_step(
            "Generating evaluation checklist",
            phase="phase2",
            fn=lambda: generate_checklist(summary, ucd),
        ) or []
        event("phase2", "info", f"Generated {len(checklist)} checklist items")

        await maybe_review(
            phase="phase2",
            phase_name="evaluation checklist",
            task_desc="Produce 20-30 atomic, testable items that distinguish THIS invention "
                      "from generic prior art in the same field. Items must be specific (not "
                      "generic ML/engineering boilerplate) and evidence-checkable against PDFs.",
            original_input=f"INVENTION SUMMARY:\n{summary}\n\nDECOMPOSITION:\n{ucd}",
            output_text="\n".join(f"{i+1}. {c}" for i, c in enumerate(checklist)),
        )

        delegation = await try_step(
            "Planning search strategy",
            phase="phase2",
            fn=lambda: plan_delegation(summary, ucd, category["invention_type"]),
            validator_input=summary[:2000],
        ) or {"atoms": [], "groups": []}
        n_atoms = len(delegation.get("atoms", []))
        n_groups = len(delegation.get("groups", []))
        event("phase2", "info", f"Plan: {n_atoms} atoms in {n_groups} search groups")

        phase2 = {"ucd": ucd, "checklist": checklist, "delegation": delegation}
        save_json(phase2, job_dir / "phase2.json")
        update("phase2", "completed", {"checklist_count": len(checklist), "atoms": n_atoms, "groups": n_groups})

        # ── Phase 3: Multi-channel recall ──
        update("phase3", "running")
        event("phase3", "start", "Multi-channel prior art recall")

        queries = build_all_queries(delegation)
        save_json(queries, job_dir / "queries.json")
        n_query_groups = len(queries.get("groups", []))
        event("phase3", "info", f"Built queries for {n_query_groups} groups")

        from patent_analyzer.searcher import download_pdf
        from patent_analyzer.recall import pool as recall_pool
        from patent_analyzer.recall import serpapi as ch_serpapi
        from patent_analyzer.recall import semantic_scholar as ch_ss
        from patent_analyzer.recall import openalex as ch_oa
        from patent_analyzer.recall import arxiv as ch_arxiv
        from app.llm import summarize_failure as _summarize_failure

        # Build recall queries for the semantic channels (SS / OA / arXiv).
        # arXiv chokes on long natural-language queries (timeout / poor relevance),
        # so use a SHORT focused query: the source paper title + a few key terms
        # extracted from the first sentence of the summary. This is high-signal.
        # OpenAlex and Semantic Scholar handle longer queries fine, so they get
        # a richer query from the summary.
        def _short_query() -> str:
            # 1) prefer the actual paper title (highest signal, no LLM noise)
            if source_title and len(source_title) > 8:
                base = source_title
            else:
                # 2) first non-trivial sentence of the summary
                first_sent = (summary or "").split(".")[0]
                base = first_sent[:200] if first_sent else (summary or "")[:200]
            return " ".join(base.split())[:200]

        def _long_query() -> str:
            return " ".join((summary or "").split())[:400]

        recall_query_short = _short_query()  # arxiv
        recall_query_long = _long_query()    # ss + openalex
        # Backwards compat: a single recall_query for downstream code (failure_reason etc.)
        recall_query = recall_query_long or recall_query_short

        # SerpAPI channels iterate the structured group queries (cap 2 patent + 2 paper queries per group)
        async def run_serpapi_patents() -> tuple[list[recall_pool.Candidate], list[dict]]:
            out: list[recall_pool.Candidate] = []
            errs: list[dict] = []
            for group in queries.get("groups", []):
                for q in (group.get("patent_queries") or [])[:2]:
                    cands, err = await ch_serpapi.search_patents(q, max_pages=1)
                    if err:
                        errs.append({"query": q, "error": err, "group": group.get("group_id", "")})
                    out.extend(cands)
                    await asyncio.sleep(0.4)
            return out, errs

        async def run_serpapi_scholar() -> tuple[list[recall_pool.Candidate], list[dict]]:
            out: list[recall_pool.Candidate] = []
            errs: list[dict] = []
            for group in queries.get("groups", []):
                for q in (group.get("paper_queries") or [])[:2]:
                    cands, err = await ch_serpapi.search_scholar(q, max_pages=2)
                    if err:
                        errs.append({"query": q, "error": err, "group": group.get("group_id", "")})
                    out.extend(cands)
                    await asyncio.sleep(0.4)
            return out, errs

        async def run_semantic_scholar() -> tuple[list[recall_pool.Candidate], list[dict]]:
            cands, err = await ch_ss.search(recall_query_long, limit=50)
            return cands, ([{"query": recall_query_long, "error": err}] if err else [])

        async def run_openalex() -> tuple[list[recall_pool.Candidate], list[dict]]:
            cands, err = await ch_oa.search_works(recall_query_long, limit=50)
            return cands, ([{"query": recall_query_long, "error": err}] if err else [])

        async def run_arxiv() -> tuple[list[recall_pool.Candidate], list[dict]]:
            cands, err = await ch_arxiv.search(recall_query_short, limit=50)
            return cands, ([{"query": recall_query_short, "error": err}] if err else [])

        event("phase3", "info", "Launching 5 recall channels in parallel")
        channel_specs = [
            ("serpapi_patents", run_serpapi_patents),
            ("serpapi_scholar", run_serpapi_scholar),
            ("semantic_scholar", run_semantic_scholar),
            ("openalex", run_openalex),
            ("arxiv", run_arxiv),
        ]
        gathered = await asyncio.gather(
            *(spec[1]() for spec in channel_specs),
            return_exceptions=True,
        )

        channel_results: dict[str, list[recall_pool.Candidate]] = {}
        channel_meta: dict[str, dict] = {}

        for (name, _), result in zip(channel_specs, gathered):
            if isinstance(result, Exception):
                raw = f"{type(result).__name__}: {result}"
                try:
                    reason = await _summarize_failure(f"recall channel {name}", raw, recall_query[:500])
                except Exception:
                    reason = raw
                event("phase3", "channel_crashed", f"{name}: {reason[:200]}", {
                    "failure_reason": reason,
                    "raw_error": raw[:600],
                })
                channel_results[name] = []
                channel_meta[name] = {"crashed": True, "error": raw}
                continue

            cands, errs = result
            channel_results[name] = cands
            channel_meta[name] = {"count": len(cands), "errors": errs}
            event("phase3", "channel_done", f"{name}: {len(cands)} raw candidates", {
                "count": len(cands),
                "errors_n": len(errs),
            })
            # Summarize each unique error reason once per channel for the timeline
            seen_err_strs: set[str] = set()
            for e in errs[:5]:
                err_str = e.get("error", "")
                if not err_str or err_str in seen_err_strs:
                    continue
                seen_err_strs.add(err_str)
                try:
                    reason = await _summarize_failure(
                        f"recall channel {name}", err_str,
                        f"query={e.get('query', '')[:200]}",
                    )
                except Exception:
                    reason = err_str
                event("phase3", "channel_error", f"{name}: {reason[:160]}", {
                    "failure_reason": reason,
                    "raw_error": err_str[:300],
                    "query": e.get("query", "")[:200],
                })

        # Pool & dedupe across channels
        pooled = recall_pool.pool_and_dedupe(channel_results)
        active_channels = [name for name, cands in channel_results.items() if cands]
        event("phase3", "info",
              f"Pool: {len(pooled)} unique candidates from {len(active_channels)}/{len(channel_specs)} active channels",
              {"per_channel": {k: len(v) for k, v in channel_results.items()},
               "active_channels": active_channels})

        if not pooled:
            try:
                reason = await _summarize_failure(
                    "phase 3 recall (all channels)",
                    f"All {len(channel_specs)} channels returned 0 candidates. "
                    f"Channel meta: {channel_meta}",
                    recall_query[:500],
                )
            except Exception:
                reason = "All recall channels returned 0 candidates"
            event("phase3", "failure_reason", reason, {
                "failure_reason": reason,
                "channel_meta": channel_meta,
            })

        # ── Phase 3 elastic loop (evolve mode only) ──
        # After the first multi-channel pass, ask the reviewer whether the pool
        # is good enough. If "do_more", trigger a seed fan-out: take the top
        # 2-3 pooled candidates and fetch their Semantic Scholar recommendations
        # + references + citations to expand the pool. Hard cap at 1 expansion
        # round so the wall time stays bounded.
        if evolve and pooled:
            # Build a compact preview of the top 8 by source_score for the reviewer
            preview = sorted(pooled, key=lambda c: c.source_score, reverse=True)[:8]
            preview_text = "\n".join(
                f"- ({len(c.sources)} src, score={c.source_score:.1f}) "
                f"[{c.match_type}] {c.title[:140]} :: {(c.snippet or c.abstract)[:160]}"
                for c in preview
            )
            review = await maybe_review(
                phase="phase3",
                phase_name="prior art recall pool",
                task_desc="Decide whether the candidate pool is rich enough to deeply "
                          "evaluate. We want at least 5-10 candidates that are clearly in the "
                          "same technical area as the invention. If most candidates are "
                          "off-topic or shallow, ask for do_more (we will fan out via "
                          "Semantic Scholar from the closest hit).",
                original_input=f"INVENTION SUMMARY:\n{summary}",
                output_text=preview_text,
                extra_context=f"Total pool: {len(pooled)} candidates. "
                              f"Active channels: {', '.join(active_channels)}.",
            )
            if review and review.get("next_action") == "do_more":
                event("phase3", "elastic_expand", "Reviewer requested expansion — fanning out via Semantic Scholar",
                      {"hint": review.get("do_more_hint", "")})
                # Find the highest-scored candidate that has a Semantic Scholar paperId
                seed = None
                for c in preview:
                    s2_raw = (c.raw or {}).get("semantic_scholar") or {}
                    if s2_raw.get("paperId"):
                        seed = c
                        break
                # Fallback: query Semantic Scholar with the summary to find a seed
                if not seed:
                    seed_cands, _ = await ch_ss.search(recall_query, limit=1)
                    if seed_cands:
                        seed = seed_cands[0]
                        # Resolve the paperId for the seed
                        seed.raw = seed.raw or {}
                if seed:
                    seed_id = ((seed.raw or {}).get("semantic_scholar") or {}).get("paperId") \
                              or seed.pub_num or ""
                    if seed_id:
                        event("phase3", "elastic_seed", f"Seed: {seed.title[:120]}",
                              {"seed_id": seed_id, "seed_title": seed.title})
                        # Fan out: recommendations + references + citations
                        rec_cands, rec_err = await ch_ss.recommendations(seed_id, limit=30)
                        ref_cands, ref_err = await ch_ss.references(seed_id, limit=30)
                        cit_cands, cit_err = await ch_ss.citations(seed_id, limit=30)
                        for sub_name, sub_cands, sub_err in [
                            ("ss_recommendations", rec_cands, rec_err),
                            ("ss_references", ref_cands, ref_err),
                            ("ss_citations", cit_cands, cit_err),
                        ]:
                            if sub_err:
                                event("phase3", "channel_error", f"{sub_name}: {sub_err[:120]}",
                                      {"raw_error": sub_err})
                            channel_results[sub_name] = sub_cands
                            event("phase3", "channel_done", f"{sub_name}: {len(sub_cands)} candidates")
                        # Re-pool with the expanded set
                        pooled = recall_pool.pool_and_dedupe(channel_results)
                        active_channels = [name for name, cands in channel_results.items() if cands]
                        event("phase3", "info",
                              f"After expansion: {len(pooled)} unique candidates from "
                              f"{len(active_channels)} channels")

        # Convert Candidate objects to legacy doc dicts for downstream code
        all_docs = recall_pool.candidates_to_legacy_docs(pooled)

        # Approximate "patents vs papers" totals for backward compat
        patent_count = sum(1 for d in all_docs if d.get("match_type") == "Patent")
        paper_count = sum(1 for d in all_docs if d.get("match_type") != "Patent")

        search_data = {
            "channels": {k: {"count": len(v), **channel_meta.get(k, {})} for k, v in channel_results.items()},
            "all_docs": all_docs,
            "summary": {
                "total_patents": patent_count,
                "total_papers": paper_count,
                "total_unique": len(all_docs),
                "active_channels": active_channels,
            },
        }
        save_json(search_data, job_dir / "phase3_search.json")
        update("phase3", "completed", {
            "patents": patent_count,
            "papers": paper_count,
            "channels": len(active_channels),
        })

        # ── Phase 3b: Semantic ranking (title+snippet only, no PDFs yet) ──
        update("phase3b", "running")
        # all_docs already populated from the pooled multi-channel recall above

        # Filter out self-citations: documents whose title closely matches the source manuscript
        def title_similarity(a: str, b: str) -> float:
            """Quick token-based similarity for title matching."""
            if not a or not b:
                return 0.0
            ta = set(re.findall(r"\w+", a.lower()))
            tb = set(re.findall(r"\w+", b.lower()))
            if not ta or not tb:
                return 0.0
            inter = len(ta & tb)
            return inter / min(len(ta), len(tb))

        if source_title:
            before = len(all_docs)
            filtered = []
            removed = []
            for d in all_docs:
                sim = title_similarity(source_title, d.get("title", ""))
                if sim >= 0.75:
                    removed.append({"title": d.get("title", "")[:120], "sim": round(sim, 2)})
                else:
                    filtered.append(d)
            if removed:
                event("phase3b", "info", f"Filtered out {len(removed)} self-citation(s) of source manuscript", {"full": removed})
            all_docs = filtered

        event("phase3b", "start", f"Reranking {len(all_docs)} documents by title+snippet embedding")
        try:
            from patent_analyzer.semantic_search import rerank_by_embedding
            ranked = rerank_by_embedding(summary, all_docs, limit=30)
            event("phase3b", "info", f"Top 30 selected by embedding similarity")
        except Exception as e:
            event("phase3b", "warn", f"Embedding failed, fallback to keyword: {e}")
            ranked = sorted(all_docs, key=lambda x: x.get("_relevance", 0), reverse=True)[:30]

        # Now download PDFs ONLY for the top-ranked candidates
        event("phase3b", "info", f"Downloading PDFs for top {len(ranked)} candidates")
        papers_dir = job_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        download_count = 0
        for di, doc in enumerate(ranked):
            heartbeat()
            links = doc.get("pdf_link", "")
            if isinstance(links, str) and links:
                links = [links]
            elif not isinstance(links, list):
                continue
            title_clean = re.sub(r'[^\w\s-]', '', doc.get("title", "")).strip()[:80]
            fname = re.sub(r'\s+', '_', title_clean) or doc.get("pub_num", "x").replace("/", "_")
            for i, link in enumerate(links[:1]):
                local = download_pdf(link, papers_dir, f"{fname}.pdf")
                if local:
                    doc["local_pdf"] = local
                    download_count += 1
                    event("phase3b", "download", f"[{di+1}/{len(ranked)}] {doc.get('title','')[:80]}")
                    break

        update("phase3b", "completed", {"ranked": len(ranked), "downloaded": download_count})

        # In normal mode evaluate top 20; in evolve mode allow up to 30 via elastic batches.
        eval_cap = 30 if evolve else 20
        eval_candidates = [d for d in ranked if d.get("local_pdf") and Path(d["local_pdf"]).exists()][:eval_cap]
        event("phase3b", "info", f"{len(eval_candidates)} candidates with PDFs ready for deep evaluation")

        # ── Phase 4: Deep evaluation ──
        update("phase4", "running")
        event("phase4", "start", f"Evaluating up to {len(eval_candidates)} documents against {len(checklist)}-item checklist")

        scoring_report: list[dict] = []

        async def _evaluate_batch(batch_docs: list[dict]) -> list[dict]:
            for i, c in enumerate(batch_docs):
                event("phase4", "evaluating", f"{c.get('title', '')[:120]}",
                      {"type": c.get("match_type", "?")})
            llm_label("phase4", f"Per-document deep evaluation (batch of {len(batch_docs)})")
            return await evaluate_batch(summary, checklist, batch_docs, max_concurrent=5)

        def _absorb_results(eval_results: list[dict]):
            for er in eval_results:
                cr = er.get("checklist_results", {})
                score = compute_total_score(cr)
                scoring_report.append({
                    "title": er.get("title", ""),
                    "id": er.get("pub_num", ""),
                    "manuscript_type": er.get("match_type", ""),
                    "similarity_score": score,
                    "similarity_categories": cr,
                    "anticipation_assessment": er.get("anticipation_assessment", ""),
                    "key_teachings": er.get("key_teachings", ""),
                    "snippet": er.get("snippet", ""),
                    "abstract": er.get("abstract", ""),
                })

        if evolve and len(eval_candidates) > 5:
            # Elastic batches of 5; review after each batch decides whether to continue.
            cursor = 0
            batch_size = 5
            while cursor < len(eval_candidates):
                batch = eval_candidates[cursor:cursor + batch_size]
                event("phase4", "elastic_batch", f"Evaluating docs {cursor+1}-{cursor+len(batch)} of {len(eval_candidates)}")
                results_batch = await _evaluate_batch(batch)
                _absorb_results(results_batch)
                cursor += batch_size
                if cursor >= len(eval_candidates):
                    break
                # Review after batch — feed top results so far to the reviewer
                top_so_far = sorted(scoring_report, key=lambda x: x["similarity_score"], reverse=True)[:5]
                preview = "\n".join(
                    f"- ({s['similarity_score']:.0%}) {s['title'][:140]} :: {(s.get('key_teachings') or s.get('snippet') or '')[:160]}"
                    for s in top_so_far
                )
                review = await maybe_review(
                    phase="phase4",
                    phase_name="deep evaluation progress",
                    task_desc=f"We have evaluated {len(scoring_report)} of {len(eval_candidates)} candidates. "
                              "Decide whether continuing is worth it. If the top hits already strongly "
                              "anticipate the invention, proceed. If most top hits are weak/off-topic, "
                              "still do_more (we may need to dig deeper). If everything is clearly irrelevant, "
                              "you can ask to stop early.",
                    original_input=f"INVENTION SUMMARY:\n{summary}",
                    output_text=preview,
                    extra_context=f"Top score so far: {top_so_far[0]['similarity_score']:.0%}" if top_so_far else "no results yet",
                )
                if review and review.get("next_action") == "skip":
                    event("phase4", "elastic_stop", "Reviewer asked to stop early — remaining candidates skipped")
                    break
        else:
            # Normal mode: one big batch of up to 20
            results_all = await _evaluate_batch(eval_candidates)
            _absorb_results(results_all)

        event("phase4", "info", f"Got {len(scoring_report)} evaluation results from LLM")

        scoring_report.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_score = max((s["similarity_score"] for s in scoring_report), default=0)
        event("phase4", "info", f"Top match: {top_score:.0%} similarity")

        llm_label("phase4", "Generating overall novelty summary")
        overall = await generate_overall_summary(summary, scoring_report[:10])

        update("phase4", "completed", {"evaluated": len(scoring_report), "top_score": round(top_score, 4)})

        # ── Phase 5: Report ──
        update("phase5", "running")
        event("phase5", "start", "Generating HTML report and uploading to GCS")

        results = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_filename": os.path.basename(input_path),
            "source_title": source_title,
            "phase1": phase1,
            "phase2": phase2,
            "search": {
                "channels": search_data.get("channels", {}),
                "summary": search_data.get("summary", {}),
            },
            "evaluation": {
                "scoring_report": scoring_report,
                "summary": overall,
                "stats": {
                    "total_evaluated": len(scoring_report),
                    "top_score": round(top_score, 4),
                    "risk_level": classify_risk(top_score),
                },
            },
        }
        results_str = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        (job_dir / "results.json").write_text(results_str)
        _save_results_to_gcs(job_id, results_str)

        from patent_analyzer.report_generator import generate_html
        html = generate_html(results)
        (job_dir / "report.html").write_text(html, encoding="utf-8")
        _save_report_to_gcs(job_id, html)

        event("phase5", "done", f"Report uploaded to GCS: {GCS_BUCKET}/{GCS_PREFIX}{job_id}/")
        update("phase5", "completed", {"done": True})
        job["status"] = "completed"
        _save_job(job)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        event(job.get("phase", "?"), "error", f"Pipeline crashed: {e}", {"traceback": tb[:2000]})
        job["status"] = "error"
        job["error"] = str(e)
        job["traceback"] = tb
        _save_job(job)
    finally:
        from app.llm import set_llm_hook as _set_hook
        _set_hook(None)


def save_json(data: Any, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
