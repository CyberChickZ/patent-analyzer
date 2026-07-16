"""HITL (Human-in-the-Loop) API endpoints.

POST /hitl-response/{job_id} — submit user's review and resume pipeline
POST /hitl-revise/{job_id}   — revise checklist via LLM based on user instructions
GET  /hitl-status/{job_id}   — check if pipeline is waiting for HITL input
"""

import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class HitlResponse(BaseModel):
    choice: str  # "A" | "B" | "C" | "D"
    comment: str = ""
    modifications: dict | None = None


class HitlStatus(BaseModel):
    waiting: bool
    phase: str | None = None
    type: str | None = None  # "checklist_review" | "search_review"
    prompt: str | None = None
    options: list[str] | None = None
    data: dict | None = None


@router.get("/hitl-status/{job_id}")
async def get_hitl_status(job_id: str) -> HitlStatus:
    """Check if a job is waiting for human input."""
    from app.main import _get_job

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    hitl_pending = job.get("hitl_pending")
    if not hitl_pending:
        return HitlStatus(waiting=False)

    return HitlStatus(
        waiting=True,
        phase=job.get("phase"),
        type=hitl_pending.get("type"),
        prompt=hitl_pending.get("prompt"),
        options=hitl_pending.get("options"),
        data=hitl_pending.get("data"),
    )


@router.post("/hitl-response/{job_id}")
async def submit_hitl_response(job_id: str, response: HitlResponse):
    """Submit user's HITL review and resume the pipeline (second half)."""
    from app.main import _enqueue_job, _get_job, _save_job, jobs

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    hitl_pending = job.get("hitl_pending")
    if not hitl_pending:
        raise HTTPException(400, "Job is not waiting for HITL input")

    hitl_record = {
        "phase": job.get("phase"),
        "type": hitl_pending.get("type"),
        "choice": response.choice,
        "comment": response.comment,
        "modifications": response.modifications,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    job.setdefault("hitl_history", []).append(hitl_record)
    job["hitl_pending"] = None
    job["status"] = "queued"
    job["_hitl_resume"] = {
        "choice": response.choice,
        "comment": response.comment,
        "modifications": response.modifications,
    }
    _save_job(job)

    _enqueue_job(job_id)

    return {
        "status": "resumed",
        "job_id": job_id,
        "hitl_record": hitl_record,
    }


class ReviseRequest(BaseModel):
    instructions: str
    current_checklist: list[dict]


@router.post("/hitl-revise/{job_id}")
async def revise_checklist(job_id: str, req: ReviseRequest):
    """Use LLM to revise checklist based on user's natural-language instructions."""
    from app.main import _get_job, _save_job

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "waiting_for_hitl":
        raise HTTPException(400, "Job is not waiting for HITL input")

    from app.llm import call_llm

    cl_text = "\n".join(
        f"{i+1}. [{c.get('id','c'+str(i+1))}] (w={float(c.get('weight',0)):.2f}) {c.get('criterion','')}"
        for i, c in enumerate(req.current_checklist)
    )

    system = "You are a patent evaluation expert. Revise the checklist based on the user's instructions. Return ONLY a JSON array of objects with keys: id, criterion, weight (0-1 float). No markdown, no explanation."
    user_msg = f"Current checklist:\n{cl_text}\n\nUser instructions:\n{req.instructions}\n\nReturn the revised checklist as a JSON array."

    raw = await call_llm(system, user_msg)
    # Parse JSON from response
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        revised = json.loads(text)
        if not isinstance(revised, list):
            raise ValueError("Expected JSON array")
    except Exception:
        raise HTTPException(500, f"LLM returned invalid JSON: {raw[:200]}")

    # Update job's hitl_pending with revised checklist
    hitl_pending = job.get("hitl_pending", {})
    hitl_pending.setdefault("data", {})["checklist"] = revised
    job["hitl_pending"] = hitl_pending
    job.setdefault("hitl_history", []).append({
        "type": "revision",
        "instructions": req.instructions,
        "original_count": len(req.current_checklist),
        "revised_count": len(revised),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save_job(job)

    return {"checklist": revised, "count": len(revised)}
