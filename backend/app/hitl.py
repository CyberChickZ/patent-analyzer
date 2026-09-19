"""HITL (Human-in-the-Loop) API endpoints.

POST /hitl-response/{job_id} — submit user's review and resume pipeline
POST /hitl-revise/{job_id}   — revise checklist via LLM based on user instructions
GET  /hitl-status/{job_id}   — check if pipeline is waiting for HITL input
"""

import json
import os
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
    # HumanResponse for the gate (graph/gates.py): the legacy form only edits the checklist
    mods = response.modifications or {}
    args = {k: v for k, v in mods.items() if k in ("checklist", "extraction", "summary", "ranked_candidates", "scoring_report")}
    if args:
        job["_pending_response"] = {"type": "edit", "args": args}
    elif response.comment and response.choice not in ("A", "accept"):
        job["_pending_response"] = {"type": "response", "args": response.comment}
    else:
        job["_pending_response"] = {"type": "accept", "args": None}
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


# ─── Phase-level pause / edit / resume (graph gates) ──────────────────────
#
# GET   /jobs/{id}/state   → what the reviewer sees: paused phase, editable values, context
# PATCH /jobs/{id}/state   → stage edits (whole-value replace of editable keys); nothing runs
# POST  /jobs/{id}/resume  → {"action": "continue" | "rerun_phase", "prompt_overrides": {...}}

from graph.gates import EDITABLE, PHASES, SHOWN  # noqa: E402


@router.get("/jobs/{job_id}/state")
async def get_job_state(job_id: str):
    from app.main import _get_job

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # A failed job answers here too, with the phase that broke standing in for
    # the paused phase, so the UI can show what happened and offer the rerun.
    phase = job.get("paused_at") or job.get("failed_phase") or ""
    saved = job.get("_hitl_saved_state") or {}
    pending = job.get("_pending_edits") or {}
    values = {k: pending.get(k, saved.get(k)) for k in EDITABLE.get(phase, ()) if k in saved or k in pending}
    context = {k: saved.get(k) for k in SHOWN.get(phase, ()) if k in saved}
    if phase == "search":
        context["queries"] = [q for r in (saved.get("search_stats") or {}).get("loop_rounds", []) for q in r.get("queries", [])]
    return {"job_id": job_id, "status": job.get("status"), "phase": job.get("phase"),
            "error": job.get("error") or "", "failed_phase": job.get("failed_phase") or "",
            "failed_node": job.get("failed_node") or "", "paused_at": job.get("paused_at") or "",
            "pause_after": job.get("pause_after") or [], "editable": list(EDITABLE.get(phase, ())),
            "values": values, "context": context, "pending_edits": sorted(pending),
            "user_edits": job.get("user_edits") or [], "prompt_versions": job.get("prompt_versions") or {},
            "phase_checkpoints": sorted((job.get("phase_checkpoints") or {}).keys())}


@router.patch("/jobs/{job_id}/state")
async def patch_job_state(job_id: str, edits: dict):
    """Stage whole-value replacements for the paused phase's editable keys.
    Applied by the gate on resume (never via update_state: reducer channels
    would accumulate)."""
    from app.main import _get_job, _save_job

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "waiting_for_hitl":
        raise HTTPException(400, "Job is not paused")
    phase = job.get("paused_at") or ""
    allowed = set(EDITABLE.get(phase, ()))
    bad = sorted(set(edits) - allowed)
    if bad:
        raise HTTPException(400, f"not editable after '{phase}': {bad}; editable: {sorted(allowed)}")
    staged = job.setdefault("_pending_edits", {})
    staged.update(edits)
    _save_job(job)
    return {"job_id": job_id, "paused_at": phase, "pending_edits": sorted(staged)}


class ResumeRequest(BaseModel):
    action: str = "continue"          # continue | rerun_phase
    prompt_overrides: dict | None = None
    comment: str = ""


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str, req: ResumeRequest):
    from app.main import _enqueue_job, _get_job, _save_job

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    status = job.get("status")
    if status not in ("waiting_for_hitl", "error"):
        raise HTTPException(400, f"Job is neither paused nor failed (status={status!r})")

    # A failed job is restartable at the phase that broke: the graph thread is
    # still parked on the node that raised (app.main._record_phase_failure), so
    # "rerun_phase" retries exactly that node. Without this, a failure was a
    # dead end — the only way forward was re-uploading the PDF as a new job.
    if status == "error":
        if req.action != "rerun_phase":
            raise HTTPException(400, "a failed job can only be resumed with action=rerun_phase")
        if req.prompt_overrides:
            job.setdefault("prompt_overrides", {}).update(req.prompt_overrides)
        failed = job.get("failed_phase") or ""
        job["_retry_failed"] = True
        job.pop("_pending_response", None)
        job.setdefault("hitl_history", []).append(
            {"phase": failed, "action": "rerun_phase", "after": "error",
             "error": (job.get("error") or "")[:300],
             "timestamp": datetime.now(timezone.utc).isoformat()})
        job["status"] = "queued"
        _save_job(job)
        _enqueue_job(job_id)
        return {"job_id": job_id, "status": "resumed", "action": "rerun_phase", "from_phase": failed,
                "retried_node": job.get("failed_node") or ""}

    phase = job.get("paused_at") or ""
    if req.prompt_overrides:
        job.setdefault("prompt_overrides", {}).update(req.prompt_overrides)
    if req.action == "rerun_phase":
        if phase not in PHASES:
            raise HTTPException(400, "nothing to rerun")
        job["_replay_from"] = phase
        job.pop("_pending_response", None)
    elif req.action == "continue":
        staged = job.pop("_pending_edits", None) or {}
        if staged:
            job["_pending_response"] = {"type": "edit", "args": staged}
        elif req.comment:
            job["_pending_response"] = {"type": "response", "args": req.comment}
        else:
            job["_pending_response"] = {"type": "accept", "args": None}
    else:
        raise HTTPException(400, "action must be continue or rerun_phase")
    job.setdefault("hitl_history", []).append({"phase": phase, "action": req.action, "comment": req.comment,
                                              "edited": sorted((job.get("_pending_response") or {}).get("args") or {})
                                              if isinstance((job.get("_pending_response") or {}).get("args"), dict) else [],
                                              "timestamp": datetime.now(timezone.utc).isoformat()})
    job["hitl_pending"] = None
    job["status"] = "queued"
    _save_job(job)
    _enqueue_job(job_id)
    return {"job_id": job_id, "status": "resumed", "action": req.action, "from_phase": phase}


# ─── Prompt registry ───────────────────────────────────────────────────────
#
# GET /prompts            → names, current version, version count
# GET /prompts/{name}     → default text + every saved version
# PUT /prompts/{name}     → {"text": ..., "by": ..., "make_current": true} → new version
# PUT /prompts/{name}/current → {"version": n}

from app import prompts as _prompts  # noqa: E402
import app.llm  # noqa: E402,F401  (registers the default prompt templates)


@router.get("/prompts")
async def list_prompts():
    out = []
    for n in _prompts.names():
        d = _prompts.describe(n)
        out.append({"name": n, "current": d["current"], "n_versions": len(d["versions"])})
    return out


@router.get("/prompts/{name}")
async def get_prompt(name: str):
    try:
        return _prompts.describe(name)
    except KeyError:
        raise HTTPException(404, f"unknown prompt {name}; known: {_prompts.names()}")


class PromptPut(BaseModel):
    text: str
    by: str = ""
    make_current: bool = True
    #: What the editor was trying to do, in their words. Kept with the version
    #: so the timeline can say why a prompt changed, not only that it did.
    instruction: str = ""


def _diff_summary(before: str, after: str, width: int = 400) -> str:
    """A few lines of unified diff, enough to recognise the change in a list."""
    import difflib
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                      lineterm="", n=1))[2:]
    out, n = [], 0
    for line in lines:
        if not line.startswith(("+", "-")):
            continue
        out.append(line[:160])
        n += len(out[-1])
        if n > width:
            out.append("…")
            break
    return "\n".join(out)


@router.put("/prompts/{name}")
async def put_prompt(name: str, req: PromptPut):
    if not req.text.strip():
        raise HTTPException(400, "empty prompt")
    try:
        before, _ = _prompts.get(name)
        v = _prompts.put(name, req.text, by=req.by, make_current=req.make_current)
    except KeyError:
        raise HTTPException(404, f"unknown prompt {name}")
    # The audit line is best effort on purpose: losing it must not lose the
    # prompt the person just wrote.
    from app import feedback_store
    feedback_store.record_prompt_edit(name, v, req.by or "unknown", req.instruction,
                                      _diff_summary(before, req.text))
    return {"name": name, "version": v, "current": _prompts.describe(name)["current"]}


def _specifics_lost(before: str, after: str) -> list[str]:
    """Named things that were in the template and are not in the rewrite.

    The placeholder check is necessary and not sufficient. Dogfooding this
    endpoint on four real prompts (2026-09-20), every proposal kept its
    placeholders and two of them quietly dropped something that mattered: the
    MPEP 904.01(c) citation that is the REASON for a rule, and the
    ("icg", "indocyanine green") example that teaches the acronym rule. A
    rewrite may shorten a sentence; it may not drop the fact the sentence was
    carrying.

    So: quoted strings, statute citations and multi-digit numbers. Crude on
    purpose — it reports, the human decides, and a false positive costs a
    glance while a false negative costs a rule.
    """
    import re as _r

    def facts(t: str) -> set[str]:
        out = set(_r.findall(r'"([^"\n]{2,40})"', t))
        out |= set(_r.findall(r"\bMPEP\s+[\d.]+(?:\([a-z]\))?", t))
        out |= set(_r.findall(r"\b\d+\s*CFR\s*[\d.()]+", t))
        out |= {n for n in _r.findall(r"\b\d[\d,]{2,9}\b", t) if "," in n or len(n) >= 3}
        return out
    return sorted(facts(before) - facts(after))[:20]


class PromptRevise(BaseModel):
    instruction: str


@router.post("/prompts/{name}/revise")
async def revise_prompt(name: str, req: PromptRevise):
    """Rewrite a prompt to the house style, without changing what it asks for.

    One turn, no state, nothing stored: the caller gets a proposal and decides.
    The model is shown four things and nothing else — the style guide, the
    prompt's contract, the current text verbatim, and what the editor asked
    for — because a rewrite that cannot see the contract is a rewrite that will
    quietly break it.

    It returns the text and the reason, and it does NOT save: a prompt that
    changed because somebody typed a sentence, with no one reading the result,
    is how a pipeline stops working in a way nobody can date.
    """
    from app import llm
    if not req.instruction.strip():
        raise HTTPException(400, "empty instruction")
    try:
        current, ver = _prompts.get(name)
    except KeyError:
        raise HTTPException(404, f"unknown prompt {name}")

    system = ("You rewrite prompt templates to a house style. You never change what the prompt "
              "asks the model to do, what it outputs, or the fields it names — only how it is "
              "written. If the instruction would change behaviour, say so in `rationale` and "
              "return the text unchanged.")
    contract = _prompts.contract(name) or "(no contract recorded for this prompt)"
    user = (f"# House style\n\n{_prompts.style_guide()}\n\n"
            f"# Contract for `{name}`\n\n{contract}\n\n"
            f"# Current template (version {ver})\n\n{current}\n\n"
            f"# What the editor asked for\n\n{req.instruction.strip()}\n\n"
            "# Output\n\n"
            'Return JSON only: {"text": "<the full rewritten template>", '
            '"rationale": "<=80 words on what you changed and what you deliberately left alone"}. '
            "The template uses str.format_map: keep every {placeholder} exactly as it is, and keep "
            "literal braces doubled.")
    model = os.environ.get("PROMPT_REVISE_MODEL", "gemini-3.1-pro-preview")
    try:
        raw = await llm.call_llm(system, user, model=model, max_tokens=16384)
    except Exception as exc:
        raise HTTPException(502, f"the model did not answer: {type(exc).__name__}: {exc}")
    import json as _json
    import re as _re
    fenced = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, _re.DOTALL)
    m = fenced or _re.search(r"\{.*\}", raw, _re.DOTALL)
    if not m:
        raise HTTPException(502, "the model did not return JSON")
    blob = m.group(1) if fenced else m.group()
    out = None
    for strict in (True, False):
        try:
            # strict=False tolerates raw newlines inside a JSON string, which is
            # how a model returns a multi-line template it forgot to escape —
            # the alternative is refusing a rewrite that is otherwise fine.
            out = _json.loads(blob, strict=strict)
            break
        except Exception:
            continue
    if out is None:
        raise HTTPException(502, "the model returned JSON this endpoint could not parse")
    text = str(out.get("text") or "")
    if not text.strip():
        raise HTTPException(502, "the model returned an empty template")
    # A rewrite that drops a placeholder breaks the prompt at render time, which
    # is a long way from here. Check it now and hand the check back.
    holes = set(_re.findall(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})", current))
    kept = set(_re.findall(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})", text))
    lost = sorted(holes - kept)
    specifics = _specifics_lost(current, text)
    return {"name": name, "from_version": ver, "text": text,
            "rationale": str(out.get("rationale") or ""),
            "model": model,
            "placeholders_lost": lost,
            "placeholders_added": sorted(kept - holes),
            "specifics_lost": specifics,
            "note": ("NOT SAFE TO SAVE: the rewrite dropped a placeholder the template renders with"
                     if lost else
                     "nothing was saved; PUT /prompts/{name} with this text to create a version"
                     + (f" — but read it first: {len(specifics)} specific(s) went missing" if specifics else ""))}


class PromptCurrent(BaseModel):
    version: int


@router.put("/prompts/{name}/current")
async def set_prompt_current(name: str, req: PromptCurrent):
    try:
        _prompts.set_current(name, req.version)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"name": name, "current": req.version}
