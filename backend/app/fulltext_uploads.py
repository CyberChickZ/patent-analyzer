"""The missing-full-text list, its upload slots, and the re-read they trigger.

GET  /api/jobs/{id}/fulltext-gaps                  what the report never read, and why
POST /api/jobs/{id}/fulltext/{ref_id}/upload       multipart PDF for one reference
POST /api/jobs/{id}/fulltext/{ref_id}/from-gcs     the same, already in the bucket
DELETE /api/jobs/{id}/fulltext/{ref_id}            drop a staged upload before the re-read
POST /api/jobs/{id}/rerun-evidence                 deep-read the uploaded ones, re-adjudicate

Why an upload slot at all: the examiner-cited papers are the side we cannot
reach. On the E4/h1h gold, 2 of 17 resolvable NPL gold reached the pool against
.607 on the patent side, and the papers that do reach Phase 4 mostly arrive with
an abstract. Paywalls are a large part of that, and the automatic answer is
closed off on purpose — see patent_analyzer/fulltext.py for the policy text.
A person can still open the page and save the PDF; this is where they put it.

The two upload routes mirror the two that already exist for the job's own input
(`/analyze` multipart and `/upload-url` → GCS → `/analyze-gcs`), rather than
inventing a third shape. Uploaded PDFs land in the same DOI-keyed GCS cache the
open-access downloads use (`gs://<bucket>/fulltext/<doi-slug>.pdf`), so a paper
bought by hand once is held for every later job that finds it again.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import optional_auth
from patent_analyzer import fulltext as ft
from patent_analyzer import fulltext_gap as gap

router = APIRouter()

MAX_PDF_BYTES = 64 * 1024 * 1024


def _job_or_404(job_id: str) -> dict:
    from app.main import _get_job

    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


def _results_of(job: dict) -> dict:
    """The job's results.json: local disk first, then the bucket — the same
    order `/results/{job_id}` uses, for the same reason (Cloud Run scales out,
    and the instance that ran the job is usually not this one)."""
    import json

    from app.main import GCS_BUCKET, GCS_PREFIX, _get_gcs

    job_id = job.get("id", "")
    path = Path(job.get("output_dir", f"/tmp/outputs/{job_id}")) / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        blob = _get_gcs().bucket(GCS_BUCKET).blob(f"{GCS_PREFIX}{job_id}/results.json")
        if blob.exists():
            return json.loads(blob.download_as_bytes())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"results.json could not be read: {type(exc).__name__}: {exc}")
    raise HTTPException(404, "This job has no results.json yet")


def _rows(job: dict) -> tuple[dict, list[dict]]:
    results = _results_of(job)
    return results, gap.gap_rows(results, job.get("fulltext_uploads") or {})


@router.get("/jobs/{job_id}/fulltext-gaps")
async def get_fulltext_gaps(job_id: str, user: dict | None = Depends(optional_auth)):
    job = _job_or_404(job_id)
    results, rows = _rows(job)
    return {
        "job_id": job_id,
        "status": job.get("status", ""),
        "rows": rows,
        "summary": gap.gap_summary(results, rows),
        "policy_note": gap.POLICY_NOTE,
        "policy_url": gap.POLICY_URL,
        "rerun_history": job.get("fulltext_reruns") or [],
    }


def _accept(job: dict, ref_id: str, data: bytes, filename: str) -> dict:
    """Store one uploaded PDF against one reference. Returns the upload record."""
    from app.main import _save_job

    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, f"PDF is {len(data) / 1e6:.1f} MB; the limit is {MAX_PDF_BYTES / 1e6:.0f} MB")
    if not data[:5].startswith(b"%PDF"):
        # The deep read hands the bytes to Gemini as a PDF part; anything else
        # fails there instead of here, with a far less useful message.
        raise HTTPException(400, "That file does not start with %PDF — only PDFs can be read here")

    _, rows = _rows(job)
    row = next((r for r in rows if r["ref_id"] == ref_id), None)
    if row is None:
        raise HTTPException(404, f"No reference {ref_id!r} is missing its full text on this job")

    job_dir = Path(job.get("output_dir", f"/tmp/outputs/{job.get('id', '')}"))
    job_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]", "_", filename or "upload.pdf")[:80] or "upload.pdf"
    dest = job_dir / f"uploaded_{ref_id[:60]}.pdf"
    dest.write_bytes(data)

    doi = row.get("doi") or ""
    cached = bool(doi) and ft.cache_put(doi, data)
    record = {
        "ref_id": ref_id,
        "filename": safe,
        "bytes": len(data),
        "path": str(dest),
        "doi": doi,
        "gcs_uri": ft.cache_uri(doi) if cached else "",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "uploaded_by": job.get("submitted_by", ""),
        "reread": False,          # flipped by the evidence re-run once this PDF has been read
    }
    job.setdefault("fulltext_uploads", {})[ref_id] = record
    _save_job(job)
    return record


@router.post("/jobs/{job_id}/fulltext/{ref_id}/upload")
async def upload_fulltext(job_id: str, ref_id: str, file: UploadFile = File(...),
                          user: dict | None = Depends(optional_auth)):
    """Multipart, like `/analyze`. Cloud Run caps a request body at 32 MB;
    anything larger takes the signed-URL road and lands on `/from-gcs`."""
    job = _job_or_404(job_id)
    record = _accept(job, ref_id, await file.read(), file.filename or "upload.pdf")
    return {"job_id": job_id, "upload": record,
            "pending_reread": sum(1 for u in (job.get("fulltext_uploads") or {}).values()
                                  if not u.get("reread"))}


@router.post("/jobs/{job_id}/fulltext/{ref_id}/from-gcs")
async def upload_fulltext_from_gcs(job_id: str, ref_id: str, payload: dict,
                                   user: dict | None = Depends(optional_auth)):
    """The large-file road: the client PUT the PDF to a signed URL from
    `/upload-url`, and names the object here."""
    from app.main import _get_gcs

    job = _job_or_404(job_id)
    uri = str(payload.get("gcs_uri") or "")
    if not uri.startswith("gs://"):
        raise HTTPException(400, "Missing/invalid gcs_uri")
    bucket_name, _, object_key = uri[len("gs://"):].partition("/")
    try:
        data = _get_gcs().bucket(bucket_name).blob(object_key).download_as_bytes()
    except Exception as exc:
        raise HTTPException(502, f"Could not read {uri}: {type(exc).__name__}: {exc}")
    record = _accept(job, ref_id, data, payload.get("filename") or object_key.rsplit("/", 1)[-1])
    return {"job_id": job_id, "upload": record,
            "pending_reread": sum(1 for u in (job.get("fulltext_uploads") or {}).values()
                                  if not u.get("reread"))}


@router.delete("/jobs/{job_id}/fulltext/{ref_id}")
async def drop_fulltext_upload(job_id: str, ref_id: str, user: dict | None = Depends(optional_auth)):
    """Unstage an upload. Only one that has not been read yet — a PDF the
    report already cites cannot be taken back by deleting a record."""
    from app.main import _save_job

    job = _job_or_404(job_id)
    ups = job.get("fulltext_uploads") or {}
    rec = ups.get(ref_id)
    if not rec:
        raise HTTPException(404, "Nothing uploaded for that reference")
    if rec.get("reread"):
        raise HTTPException(409, "That PDF has already been read into the report; re-run the job to undo it")
    ups.pop(ref_id, None)
    try:
        Path(rec.get("path", "")).unlink(missing_ok=True)
    except Exception:
        pass
    _save_job(job)
    return {"job_id": job_id, "dropped": ref_id}


@router.post("/jobs/{job_id}/rerun-evidence")
async def rerun_evidence(job_id: str, payload: dict | None = None,
                         user: dict | None = Depends(optional_auth)):
    """Queue a deep read of the uploaded PDFs alone, then re-adjudicate and
    re-report.

    Not `POST /jobs/{id}/resume {"action":"rerun_phase"}`, deliberately. That
    route replays the graph from the checkpoint before a phase, which means the
    whole Phase 4 fan-out: 25 documents, 25 Vertex calls, most of them re-reading
    text that has not changed — and it needs a live checkpoint and a job that is
    paused or failed, neither of which a finished job has. What is reused is the
    machinery around it: the same serial job queue, the same job record, the same
    event feed, so the page that watches a run watches this too.
    """
    from app.main import _enqueue_job, _save_job

    job = _job_or_404(job_id)
    if job.get("status") in ("running", "queued"):
        raise HTTPException(409, f"Job is {job['status']}; wait for it to finish")
    ups = job.get("fulltext_uploads") or {}
    asked = [r for r in ((payload or {}).get("refs") or []) if r in ups]
    # No explicit list = everything uploaded but not yet read. An explicit list
    # may name a PDF that was read already, which is how a reader re-reads one
    # after a checklist edit.
    pending = asked or [r for r, u in ups.items() if not u.get("reread")]
    if not pending:
        raise HTTPException(400, "No uploaded full text is waiting to be read")

    job["_rerun_evidence"] = pending
    job["status"] = "queued"
    job["phase"] = "phase4"
    _save_job(job)
    _enqueue_job(job_id)
    return {"job_id": job_id, "status": "queued", "refs": pending}
