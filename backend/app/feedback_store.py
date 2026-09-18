"""Everything anybody said about a run, in one place.

Four kinds, and the distinction that matters is who wrote them:

  prompt_edit    written by the system when somebody saves a prompt version
  reviewer_edit  written by the system when a HITL gate edit is accepted
  comment        typed by a person, against a place in the report
  rating         typed by a person, a thumb and optionally a sentence

The first two carry `auto: true`. They are a record of a change, not an
opinion about one, and the page keeps them in their own column so "the system
noticed this" never reads as "somebody complained about this".

Storage is GCS, one object per entry under feedback/<yyyy-mm>/<id>.json — not
the KV, which on Cloud Run is a SQLite file inside the container and would
give each instance its own private feedback list. Local development with no
bucket falls back to files through the same functions.

Listing reads the month objects newest first. That is fine at this volume and
honest about its limit: it is a scan, so the page takes a page size and the
module says so rather than pretending to be a database.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

KINDS = ("prompt_edit", "reviewer_edit", "comment", "rating")
AUTO_KINDS = ("prompt_edit", "reviewer_edit")
STATUSES = ("open", "addressed")
PAGE = 100


def _bucket_name() -> str:
    return os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1")


def _prefix() -> str:
    return os.environ.get("FEEDBACK_PREFIX", "feedback/")


def _local_dir() -> str:
    return os.environ.get("FEEDBACK_LOCAL_DIR", "")


def _use_gcs() -> bool:
    v = os.environ.get("FEEDBACK_STORE", "").strip().lower()
    if v:
        return v == "gcs"
    return not _local_dir()


def _dir():
    from pathlib import Path
    base = Path(_local_dir()) if _local_dir() else \
        Path(__file__).parent.parent / "eval_data" / "feedback"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _month(ts: str) -> str:
    return (ts or "")[:7] or datetime.now(timezone.utc).strftime("%Y-%m")


def new_entry(payload: dict, by: str) -> dict:
    """One entry, with every field the page needs and nothing invented.

    `by` comes from the token, never from the payload: a comment that says who
    wrote it, where the writer chose the name, is not a record of anything.
    """
    kind = str(payload.get("kind") or "comment")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": uuid.uuid4().hex[:12],
        "ts": now,
        "by": by,
        "auto": kind in AUTO_KINDS,
        "kind": kind,
        "job_id": str(payload.get("job_id") or ""),
        "job_title": str(payload.get("job_title") or ""),
        # where in the UI this is about: {"tab": "evidence", "anchor": "US123B2"}
        "target": payload.get("target") or {},
        "text": str(payload.get("text") or "")[:8000],
        "prompt_name": str(payload.get("prompt_name") or ""),
        "prompt_version": payload.get("prompt_version"),
        "instruction": str(payload.get("instruction") or "")[:4000],
        "diff_summary": str(payload.get("diff_summary") or "")[:2000],
        "rating": payload.get("rating"),
        "status": "open",
        "addressed_by": None,
        "addressed_at": "",
    }


def _blob(month: str, eid: str):
    from google.cloud import storage
    return storage.Client().bucket(_bucket_name()).blob(f"{_prefix()}{month}/{eid}.json")


def save(entry: dict) -> dict:
    month = _month(entry.get("ts", ""))
    body = json.dumps(entry, ensure_ascii=False, indent=1)
    if _use_gcs():
        _blob(month, entry["id"]).upload_from_string(body, content_type="application/json")
    else:
        d = _dir() / month
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{entry['id']}.json").write_text(body)
    return entry


def get(eid: str) -> dict | None:
    for e in _iter_all():
        if e.get("id") == eid:
            return e
    return None


def _iter_all():
    if _use_gcs():
        from google.cloud import storage
        client = storage.Client()
        for b in client.list_blobs(_bucket_name(), prefix=_prefix()):
            if not b.name.endswith(".json"):
                continue
            try:
                yield json.loads(b.download_as_text())
            except Exception:
                continue
        return
    base = _dir()
    for p in sorted(base.rglob("*.json")):
        try:
            yield json.loads(p.read_text())
        except Exception:
            continue


def listing(job: str = "", kind: str = "", status: str = "",
            prompt_name: str = "", limit: int = PAGE, offset: int = 0) -> dict:
    rows = [e for e in _iter_all()
            if (not job or e.get("job_id") == job)
            and (not kind or e.get("kind") == kind)
            and (not status or e.get("status") == status)
            and (not prompt_name or e.get("prompt_name") == prompt_name)]
    rows.sort(key=lambda e: e.get("ts", ""), reverse=True)
    total = len(rows)
    page = rows[offset:offset + max(1, min(int(limit or PAGE), PAGE))]
    return {"total": total, "offset": offset, "limit": limit, "entries": page,
            "note": ("the store is a scan over one object per entry, not a database; "
                     "at this volume that is fine and this field is here so nobody "
                     "is surprised when it stops being fine")}


def patch(eid: str, changes: dict, by: str) -> dict | None:
    e = get(eid)
    if not e:
        return None
    status = changes.get("status")
    if status in STATUSES:
        e["status"] = status
        e["addressed_at"] = datetime.now(timezone.utc).isoformat() if status == "addressed" else ""
        e["addressed_by"] = changes.get("addressed_by") if status == "addressed" else None
    if "text" in changes and not e.get("auto"):
        e["text"] = str(changes["text"])[:8000]
    e["last_edited_by"] = by
    return save(e)


def record_prompt_edit(name: str, version: int, by: str, instruction: str = "",
                       diff_summary: str = "") -> dict | None:
    """Called when a prompt version is saved. Never raises into the caller:
    losing the audit line must not lose the prompt."""
    try:
        return save(new_entry({"kind": "prompt_edit", "prompt_name": name,
                               "prompt_version": version, "instruction": instruction,
                               "diff_summary": diff_summary,
                               "text": f"saved {name} v{version}"}, by))
    except Exception as exc:
        print(f"[feedback] prompt_edit not recorded: {exc}", flush=True)
        return None


def record_reviewer_edits(job_id: str, edits: list, by: str) -> list:
    """One entry per accepted gate edit, so the timeline shows what a reviewer
    changed next to what the prompts did."""
    out = []
    for ed in edits or []:
        try:
            field = ed.get("field") or ed.get("path") or ""
            before = str(ed.get("before") or "")[:400]
            after = str(ed.get("after") or "")[:400]
            out.append(save(new_entry({
                "kind": "reviewer_edit", "job_id": job_id,
                "target": {"tab": "candidates", "anchor": field},
                "text": f"{field}: {before} → {after}" if field else f"{before} → {after}",
            }, by)))
        except Exception as exc:
            print(f"[feedback] reviewer_edit not recorded: {exc}", flush=True)
    return out
