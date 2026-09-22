"""A daily spend ceiling the application enforces on itself.

Harry has no access to the billing account, so a GCP budget alert has to come
from the boss. In the meantime the only place that can stop a runaway is the
application, and it already knows what it is spending: `metering` prices every
LLM call from the token counts and every BigQuery job from the bytes billed.

So the number here is an ESTIMATE FROM LIST PRICES, never a bill. It is
deliberately the same arithmetic the per-job cost card shows, so the two can
never disagree.

Why GCS and not the KV: `cache.kv()` on Cloud Run is SqliteKV inside the
container (CACHE_BACKEND is unset, and Firestore does not exist in this
project — checked 2026-09-20). A per-instance counter cannot enforce an
account-wide cap: two instances would each allow $30. One object per day in
GCS is shared, survives a redeploy, and is cheap to read.

Concurrent instances read-modify-write the same object. The write uses a
generation precondition and retries, so two jobs finishing at once cannot lose
one of the two amounts.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

# Read per call, never at import: a module-level os.environ.get is invisible to
# anything that sets the variable afterwards, which in practice means every test
# and every eval script. (Caught here by three tests failing into each other's
# day file.)


def _bucket_name() -> str:
    return os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1")


def _prefix() -> str:
    return os.environ.get("SPEND_PREFIX", "spend/")


def _local_dir() -> str:
    return os.environ.get("SPEND_LOCAL_DIR", "")


def cap_usd() -> float:
    """Read per call, not at import: the eval harness raises it for a run."""
    try:
        return float(os.environ.get("DAILY_SPEND_CAP_USD", "30"))
    except ValueError:
        return 30.0


def _today(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


def resets_at(now: datetime | None = None) -> datetime:
    """Midnight UTC, which is also when the date key rolls over. Said in the
    refusal message, because "try later" is not an answer."""
    now = now or datetime.now(timezone.utc)
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1)


def _blank(day: str) -> dict:
    return {"date": day, "usd": 0.0, "by_kind": {}, "by_job": {}, "updated_at": ""}


# ── storage ────────────────────────────────────────────────────────────────

def _local_path(day: str):
    from pathlib import Path
    base = Path(_local_dir()) if _local_dir() else Path(__file__).parent.parent / "eval_data" / "spend"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{day}.json"


def _use_gcs() -> bool:
    """SPEND_STORE=local forces the file store. Tests and eval scripts set it:
    a test that writes a fake amount into the shared day object makes the
    production panel lie, which happened once here already (2026-09-20) and had
    to be cleaned out of the bucket by hand."""
    v = os.environ.get("SPEND_STORE", "").strip().lower()
    if v:
        return v == "gcs"
    return not bool(_local_dir())


def _bucket():
    from google.cloud import storage
    return storage.Client().bucket(_bucket_name())


def _load(day: str) -> tuple[dict, int | None, str]:
    """(doc, generation, error). generation is None for the local store."""
    if not _use_gcs():
        p = _local_path(day)
        try:
            return (json.loads(p.read_text()) if p.exists() else _blank(day)), None, ""
        except Exception as exc:
            return _blank(day), None, f"{type(exc).__name__}: {exc}"[:160]
    try:
        blob = _bucket().blob(f"{_prefix()}{day}.json")
        if not blob.exists():
            return _blank(day), 0, ""
        doc = json.loads(blob.download_as_text())
        return doc, blob.generation, ""
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:160]
    p = _local_path(day)
    try:
        return (json.loads(p.read_text()) if p.exists() else _blank(day)), None, err
    except Exception:
        return _blank(day), None, err


def _save(day: str, doc: dict, generation: int | None) -> bool:
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    if generation is not None:
        try:
            _bucket().blob(f"{_prefix()}{day}.json").upload_from_string(
                json.dumps(doc, ensure_ascii=False, indent=1), content_type="application/json",
                if_generation_match=generation)
            return True
        except Exception:
            return False
    try:
        _local_path(day).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        return True
    except Exception:
        return False


# ── the two things callers do ──────────────────────────────────────────────

def record_job_total(job_id: str, usd: float, kind: str = "job", day: str | None = None) -> dict:
    """Set this job's running total for the day, and add only the difference.

    Callers hand over the job's total so far, not a delta, so a phase that is
    reported twice cannot double-count — which is how a spend counter quietly
    becomes fiction.
    """
    day = day or _today()
    usd = round(float(usd or 0), 6)
    for attempt in range(5):
        doc, gen, _ = _load(day)
        prev = float((doc.get("by_job") or {}).get(job_id, 0.0))
        delta = usd - prev
        if delta <= 0 and job_id in (doc.get("by_job") or {}):
            return doc
        doc.setdefault("by_job", {})[job_id] = usd
        doc["usd"] = round(float(doc.get("usd", 0.0)) + max(0.0, delta), 6)
        doc.setdefault("by_kind", {})[kind] = round(
            float((doc.get("by_kind") or {}).get(kind, 0.0)) + max(0.0, delta), 6)
        if _save(day, doc, gen):
            return doc
        time.sleep(0.15 * (attempt + 1))
    return doc


def today(day: str | None = None) -> dict:
    doc, _, err = _load(day or _today())
    doc["cap_usd"] = cap_usd()
    doc["over_cap"] = float(doc.get("usd", 0)) >= doc["cap_usd"]
    doc["resets_at"] = resets_at().isoformat()
    doc["error"] = err
    return doc


def month_usd(prefix: str | None = None) -> tuple[float, str]:
    """(total, error). Sums the day objects of the current month."""
    pre = prefix or datetime.now(timezone.utc).strftime("%Y-%m")
    if not _use_gcs():
        from pathlib import Path
        base = _local_path("x").parent
        return round(sum(float(json.loads(Path(f).read_text()).get("usd", 0.0))
                         for f in base.glob(f"{pre}-*.json")), 4), ""
    try:
        from google.cloud import storage
        client = storage.Client()
        total = 0.0
        for b in client.list_blobs(_bucket_name(), prefix=f"{_prefix()}{pre}"):
            total += float(json.loads(b.download_as_text()).get("usd", 0.0))
        return round(total, 4), ""
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:160]
    try:
        from pathlib import Path
        base = _local_path("x").parent
        total = sum(float(json.loads(Path(p).read_text()).get("usd", 0.0))
                    for p in base.glob(f"{pre}-*.json"))
        return round(total, 4), err
    except Exception:
        return 0.0, err


def refusal() -> dict | None:
    """None when there is room, otherwise the whole reason, in words a person
    can act on: how much has gone, what the ceiling is, and when it lifts."""
    d = today()
    if not d["over_cap"]:
        return None
    left = resets_at() - datetime.now(timezone.utc)
    hours = max(0, round(left.total_seconds() / 3600, 1))
    return {
        "spent_usd": round(float(d.get("usd", 0)), 2),
        "cap_usd": d["cap_usd"],
        "resets_at": d["resets_at"],
        "resets_in_hours": hours,
        "detail": (f"Today's estimated spend is ${float(d.get('usd', 0)):.2f}, which is at or over "
                   f"the ${d['cap_usd']:.2f} daily ceiling this deployment sets for itself. "
                   f"Jobs already running are left alone; new ones start again at "
                   f"{d['resets_at'][:16]}Z, in about {hours} hours. "
                   f"The ceiling is DAILY_SPEND_CAP_USD, and the figure is an estimate from list "
                   f"prices — the LLM tokens and BigQuery bytes this deployment metered — not a bill."),
    }
