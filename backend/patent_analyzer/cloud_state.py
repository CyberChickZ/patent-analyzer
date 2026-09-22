"""Every counter that has to survive a request, in one place, in the cloud.

Harry, 2026-09-20: "不能有 local counter，必须云端".

The rule this module exists to enforce:

  A counter or a piece of state that outlives one request lives in GCS.
  A container-local store may hold a CACHE — something whose loss costs time
  and nothing else.

Why it is not the KV: `cache.kv()` on Cloud Run is SqliteKV inside the
container (CACHE_BACKEND is unset and Firestore does not exist in this
project). The service runs up to three instances, so every counter on it was
three counters, each certain it was the only one, and a deploy reset all
three. That is how a SerpAPI counter read ~55 while the account said 272.

One JSON object per key under `state/`. Writes carry the object's generation
as a precondition and retry on conflict, so two instances incrementing at the
same moment cannot lose one of the increments — which a read-modify-write
without a precondition silently does.

Local development with no bucket falls back to a file, through the same
functions. There is one code path; the backend underneath it changes.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

ATTEMPTS = 6


def bucket_name() -> str:
    return os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1")


def prefix() -> str:
    return os.environ.get("CLOUD_STATE_PREFIX", "state/")


def local_dir() -> str:
    return os.environ.get("CLOUD_STATE_LOCAL_DIR", "")


def use_gcs() -> bool:
    """CLOUD_STATE=local forces the file backend. Tests and eval scripts set
    it: a test that increments the shared counter makes the panel lie."""
    v = os.environ.get("CLOUD_STATE", "").strip().lower()
    if v:
        return v == "gcs"
    return not local_dir()


def _path(key: str) -> Path:
    base = Path(local_dir()) if local_dir() else Path(__file__).parent.parent / "eval_data" / "state"
    base.mkdir(parents=True, exist_ok=True)
    return base / (key.replace("/", "_").replace(":", "_") + ".json")


def _blob(key: str):
    from google.cloud import storage
    return storage.Client().bucket(bucket_name()).blob(f"{prefix()}{key}.json")


def read(key: str) -> tuple[dict, int | None, str]:
    """(document, generation, error). generation is None on the file backend,
    and 0 means "there is no object yet" — which is the precondition that
    creates it exactly once."""
    if not use_gcs():
        p = _path(key)
        try:
            return (json.loads(p.read_text()) if p.exists() else {}), None, ""
        except Exception as exc:
            return {}, None, f"{type(exc).__name__}: {exc}"[:160]
    try:
        b = _blob(key)
        if not b.exists():
            return {}, 0, ""
        return json.loads(b.download_as_text()), b.generation, ""
    except Exception as exc:
        return {}, None, f"{type(exc).__name__}: {exc}"[:160]


def write(key: str, doc: dict, generation: int | None) -> bool:
    if generation is None:
        try:
            _path(key).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
            return True
        except Exception:
            return False
    try:
        _blob(key).upload_from_string(json.dumps(doc, ensure_ascii=False, indent=1),
                                      content_type="application/json",
                                      if_generation_match=generation)
        return True
    except Exception:
        return False


def update(key: str, fn, attempts: int = ATTEMPTS) -> dict:
    """Read, apply `fn(doc)` in place, write with a precondition; retry on a
    lost race with a little jitter so two instances do not lock step."""
    doc: dict = {}
    for i in range(attempts):
        doc, gen, _ = read(key)
        fn(doc)
        if write(key, doc, gen):
            return doc
        time.sleep(0.08 * (i + 1) + random.random() * 0.05)
    return doc


def get(key: str, field: str = "n", default=0):
    return read(key)[0].get(field, default)


def incr(key: str, field: str = "n", by: int = 1) -> int:
    def _f(doc):
        doc[field] = int(doc.get(field, 0)) + by
    return int(update(key, _f).get(field, 0))


def put(key: str, doc: dict) -> dict:
    def _f(d):
        d.clear()
        d.update(doc)
    return update(key, _f)
