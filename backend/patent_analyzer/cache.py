"""Small key/value store for search results, BigQuery lookups and runtime
counters. Two backends behind one interface:

  SqliteKV     — evals / single process (file under eval_data or /tmp)
  FirestoreKV  — Cloud Run, shared across instances (collection per namespace)

Selected by CACHE_BACKEND=sqlite|firestore (default sqlite). Values are
JSON-serialisable dicts. `incr` is an atomic counter for quotas.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

_DEFAULT_SQLITE = Path(os.environ.get("CACHE_SQLITE_PATH",
                                      str(Path(__file__).parent.parent / "eval_data" / ".kv_cache.sqlite")))


class SqliteKV:
    def __init__(self, path: Path | str = _DEFAULT_SQLITE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v TEXT, ts REAL, PRIMARY KEY (ns, k))")
        self._conn.commit()

    def get(self, ns: str, key: str, max_age_days: float | None = None) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT v, ts FROM kv WHERE ns=? AND k=?", (ns, key)).fetchone()
        if not row:
            return None
        if max_age_days is not None and time.time() - row[1] > max_age_days * 86400:
            return None
        return json.loads(row[0])

    def put(self, ns: str, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv (ns, k, v, ts) VALUES (?, ?, ?, ?)",
                               (ns, key, json.dumps(value, ensure_ascii=False), time.time()))
            self._conn.commit()

    def incr(self, ns: str, key: str, field: str = "n", by: int = 1) -> int:
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE ns=? AND k=?", (ns, key)).fetchone()
            doc = json.loads(row[0]) if row else {}
            doc[field] = int(doc.get(field, 0)) + by
            self._conn.execute("INSERT OR REPLACE INTO kv (ns, k, v, ts) VALUES (?, ?, ?, ?)",
                               (ns, key, json.dumps(doc), time.time()))
            self._conn.commit()
            return doc[field]


class FirestoreKV:
    def __init__(self, prefix: str = "amie_kv"):
        import firebase_admin
        from firebase_admin import firestore
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        self._db = firestore.client()
        self._prefix = prefix

    def _ref(self, ns: str, key: str):
        return self._db.collection(f"{self._prefix}_{ns}").document(key[:1500])

    def get(self, ns: str, key: str, max_age_days: float | None = None) -> dict | None:
        snap = self._ref(ns, key).get()
        if not snap.exists:
            return None
        doc = snap.to_dict() or {}
        if max_age_days is not None and time.time() - float(doc.get("_ts", 0)) > max_age_days * 86400:
            return None
        return json.loads(doc.get("_v", "{}"))

    def put(self, ns: str, key: str, value: dict) -> None:
        self._ref(ns, key).set({"_v": json.dumps(value, ensure_ascii=False), "_ts": time.time()})

    def incr(self, ns: str, key: str, field: str = "n", by: int = 1) -> int:
        from firebase_admin import firestore
        ref = self._ref(ns, key)
        ref.set({field: firestore.Increment(by), "_ts": time.time()}, merge=True)
        return int((ref.get().to_dict() or {}).get(field, 0))


_kv = None


def kv():
    """Process-wide KV chosen by CACHE_BACKEND."""
    global _kv
    if _kv is None:
        backend = os.environ.get("CACHE_BACKEND", "sqlite").lower()
        _kv = FirestoreKV() if backend == "firestore" else SqliteKV()
    return _kv


def reset_for_tests(path: Path | str | None = None):
    global _kv
    _kv = SqliteKV(path) if path else None
