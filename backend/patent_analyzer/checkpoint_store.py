"""Persisted checkpointer: InMemorySaver whose per-thread state is written
to a blob (local file or GCS) after every put / put_writes and loaded back
on first read in a new process.

LangGraph's docs: a checkpointer "should be a persistent checkpointer (e.g.
backed by a database)" for human-in-the-loop, and the BaseCheckpointSaver
contract is `.put`, `.put_writes`, `.get_tuple`, `.list`. Rather than
re-implement the blob/version bookkeeping of the in-memory saver, this
keeps its exact structures (storage / writes / blobs, all already serde
bytes) and persists them as one JSON document per thread — one job = one
thread = one blob of a few MB. Cloud Run runs one job at a time
(concurrency=1) so there is no concurrent writer per thread.
"""

from __future__ import annotations

import base64
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver


def _b(x: bytes) -> str:
    return base64.b64encode(x).decode()


def _ub(s: str) -> bytes:
    return base64.b64decode(s)


class _FileBlobs:
    def __init__(self, root: Path):
        self.root = Path(root)

    def read(self, thread_id: str) -> str | None:
        p = self.root / f"{thread_id}.json"
        return p.read_text() if p.exists() else None

    def write(self, thread_id: str, text: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f"{thread_id}.json.tmp"
        tmp.write_text(text)
        tmp.replace(self.root / f"{thread_id}.json")


class _GCSBlobs:
    def __init__(self, bucket: str, prefix: str):
        from google.cloud import storage
        self.bucket = storage.Client().bucket(bucket)
        self.prefix = prefix

    def _blob(self, thread_id: str):
        return self.bucket.blob(f"{self.prefix}{thread_id}/checkpoints.json")

    def read(self, thread_id: str) -> str | None:
        b = self._blob(thread_id)
        return b.download_as_text() if b.exists() else None

    def write(self, thread_id: str, text: str) -> None:
        self._blob(thread_id).upload_from_string(text, content_type="application/json")


class PersistedSaver(InMemorySaver):
    def __init__(self, blobs, **kw):
        super().__init__(**kw)
        self._blobs = blobs
        self._loaded: set[str] = set()

    # ── (de)serialisation of one thread ────────────────────────────────
    def _dump_thread(self, thread_id: str) -> str:
        storage = {ns: {cid: [[t, _b(v)], [mt, _b(mv)], parent] for cid, ((t, v), (mt, mv), parent) in cps.items()}
                   for ns, cps in self.storage.get(thread_id, {}).items()}
        writes = []
        for (tid, ns, cid), inner in self.writes.items():
            if tid != thread_id:
                continue
            for (task_id, idx), (t_id, ch, (t, v), path) in inner.items():
                writes.append([ns, cid, task_id, idx, t_id, ch, [t, _b(v)], path])
        blobs = [[ns, ch, ver, [t, _b(v)]] for (tid, ns, ch, ver), (t, v) in self.blobs.items() if tid == thread_id]
        return json.dumps({"v": 1, "thread_id": thread_id, "storage": storage, "writes": writes, "blobs": blobs})

    def _load_thread(self, thread_id: str) -> None:
        if thread_id in self._loaded:
            return
        self._loaded.add(thread_id)
        text = self._blobs.read(thread_id)
        if not text:
            return
        d = json.loads(text)
        for ns, cps in d["storage"].items():
            for cid, ((t, v), (mt, mv), parent) in cps.items():
                self.storage[thread_id][ns][cid] = ((t, _ub(v)), (mt, _ub(mv)), parent)
        for ns, cid, task_id, idx, t_id, ch, (t, v), path in d["writes"]:
            self.writes[(thread_id, ns, cid)][(task_id, idx)] = (t_id, ch, (t, _ub(v)), path)
        for ns, ch, ver, (t, v) in d["blobs"]:
            self.blobs[(thread_id, ns, ch, ver)] = (t, _ub(v))

    def _flush(self, thread_id: str) -> None:
        self._loaded.add(thread_id)
        self._blobs.write(thread_id, self._dump_thread(thread_id))

    # ── BaseCheckpointSaver ─────────────────────────────────────────────
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self._load_thread(config["configurable"]["thread_id"])
        return super().get_tuple(config)

    def list(self, config, *, filter=None, before=None, limit=None):
        if config:
            self._load_thread(config["configurable"]["thread_id"])
        return super().list(config, filter=filter, before=before, limit=limit)

    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions) -> RunnableConfig:
        tid = config["configurable"]["thread_id"]
        self._load_thread(tid)
        out = super().put(config, checkpoint, metadata, new_versions)
        self._flush(tid)
        return out

    def put_writes(self, config: RunnableConfig, writes, task_id: str, task_path: str = "") -> None:
        tid = config["configurable"]["thread_id"]
        self._load_thread(tid)
        super().put_writes(config, writes, task_id, task_path)
        self._flush(tid)

    async def aget_tuple(self, config):
        return self.get_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        for x in self.list(config, filter=filter, before=before, limit=limit):
            yield x

    async def aput(self, config, checkpoint, metadata, new_versions):
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return self.put_writes(config, writes, task_id, task_path)


def make_checkpointer(backend: str | None = None, root: str | Path | None = None):
    """CHECKPOINT_BACKEND = memory (default) | file | gcs."""
    backend = backend or os.environ.get("CHECKPOINT_BACKEND", "memory")
    if backend == "file":
        return PersistedSaver(_FileBlobs(Path(root or os.environ.get("CHECKPOINT_DIR", "/tmp/amie_checkpoints"))))
    if backend == "gcs":
        return PersistedSaver(_GCSBlobs(os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1"),
                                        os.environ.get("GCS_PREFIX", "patent-analyzer/jobs/")))
    return InMemorySaver()
