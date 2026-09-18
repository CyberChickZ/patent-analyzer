"""Prompt registry: versioned prompt templates that nodes fetch at run time.

A prompt's version 0 is the template in code (registered at import). Saved
versions live in backend/prompts/<name>.json locally and under
gs://<GCS_BUCKET>/prompts/<name>.json on Cloud Run, which is the default there
(see `_want_gcs`) rather than something a deploy has to remember to switch on.
One object per prompt holds every version and which one is current; nothing
reads a prompt without going through `store()`, so there is no second copy to
keep in step.

A job can pin a version or supply inline text through `prompt_overrides` (set
by the runner in a ContextVar so nodes and subgraph nodes see it without
plumbing).
Templates are str.format_map templates: literal braces are doubled.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

PROMPT_DIR = Path(__file__).parent.parent / "prompts"
_DEFAULTS: dict[str, str] = {}
_overrides: contextvars.ContextVar[dict | None] = contextvars.ContextVar("prompt_overrides", default=None)
_used: contextvars.ContextVar[dict | None] = contextvars.ContextVar("prompt_versions_used", default=None)
_cache: dict[str, tuple[float, dict]] = {}
CACHE_S = 60.0


#: What each prompt is for, in plain English: what it does, what it must output,
#: who reads that output, and what cannot be relaxed. Written next to the
#: template so an editor — or the revise assistant — can tell a rewrite from a
#: change. See app/prompt_style.md §5.
_CONTRACTS: dict[str, str] = {}


def register_default(name: str, template: str, contract: str = "") -> str:
    _DEFAULTS[name] = template
    if contract:
        _CONTRACTS[name] = contract.strip()
    return template


def contract(name: str) -> str:
    return _CONTRACTS.get(name, "")


def style_guide() -> str:
    """The house style, as the model is shown it. Same file the humans read."""
    from pathlib import Path
    p = Path(__file__).parent / "prompt_style.md"
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def names() -> list[str]:
    return sorted(_DEFAULTS)


class _FileStore:
    def load(self, name: str) -> dict:
        p = PROMPT_DIR / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else {"current": 0, "versions": []}

    def save(self, name: str, doc: dict) -> None:
        PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        (PROMPT_DIR / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))


class _GCSStore:
    def __init__(self):
        from google.cloud import storage
        self.bucket = storage.Client().bucket(os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1"))

    def load(self, name: str) -> dict:
        b = self.bucket.blob(f"prompts/{name}.json")
        return json.loads(b.download_as_text()) if b.exists() else {"current": 0, "versions": []}

    def save(self, name: str, doc: dict) -> None:
        self.bucket.blob(f"prompts/{name}.json").upload_from_string(json.dumps(doc, ensure_ascii=False, indent=1),
                                                                    content_type="application/json")


_store = None
store_note = ""


def _want_gcs() -> bool:
    """Where saved prompts live, and why the default is not a file.

    A file store on Cloud Run writes inside the container: an edited prompt
    survives until the next deploy and is invisible to every other instance
    while it lasts. That is not persistence, and nothing reported it — the
    prompt simply went back to the built-in default (Harry, 2026-09-20).

    PROMPT_STORE still decides when it is set. When it is not, Cloud Run is
    detected by K_SERVICE, which the runtime always injects (Cloud Run docs,
    "Container runtime contract"), so a deployment cannot lose its prompts by
    forgetting an environment variable.
    """
    v = os.environ.get("PROMPT_STORE", "").strip().lower()
    if v:
        return v == "gcs"
    return bool(os.environ.get("K_SERVICE"))


def store():
    """The store, with the reason for it. A GCS store that cannot be built
    falls back to the file store rather than taking the process down — but it
    says so in `store_note`, because a silent fallback here is the original bug
    wearing a different hat."""
    global _store, store_note
    if _store is None:
        if _want_gcs():
            try:
                _store = _GCSStore()
                store_note = "gcs"
            except Exception as exc:
                _store = _FileStore()
                store_note = f"file (GCS unavailable: {type(exc).__name__}: {exc})"[:200]
        else:
            _store = _FileStore()
            store_note = "file"
    return _store


def _doc(name: str) -> dict:
    hit = _cache.get(name)
    if hit and time.time() - hit[0] < CACHE_S:
        return hit[1]
    d = store().load(name)
    _cache[name] = (time.time(), d)
    return d


def describe(name: str) -> dict:
    if name not in _DEFAULTS:
        raise KeyError(name)
    d = _doc(name)
    return {"name": name, "current": d.get("current", 0), "default": _DEFAULTS[name],
            "contract": contract(name),
            "versions": [{k: v for k, v in ver.items()} for ver in d.get("versions", [])]}


def put(name: str, text: str, by: str = "", make_current: bool = True) -> int:
    """Save a new version; returns its number (defaults are version 0)."""
    if name not in _DEFAULTS:
        raise KeyError(name)
    d = store().load(name)
    v = max([ver["v"] for ver in d.get("versions", [])] + [0]) + 1
    d.setdefault("versions", []).append({"v": v, "text": text, "by": by, "ts": datetime.now(timezone.utc).isoformat()})
    if make_current:
        d["current"] = v
    store().save(name, d)
    _cache.pop(name, None)
    return v


def set_current(name: str, version: int) -> None:
    d = store().load(name)
    if version != 0 and version not in {ver["v"] for ver in d.get("versions", [])}:
        raise KeyError(f"{name} has no version {version}")
    d["current"] = version
    store().save(name, d)
    _cache.pop(name, None)


def set_overrides(overrides: dict | None) -> None:
    """Runner: per-job {name: version | text}. Resets the used-versions record."""
    _overrides.set(dict(overrides or {}))
    _used.set({})


def used_versions() -> dict[str, int | str]:
    return dict(_used.get() or {})


def get(name: str) -> tuple[str, int | str]:
    """(template, version): job override (int = saved version, str = inline
    text → 'inline'), else the store's current version, else the default."""
    if name not in _DEFAULTS:
        raise KeyError(name)
    ov = (_overrides.get() or {}).get(name)
    if isinstance(ov, str) and ov.strip():
        text, ver = ov, "inline"
    else:
        d = _doc(name)
        want = ov if isinstance(ov, int) else d.get("current", 0)
        text, ver = _DEFAULTS[name], 0
        if want:
            for v in d.get("versions", []):
                if v["v"] == want:
                    text, ver = v["text"], want
                    break
    used = _used.get()
    if used is not None:
        used[name] = ver
    return text, ver


def render(prompt_name: str, /, **fields) -> str:
    text, _ = get(prompt_name)
    return text.format_map(_Safe(fields))


class _Safe(dict):
    def __missing__(self, key):
        return "{" + key + "}"
