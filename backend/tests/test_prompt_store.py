import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.prompts as pr


@pytest.fixture
def env(monkeypatch):
    """`_want_gcs` reads the environment on every call, so nothing here needs to
    reload the module — and it must not: reloading clears `_DEFAULTS`, which
    every prompt registered at import time lives in, and the rest of the suite
    then fails with KeyError somewhere else entirely."""
    for k in ("PROMPT_STORE", "K_SERVICE"):
        monkeypatch.delenv(k, raising=False)
    saved_store, saved_note = pr._store, pr.store_note
    yield monkeypatch
    pr._store, pr.store_note = saved_store, saved_note


def test_cloud_run_stores_prompts_in_gcs_without_being_told_to(env):
    """A file store on Cloud Run writes inside the container: an edited prompt
    lives until the next deploy and no other instance ever sees it. K_SERVICE is
    injected by the runtime, so persistence cannot be lost by forgetting an
    environment variable (Harry, 2026-09-20)."""
    env.setenv("K_SERVICE", "patent-analyzer")
    assert pr._want_gcs() is True


def test_locally_it_is_a_file(env):
    assert pr._want_gcs() is False
    pr._store = None
    assert isinstance(pr.store(), pr._FileStore) and pr.store_note == "file"


def test_the_env_var_still_wins_both_ways(env):
    env.setenv("K_SERVICE", "x")
    env.setenv("PROMPT_STORE", "file")
    assert pr._want_gcs() is False
    env.setenv("PROMPT_STORE", "gcs")
    assert pr._want_gcs() is True


def test_an_unreachable_bucket_falls_back_but_says_so(env):
    """Silently dropping back to a file store is the original bug in a new hat,
    so the reason is recorded where the panel can read it."""
    env.setenv("PROMPT_STORE", "gcs")

    class Boom:
        def __init__(self):
            raise RuntimeError("no credentials")
    env.setattr(pr, "_GCSStore", Boom)
    pr._store = None
    assert isinstance(pr.store(), pr._FileStore)
    assert pr.store_note.startswith("file (GCS unavailable: RuntimeError: no credentials")


def test_a_saved_version_round_trips_through_whatever_store_is_active(env, tmp_path):
    env.setattr(pr, "PROMPT_DIR", tmp_path)
    pr._store = None
    pr._cache.pop("t.example", None)
    pr.register_default("t.example", "DEFAULT {x}")
    try:
        assert pr.get("t.example") == ("DEFAULT {x}", 0)
        v = pr.put("t.example", "EDITED {x}", by="harry")
        assert v == 1 and pr.get("t.example") == ("EDITED {x}", 1)
        assert (tmp_path / "t.example.json").exists()
        pr.set_current("t.example", 0)
        assert pr.get("t.example") == ("DEFAULT {x}", 0)
    finally:
        pr._DEFAULTS.pop("t.example", None)
        pr._cache.pop("t.example", None)
