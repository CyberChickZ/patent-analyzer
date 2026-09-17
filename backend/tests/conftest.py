"""Suite-wide guards.

The rule here is narrow: no test may touch shared cloud state. Not for speed —
though a SerpAPI probe timing out on a fake key did cost 15 s a go — but
because a test that writes into a shared object makes production lie. The
spend ledger is one JSON object per day in the bucket that every instance adds
to; a test that recorded $2.81 against a job that never ran put that number on
the real quota panel, and it had to be deleted from the bucket by hand
(2026-09-20).
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _never_shared_spend_state(tmp_path_factory, monkeypatch):
    """Every test writes its spend ledger to its own directory."""
    if os.environ.get("SPEND_STORE") is None:
        monkeypatch.setenv("SPEND_STORE", "local")
    if os.environ.get("SPEND_LOCAL_DIR") is None:
        monkeypatch.setenv("SPEND_LOCAL_DIR", str(tmp_path_factory.mktemp("spend")))


@pytest.fixture(autouse=True)
def _no_accidental_urlopen(monkeypatch):
    """`_search` now reconciles with SerpAPI's /account before spending a key,
    which with the suite's fake keys is a real request per key, each waiting
    out a 15 s timeout for an answer nobody reads.

    The guard is on the socket, not on the function: a test that wants that
    path patches `urllib.request.urlopen` itself and its patch wins, so this
    catches the calls nobody meant to make without blocking the ones they did.
    """
    import urllib.request

    def _refuse(*a, **k):
        raise OSError("urllib.request.urlopen is blocked in tests; patch it if the test needs it")
    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
