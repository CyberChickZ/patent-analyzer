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
