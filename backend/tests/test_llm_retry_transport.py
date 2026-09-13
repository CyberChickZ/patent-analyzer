import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import pytest

from app.llm import _is_retryable


@pytest.mark.parametrize("exc", [
    httpx.ConnectError(""),          # the one that killed 39 Phase 4 documents
    httpx.ReadError(""),
    httpx.WriteError(""),
    httpx.ConnectTimeout(""),
    httpx.ReadTimeout(""),
    httpx.PoolTimeout(""),
    httpx.RemoteProtocolError(""),
    httpx.ProxyError(""),
])
def test_transport_failures_are_retried(exc):
    assert _is_retryable(exc) is True


@pytest.mark.parametrize("exc", [
    httpx.LocalProtocolError(""),    # this process is wrong; four tries is four bugs
    httpx.UnsupportedProtocol(""),
    ValueError("bad checklist"),
])
def test_our_own_mistakes_are_not_retried(exc):
    assert _is_retryable(exc) is False


def test_a_transport_name_from_a_non_httpx_stack_still_retries():
    class ConnectError(Exception):      # e.g. httpcore leaking through un-wrapped
        pass
    assert _is_retryable(ConnectError("")) is True
