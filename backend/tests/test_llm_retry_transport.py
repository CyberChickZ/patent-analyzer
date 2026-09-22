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


def _recorded(monkeypatch):
    import app.llm as llm
    seen = []
    monkeypatch.setattr(llm, "_incident", lambda model, kind, detail: seen.append((kind, detail)))
    return llm, seen


def test_running_out_of_attempts_is_recorded_as_a_failure(monkeypatch):
    """Job 99a35c00 closed with failures: 0 against 58 retry incidents while 14
    of 60 references came back unread. Retries that never succeed are a loss."""
    import asyncio
    llm, seen = _recorded(monkeypatch)
    monkeypatch.setattr(llm, "_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(llm, "_tenacity_retry", retry_n(3))

    calls = []

    @llm._retry_decorator
    async def always_drops():
        calls.append(1)
        raise httpx.ConnectError("")

    with pytest.raises(httpx.ConnectError):
        asyncio.run(always_drops())
    assert len(calls) == 3                                   # every attempt spent
    failed = [d for k, d in seen if k == "failed"]
    assert len(failed) == 1                                  # and the loss is on the books
    assert "gave up after 3 attempts" in failed[0]


def test_a_call_that_recovers_records_no_failure(monkeypatch):
    import asyncio
    llm, seen = _recorded(monkeypatch)
    monkeypatch.setattr(llm, "_tenacity_retry", retry_n(4))
    state = {"n": 0}

    @llm._retry_decorator
    async def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise httpx.ConnectError("")
        return "ok"

    assert asyncio.run(flaky()) == "ok"
    assert [k for k, _ in seen].count("failed") == 0


def test_a_non_retryable_error_is_not_counted_twice(monkeypatch):
    import asyncio
    llm, seen = _recorded(monkeypatch)
    monkeypatch.setattr(llm, "_tenacity_retry", retry_n(4))

    @llm._retry_decorator
    async def bad_input():
        raise ValueError("bad checklist")

    with pytest.raises(ValueError):
        asyncio.run(bad_input())
    # _is_retryable already booked it; the give-up path must not book it again
    assert [k for k, _ in seen].count("failed") == 1


def retry_n(n):
    """A tenacity decorator identical to the real one but with n attempts and
    no waiting, so the tests do not sleep through an exponential backoff."""
    from tenacity import retry, retry_if_exception, stop_after_attempt
    import app.llm as llm
    return retry(retry=retry_if_exception(llm._is_retryable),
                 stop=stop_after_attempt(n), reraise=True)


def test_attempt_count_is_a_knob_and_not_decoration(monkeypatch):
    """LLM_RETRY_ATTEMPTS is set in the Cloud Run env, so it has to reach
    tenacity's stop condition — a number nobody reads is worse than no knob."""
    import asyncio
    import importlib
    import os

    os.environ["LLM_RETRY_ATTEMPTS"] = "6"
    try:
        import app.llm as llm
        llm = importlib.reload(llm)
        assert llm._RETRY_ATTEMPTS == 6
        seen = []
        monkeypatch.setattr(llm, "_incident", lambda m, k, d: seen.append((k, d)))
        monkeypatch.setattr(llm, "_tenacity_retry", retry_n(6))   # same stop, no sleeping

        calls = []

        @llm._retry_decorator
        async def always_drops():
            calls.append(1)
            raise httpx.ConnectError("")

        with pytest.raises(httpx.ConnectError):
            asyncio.run(always_drops())
        assert len(calls) == 6
        assert any("gave up after 6 attempts" in d for k, d in seen if k == "failed")
    finally:
        del os.environ["LLM_RETRY_ATTEMPTS"]
        import app.llm as llm2
        importlib.reload(llm2)
