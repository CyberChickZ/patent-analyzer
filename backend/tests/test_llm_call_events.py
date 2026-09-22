"""A long model call must not look like a hung one."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm
from patent_analyzer import event_log


class _Resp:
    class usage_metadata:
        prompt_token_count = 12000
        candidates_token_count = 5500
        thoughts_token_count = 700


@pytest.fixture
def seen(monkeypatch):
    rows = []
    tok = event_log.set_sink(rows.append)
    monkeypatch.setattr(llm, "HEARTBEAT_EVERY_S", 0.05)
    yield rows
    event_log.reset_sink(tok)


def test_start_and_finish_say_which_model_how_long_and_what_it_cost(seen, monkeypatch):
    monkeypatch.setattr(llm.prompts, "last_rendered", lambda: "extract.elements")

    async def go():
        w = await llm._CallWatch("gemini-2.5-pro").start()
        w.done(_Resp())
    asyncio.run(go())
    kinds = [e["kind"] for e in seen]
    assert kinds[0] == "llm_start" and kinds[-1] == "llm_done"
    assert seen[0]["message"] == "calling gemini-2.5-pro · extract.elements"
    assert "returned ·" in seen[-1]["message"] and "18k tokens" in seen[-1]["message"]
    assert "gemini-2.5-pro" in seen[-1]["message"]


def test_a_long_call_keeps_saying_it_is_still_running(seen, monkeypatch):
    monkeypatch.setattr(llm.prompts, "last_rendered", lambda: "")

    async def go():
        w = await llm._CallWatch("gemini-3.8-flash").start()
        await asyncio.sleep(0.17)
        w.done(_Resp())
    asyncio.run(go())
    beats = [e for e in seen if e["kind"] == "llm_running"]
    assert beats, "a caller cannot tell a slow call from a dead one without these"
    assert "still waiting on gemini-3.8-flash" in beats[0]["message"]
    assert beats[0]["message"].rstrip("s").split("· ")[-1].isdigit()


def test_the_heartbeat_stops_when_the_call_does(seen, monkeypatch):
    monkeypatch.setattr(llm.prompts, "last_rendered", lambda: "")

    async def go():
        w = await llm._CallWatch("m").start()
        w.done(_Resp())
        await asyncio.sleep(0.16)
    asyncio.run(go())
    assert [e["kind"] for e in seen].count("llm_running") == 0


def test_a_failed_call_is_an_event_too(seen, monkeypatch):
    monkeypatch.setattr(llm.prompts, "last_rendered", lambda: "")

    async def go():
        w = await llm._CallWatch("m").start()
        w.failed(RuntimeError("connection reset"))
    asyncio.run(go())
    assert seen[-1]["kind"] == "llm_failed" and "connection reset" in seen[-1]["message"]


def test_outside_a_run_these_cost_nothing():
    """No sink, no events — the eval scripts and the tests must not pay for a
    feature that only the server uses."""
    async def go():
        w = await llm._CallWatch("m").start()
        w.done(_Resp())
    asyncio.run(go())            # must not raise with no sink installed
