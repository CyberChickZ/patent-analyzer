import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import cache, cloud_state
from patent_analyzer.runtime_state import Breaker, MonthlyQuota


@pytest.fixture(autouse=True)
def _kv(tmp_path):
    cache.reset_for_tests(tmp_path / "kv.sqlite")
    yield
    cache.reset_for_tests(None)


def test_breaker_shared_between_instances():
    a, b = Breaker("gp", cooldown_s=60, local_ttl_s=0), Breaker("gp", cooldown_s=60, local_ttl_s=0)
    assert not a.is_open() and not b.is_open()
    a.trip("Sorry page")
    assert b.is_open()
    doc, _, _ = cloud_state.read("breaker:gp")
    assert doc["trips"] == 1 and doc["reason"] == "Sorry page"


def test_breaker_expires(monkeypatch):
    import patent_analyzer.runtime_state as rs
    b = Breaker("gp", cooldown_s=10, local_ttl_s=0)
    b.trip()
    assert b.is_open()
    monkeypatch.setattr(rs.time, "time", lambda: 10 ** 10)
    assert not b.is_open()


def test_monthly_quota_take_and_cap():
    q = MonthlyQuota("serpapi:k1", cap=3)
    assert q.take() and q.take() and q.take()
    assert not q.take()
    assert q.used() == 3 and q.remaining() == 0
    q.release()
    assert q.take()


def test_quota_isolated_per_name():
    MonthlyQuota("serpapi:k1", 1).take()
    assert MonthlyQuota("serpapi:k2", 1).remaining() == 1


def test_minute_gate_paces_in_process_and_divides_the_allowance(monkeypatch):
    """The gate is the one piece of state here that is deliberately NOT shared.

    It answers "sleep or go" in the milliseconds before a request, and asking
    GCS that question on every request costs more than the gate saves. So each
    instance gets its share of the allowance instead: with MAX_INSTANCES=3 a
    limit of 6/min becomes 2/min here, and the aggregate stays under the
    provider's limit whatever the autoscaler does. Fewer instances live only
    means we go slower than we are allowed."""
    import patent_analyzer.runtime_state as rs
    monkeypatch.setenv("MAX_INSTANCES", "3")
    g = rs.MinuteGate("t", 6)
    assert g.allowance == 6 and g.per_minute == 2
    assert g.take() == 0.0 and g.take() == 0.0
    wait = g.take()
    assert 0 < wait <= 60.1, "the third take in the same minute has to wait"

    monkeypatch.setenv("MAX_INSTANCES", "1")
    assert rs.MinuteGate("t", 6).per_minute == 6
    assert rs.MinuteGate("t", 0).take() == 0.0
    assert rs.MinuteGate("t", 2).per_minute >= 1, "a share never rounds down to zero"


def test_serial_lock_spaces_requests_by_cooldown_across_instances(tmp_path, monkeypatch):
    import asyncio, time
    monkeypatch.setenv("LOCK_DIR", str(tmp_path))
    from patent_analyzer.runtime_state import SerialLock

    async def go():
        t = []
        async def one():
            async with SerialLock("x", 0.3):
                t.append(time.time())
                await asyncio.sleep(0.05)
        await asyncio.gather(one(), one(), one())
        return sorted(t)
    ts = asyncio.run(go())
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert all(g >= 0.3 for g in gaps), gaps
