import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer import cache
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
    doc = cache.kv().get("runtime", "breaker:gp")
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
