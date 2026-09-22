import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.cache import SqliteKV


def test_put_get_roundtrip(tmp_path):
    kv = SqliteKV(tmp_path / "kv.sqlite")
    assert kv.get("search", "k1") is None
    kv.put("search", "k1", {"cands": [1, 2], "meta": {"total": 5}})
    assert kv.get("search", "k1") == {"cands": [1, 2], "meta": {"total": 5}}


def test_namespaces_isolated(tmp_path):
    kv = SqliteKV(tmp_path / "kv.sqlite")
    kv.put("a", "k", {"x": 1})
    assert kv.get("b", "k") is None


def test_max_age_expiry(tmp_path, monkeypatch):
    import patent_analyzer.cache as c
    kv = SqliteKV(tmp_path / "kv.sqlite")
    kv.put("s", "k", {"v": 1})
    monkeypatch.setattr(c.time, "time", lambda: 10 ** 10)
    assert kv.get("s", "k", max_age_days=1) is None
    assert kv.get("s", "k") == {"v": 1}


def test_incr_is_cumulative_across_instances(tmp_path):
    p = tmp_path / "kv.sqlite"
    a, b = SqliteKV(p), SqliteKV(p)
    assert a.incr("quota", "serpapi:key1:2026-09") == 1
    assert b.incr("quota", "serpapi:key1:2026-09") == 2
    assert a.incr("quota", "serpapi:key1:2026-09", by=3) == 5
    assert a.get("quota", "serpapi:key1:2026-09")["n"] == 5
