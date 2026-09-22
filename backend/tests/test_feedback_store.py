import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import feedback_store as fs


@pytest.fixture(autouse=True)
def local(monkeypatch, tmp_path):
    monkeypatch.setenv("FEEDBACK_STORE", "local")
    monkeypatch.setenv("FEEDBACK_LOCAL_DIR", str(tmp_path))
    monkeypatch.setenv("AUTH_DISABLED", "1")
    return tmp_path


def _client():
    import app.main as m
    return TestClient(m.app), m


def test_who_wrote_it_comes_from_the_token_not_the_payload():
    """A comment that says who wrote it, where the writer chose the name, is
    not a record of anything."""
    e = fs.new_entry({"kind": "comment", "text": "hi", "by": "someone-else@x"}, "real@oregonstate.edu")
    assert e["by"] == "real@oregonstate.edu"


def test_system_written_kinds_are_marked_auto():
    """"The system noticed this" must never read as "somebody complained"."""
    assert fs.new_entry({"kind": "prompt_edit"}, "x")["auto"] is True
    assert fs.new_entry({"kind": "reviewer_edit"}, "x")["auto"] is True
    assert fs.new_entry({"kind": "comment"}, "x")["auto"] is False
    assert fs.new_entry({"kind": "rating", "rating": 1}, "x")["auto"] is False
    with pytest.raises(ValueError):
        fs.new_entry({"kind": "nonsense"}, "x")


def test_round_trip_and_filters():
    fs.save(fs.new_entry({"kind": "comment", "job_id": "j1", "text": "a"}, "u"))
    fs.save(fs.new_entry({"kind": "rating", "job_id": "j2", "rating": -1}, "u"))
    fs.save(fs.new_entry({"kind": "prompt_edit", "prompt_name": "search.react_step",
                          "prompt_version": 3}, "u"))
    assert fs.listing()["total"] == 3
    assert fs.listing(job="j1")["total"] == 1
    assert fs.listing(kind="prompt_edit")["total"] == 1
    assert fs.listing(prompt_name="search.react_step")["total"] == 1
    assert fs.listing(status="addressed")["total"] == 0


def test_newest_first():
    import time
    a = fs.save(fs.new_entry({"kind": "comment", "text": "first"}, "u"))
    time.sleep(0.01)
    b = fs.save(fs.new_entry({"kind": "comment", "text": "second"}, "u"))
    ids = [e["id"] for e in fs.listing()["entries"]]
    assert ids[0] == b["id"] and ids[1] == a["id"]


def test_marking_addressed_records_what_addressed_it():
    e = fs.save(fs.new_entry({"kind": "comment", "text": "queries are too narrow"}, "u"))
    out = fs.patch(e["id"], {"status": "addressed",
                             "addressed_by": {"prompt_name": "search.react_step", "version": 4}}, "u2")
    assert out["status"] == "addressed" and out["addressed_by"]["version"] == 4
    assert out["addressed_at"]
    back = fs.patch(e["id"], {"status": "open"}, "u2")
    assert back["status"] == "open" and back["addressed_by"] is None and back["addressed_at"] == ""


def test_an_auto_entry_cannot_be_reworded():
    """A record of a change is not an opinion; editing its text would make it
    a forgery of one."""
    e = fs.save(fs.new_entry({"kind": "prompt_edit", "prompt_name": "p", "prompt_version": 1}, "u"))
    out = fs.patch(e["id"], {"text": "something else"}, "u")
    assert out["text"] == e["text"]


def test_the_routes_need_a_token(monkeypatch):
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    c, m = _client()
    monkeypatch.delitem(m.app.dependency_overrides, m.require_auth, raising=False)
    assert c.get("/feedback").status_code == 401
    assert c.post("/feedback", json={"kind": "comment"}).status_code == 401
    assert c.patch("/feedback/abc", json={"status": "addressed"}).status_code == 401


def test_the_routes_create_list_and_patch():
    c, m = _client()
    m.app.dependency_overrides.pop(m.require_auth, None)
    r = c.post("/feedback", json={"kind": "comment", "job_id": "j9", "text": "wrong element",
                                  "target": {"tab": "evidence", "anchor": "US1B2"}})
    assert r.status_code == 200, r.text
    eid = r.json()["id"]
    assert r.json()["target"]["anchor"] == "US1B2"
    assert c.get("/feedback?job=j9").json()["total"] == 1
    assert c.patch(f"/feedback/{eid}", json={"status": "addressed"}).json()["status"] == "addressed"
    assert c.patch("/feedback/nosuch", json={"status": "addressed"}).status_code == 404
    assert c.post("/feedback", json={"kind": "nope"}).status_code == 400


def test_saving_a_prompt_version_writes_one_entry_and_a_diff(monkeypatch, tmp_path):
    import app.prompts as pr
    # its own store, or a version left on disk by an earlier run becomes the
    # "before" and the diff comes out empty — which is what happened
    monkeypatch.setattr(pr, "PROMPT_DIR", tmp_path / "prompts")
    monkeypatch.setenv("PROMPT_STORE", "file")
    pr._store = None
    pr._cache.pop("t.fb", None)
    pr.register_default("t.fb", "line one\nline two\n")
    try:
        c, m = _client()
        m.app.dependency_overrides.pop(m.require_auth, None)
        r = c.put("/api/prompts/t.fb", json={"text": "line one\nline three\n", "by": "harry",
                                         "instruction": "say three"})
        assert r.status_code == 200, r.text
        rows = fs.listing(kind="prompt_edit")["entries"]
        assert len(rows) == 1
        assert rows[0]["prompt_name"] == "t.fb" and rows[0]["instruction"] == "say three"
        assert "line three" in rows[0]["diff_summary"] and rows[0]["auto"] is True
    finally:
        pr._DEFAULTS.pop("t.fb", None)
        pr._cache.pop("t.fb", None)
        pr._store = None
