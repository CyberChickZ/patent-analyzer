import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


def _client(monkeypatch, job):
    import app.main as m
    monkeypatch.setattr(m, "_save_job", lambda j: None)
    monkeypatch.setattr(m, "_enqueue_job", lambda jid: job.__setitem__("_enqueued", True))
    m.jobs[job["id"]] = job
    return TestClient(m.app)


def _paused_job():
    return {"id": "j1", "status": "waiting_for_hitl", "paused_at": "extract", "phase": "phase2", "output_dir": "/tmp/x",
            "pause_after": ["extract"], "phase_checkpoints": {"extract": "ck1"},
            "_hitl_saved_state": {"summary": "S", "extraction": {"candidate_inventions": [{"id": "inv1", "elements": [{"id": "inv1.e0", "text": "A"}]}]},
                                  "checklist": [{"id": "c1", "criterion": "x"}]}}


def test_state_shows_editable_values_and_context(monkeypatch):
    c = _client(monkeypatch, _paused_job())
    r = c.get("/api/jobs/j1/state").json()
    assert r["paused_at"] == "extract" and set(r["editable"]) == {"extraction", "checklist"}
    assert r["values"]["extraction"]["candidate_inventions"][0]["id"] == "inv1" and r["context"] == {"summary": "S"}


def test_patch_rejects_keys_outside_the_phase_and_stages_the_rest(monkeypatch):
    job = _paused_job()
    c = _client(monkeypatch, job)
    assert c.patch("/api/jobs/j1/state", json={"ranked_candidates": []}).status_code == 400
    r = c.patch("/api/jobs/j1/state", json={"extraction": {"candidate_inventions": []}})
    assert r.status_code == 200 and r.json()["pending_edits"] == ["extraction"]
    assert c.get("/api/jobs/j1/state").json()["values"]["extraction"] == {"candidate_inventions": []}


def test_resume_continue_builds_edit_response_and_enqueues(monkeypatch):
    job = _paused_job()
    c = _client(monkeypatch, job)
    c.patch("/api/jobs/j1/state", json={"checklist": []})
    r = c.post("/api/jobs/j1/resume", json={"action": "continue"})
    assert r.status_code == 200 and job["_pending_response"] == {"type": "edit", "args": {"checklist": []}}
    assert job["status"] == "queued" and job.get("_enqueued") and job["hitl_pending"] is None


def test_resume_rerun_phase_sets_replay_and_prompt_overrides(monkeypatch):
    job = _paused_job()
    c = _client(monkeypatch, job)
    r = c.post("/api/jobs/j1/resume", json={"action": "rerun_phase", "prompt_overrides": {"extract.elements": 3}})
    assert r.status_code == 200 and job["_replay_from"] == "extract" and job["prompt_overrides"] == {"extract.elements": 3}
    assert c.post("/api/jobs/j1/resume", json={"action": "continue"}).status_code == 400


def test_prompt_endpoints_list_get_put(monkeypatch, tmp_path):
    from app import prompts
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(prompts, "_store", None)
    prompts._cache.clear()
    c = _client(monkeypatch, _paused_job())
    names = [p["name"] for p in c.get("/api/prompts").json()]
    assert "extract.elements" in names and "search.facets" in names
    r = c.put("/api/prompts/search.facets", json={"text": "NEW {listing} {summary}", "by": "harry"}).json()
    assert r["version"] == 1 and r["current"] == 1
    d = c.get("/api/prompts/search.facets").json()
    assert d["current"] == 1 and d["versions"][0]["text"].startswith("NEW") and "{listing}" in d["default"]
    assert c.put("/api/prompts/search.facets/current", json={"version": 0}).json()["current"] == 0
    assert c.get("/api/prompts/nope").status_code == 404
