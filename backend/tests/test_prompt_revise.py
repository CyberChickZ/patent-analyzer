import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.prompts as pr


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_DISABLED", "1")
    monkeypatch.setattr(pr, "PROMPT_DIR", tmp_path)
    pr._store = None
    pr._cache.pop("t.rev", None)
    pr.register_default("t.rev", "You do a thing.\nELEMENTS: {elements}\nOutput {{\"a\": 1}}",
                        contract="Does: a thing.\nMust output: {\"a\": int}.")
    import app.main as m
    m.app.dependency_overrides.pop(m.require_auth, None)
    yield TestClient(m.app)
    pr._DEFAULTS.pop("t.rev", None)
    pr._CONTRACTS.pop("t.rev", None)
    pr._cache.pop("t.rev", None)
    pr._store = None


def _answer(monkeypatch, text, rationale="tightened the wording"):
    import app.llm as llm
    seen = {}

    async def fake(system, user, **kw):
        seen["system"], seen["user"], seen["kw"] = system, user, kw
        return json.dumps({"text": text, "rationale": rationale})
    monkeypatch.setattr(llm, "call_llm", fake)
    return seen


def test_the_model_is_shown_the_style_the_contract_and_the_current_text(client, monkeypatch):
    """A rewrite that cannot see the contract is a rewrite that will quietly
    break it."""
    seen = _answer(monkeypatch, "# Task\n\nYou do a thing.\n\n# Elements\n\n{elements}\n\nOutput {{\"a\": 1}}")
    r = client.post("/api/prompts/t.rev/revise", json={"instruction": "use the house headings"})
    assert r.status_code == 200, r.text
    assert "# Prompt style" in seen["user"]
    assert "Does: a thing." in seen["user"]
    assert "You do a thing." in seen["user"]
    assert "use the house headings" in seen["user"]
    assert seen["kw"].get("model") == "gemini-3.1-pro-preview"


def test_it_proposes_and_does_not_save(client, monkeypatch):
    _answer(monkeypatch, "rewritten {elements}")
    before = pr.describe("t.rev")["current"]
    r = client.post("/api/prompts/t.rev/revise", json={"instruction": "x"}).json()
    assert r["text"] == "rewritten {elements}" and r["rationale"]
    assert pr.describe("t.rev")["current"] == before, "revise must not write a version"
    assert "nothing was saved" in r["note"]


def test_a_rewrite_that_drops_a_placeholder_is_flagged_not_silently_returned(client, monkeypatch):
    """A missing {placeholder} breaks the prompt at render time, a long way
    from here."""
    _answer(monkeypatch, "no placeholders at all")
    r = client.post("/api/prompts/t.rev/revise", json={"instruction": "simplify"}).json()
    assert r["placeholders_lost"] == ["elements"]
    assert "NOT SAFE TO SAVE" in r["note"]


def test_an_empty_instruction_and_an_unknown_prompt_are_refused(client, monkeypatch):
    _answer(monkeypatch, "x {elements}")
    assert client.post("/api/prompts/t.rev/revise", json={"instruction": "  "}).status_code == 400
    assert client.post("/api/prompts/t.nope/revise", json={"instruction": "x"}).status_code == 404


def test_a_model_that_answers_with_prose_is_a_502_not_a_saved_prompt(client, monkeypatch):
    import app.llm as llm

    async def prose(system, user, **kw):
        return "Sure! Here is your rewritten prompt, I hope you like it."
    monkeypatch.setattr(llm, "call_llm", prose)
    assert client.post("/api/prompts/t.rev/revise", json={"instruction": "x"}).status_code == 502


def test_an_unpriced_model_is_reported_as_a_gap_not_as_free():
    """gemini-3.1-pro-preview has no entry in PRICES, so every call on it costs
    $0 in the ledger — which is the same shape as a call that did not happen."""
    from patent_analyzer import metering
    metering._unpriced.discard("made-up-model")
    metering.incidents.clear()
    assert metering.cost_usd("made-up-model", 1000, 100, 0) == 0.0
    assert any(i["kind"] == metering.UNPRICED for i in metering.incidents)
    n = len(metering.incidents)
    metering.cost_usd("made-up-model", 1000, 100, 0)
    assert len(metering.incidents) == n, "one incident per model, not per call"


def test_a_rewrite_that_drops_a_citation_or_an_example_is_reported(client, monkeypatch):
    """The placeholder check is necessary and not sufficient. Dogfooding this
    endpoint on four real prompts, every proposal kept its placeholders and two
    of them quietly dropped the MPEP citation that was a rule's whole reason,
    and the ("icg", "indocyanine green") example that teaches the acronym rule.
    A rewrite may shorten a sentence; it may not drop the fact it carried."""
    import app.prompts as pr
    pr._DEFAULTS["t.rev"] = ('Try every group: examiners search all analogous arts (MPEP 904.01(c)). '
                             'Give the acronym and its expansion ("icg", "indocyanine green"). '
                             'The field is about 100,000 documents. {elements}')
    _answer(monkeypatch, "Try every group, as examiners do. Give both forms. A large field. {elements}")
    r = client.post("/api/prompts/t.rev/revise", json={"instruction": "shorten"}).json()
    assert r["placeholders_lost"] == [], "the placeholder check passes — that is the point"
    assert "MPEP 904.01(c)" in r["specifics_lost"]
    assert "icg" in r["specifics_lost"] and "indocyanine green" in r["specifics_lost"]
    assert "100,000" in r["specifics_lost"]
    assert "went missing" in r["note"]
