import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm
from app import prompts


def _patch(monkeypatch, reply, seen: dict):
    async def fake_call_llm(system, user, max_tokens=llm.MAX_TOKENS, thinking_budget=0, response_schema=None, model=None):
        seen["system"], seen["user"], seen["thinking_budget"], seen["model"] = system, user, thinking_budget, model
        return reply if isinstance(reply, str) else json.dumps(reply)
    monkeypatch.setattr(llm, "call_llm", fake_call_llm)


def test_draft_prompts_are_registered():
    assert {"draft.claims", "draft.reword", "draft.definiteness"} <= set(prompts.names())
    assert "draft" in llm.STAGES


def test_draft_claims_keeps_lids_and_cleans_quotes(monkeypatch):
    seen = {}
    _patch(monkeypatch, {"primary": [{"lid": "c1.l1", "text": "  receiving   a signal "}, {"lid": "c1.l2", "text": "filtering the signal"}],
                         "mirror": [{"lid": "m.l1", "text": "a receiver configured to receive a signal"}],
                         "pool": [{"pid": "hint0", "text": "wherein the signal is ultra-wideband", "evidence_quote": "UWB  pulses at\n10Hz"},
                                  {"text": "no pid"}],
                         "refinements": [{"element_id": "inv1.e2", "text": "wherein the filter is a 5-tap FIR filter", "evidence_quote": "5-tap FIR"},
                                         {"text": ""}, {"element_id": "x", "text": "r2"}, {"element_id": "y", "text": "r3"}, {"element_id": "z", "text": "r4"}]}, seen)
    out = asyncio.run(llm.draft_claims("method", [{"lid": "c1.l1", "text": "receiving a signal"}, {"lid": "c1.l2", "text": "filtering the signal"}],
                                       [{"lid": "m.l1", "text": "one or more processors configured to receive a signal"}],
                                       [{"pid": "hint0", "text": "the signal is UWB", "evidence_quote": ""}],
                                       [{"element_id": "inv1.e2", "text": "filtering the signal", "passage": "… a 5-tap FIR …"}],
                                       ["US1 discloses inv1.e1"], "DOC TEXT"))
    assert out["primary"] == {"c1.l1": "receiving a signal", "c1.l2": "filtering the signal"}
    assert out["mirror"] == {"m.l1": "a receiver configured to receive a signal"}
    assert out["pool"]["hint0"] == {"text": "wherein the signal is ultra-wideband", "evidence_quote": "UWB pulses at 10Hz"}
    assert [r["element_id"] for r in out["refinements"]] == ["inv1.e2", "x", "y"]         # capped at 3, empty text dropped
    assert seen["thinking_budget"] == 2048 and seen["model"] == llm.stage_model("draft")
    assert "DOC TEXT" in seen["user"] and "US1 discloses inv1.e1" in seen["user"] and "5-tap FIR" in seen["user"]
    assert "may NOT add, drop, merge, split or reorder" in seen["user"]


def test_draft_claims_unparseable_reply_returns_error(monkeypatch):
    _patch(monkeypatch, "not json", {})
    out = asyncio.run(llm.draft_claims("method", [], [], [], [], [], ""))
    assert out["primary"] == {} and "parsed" in out["error"]


def test_reword_limitations_maps_lid_to_text(monkeypatch):
    seen = {}
    _patch(monkeypatch, {"limitations": [{"lid": "c1.l2", "text": " filtering the signal with a 5-tap FIR filter "}, {"lid": "", "text": "x"}]}, seen)
    out = asyncio.run(llm.reword_limitations([{"lid": "c1.l2", "text": "filtering the signal substantially",
                                                "flags": [{"category": "relative_term", "span": "substantially", "note": "no standard"}],
                                                "quotes": ["a 5-tap FIR filter"]}]))
    assert out == {"c1.l2": "filtering the signal with a 5-tap FIR filter"}
    assert "relative_term — 'substantially'" in seen["user"] and "5-tap FIR" in seen["user"]
    assert asyncio.run(llm.reword_limitations([])) == {}


def test_definiteness_advisory_normalizes_pedantic_shape(monkeypatch):
    seen = {}
    _patch(monkeypatch, {"claims": [
        {"no": 1, "likelihood_indefinite": "unlikely", "indefiniteness_reasons": []},
        {"no": "2", "likelihood_indefinite": "Very Good Chance", "indefiniteness_reasons": [
            {"category": "antecedent basis", "reason_text": "the filter has no antecedent", "claim_recitations": ["the filter"], "likelihood": "likely"},
            {"category": "weird", "reason_text": "?", "claim_recitations": [], "likelihood": "about even"}]},
        {"no": "n/a"}]}, seen)
    out = asyncio.run(llm.definiteness_advisory([{"no": 1, "text": "1. A method…", "depends_on": None},
                                                 {"no": 2, "text": "2. The method of claim 1, wherein the filter…", "depends_on": 1}], "SPEC"))
    assert out[1]["p_indefinite"] == 0.2 and out[1]["reasons"] == []
    assert out[2]["p_indefinite"] == 0.8 and out[2]["reasons"][0]["category"] == "antecedent_basis" and out[2]["reasons"][0]["p"] == 0.7
    assert out[2]["reasons"][1]["category"] == "other"
    assert 3 not in out and "SPEC" in seen["user"] and "parent text: 1. A method" in seen["user"]
    assert "a single issue renders the entire claim indefinite" in seen["user"] and '"antecedent basis"' in seen["user"]
