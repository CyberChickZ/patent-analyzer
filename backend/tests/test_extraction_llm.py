import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm


def _patch(monkeypatch, reply, seen: dict):
    async def fake_call_llm(system, user, max_tokens=llm.MAX_TOKENS, thinking_budget=0, model=None):
        seen["system"], seen["user"] = system, user
        seen["thinking_budget"], seen["max_tokens"], seen["model"] = thinking_budget, max_tokens, model
        return reply if isinstance(reply, str) else json.dumps(reply)
    monkeypatch.setattr(llm, "call_llm", fake_call_llm)


def test_extract_candidates_normalizes_and_sorts(monkeypatch):
    seen = {}
    _patch(monkeypatch, {"candidate_inventions": [
        {"id": "x", "concept": "  A widget aligner   driven by a learned offset. ", "level": "application",
         "cpc_pred": ["G06T7/00", "G06N3/08", "G06T5/00", "H04N1/00"]},
        {"id": "y", "concept": "The offset network itself.", "level": "bogus"},
        {"id": "z", "concept": "Aligning widgets in printers.", "level": "core"},
        {"concept": ""},
    ], "no_invention_reason": None}, seen)
    out = asyncio.run(llm.extract_candidates("DOC TEXT", "SUMMARY", "paper"))
    cands = out["candidate_inventions"]
    assert [c["level"] for c in cands] == ["core", "component", "application"]
    assert [c["id"] for c in cands] == ["inv1", "inv2", "inv3"]
    assert cands[0]["concept"] == "Aligning widgets in printers."
    assert cands[2]["cpc_pred"] == ["G06T7/00", "G06N3/08", "G06T5/00"]
    assert cands[2]["concept"] == "A widget aligner driven by a learned offset."
    assert out["no_invention_reason"] is None
    assert seen["thinking_budget"] == 4096
    assert "DOC TEXT" in seen["user"] and "SUMMARY" in seen["user"]
    assert "no_invention_reason" in seen["user"]
    assert "Method / Approach" in seen["user"]


def test_extract_candidates_no_invention_exit(monkeypatch):
    _patch(monkeypatch, {"candidate_inventions": [], "no_invention_reason": "literature review"}, {})
    out = asyncio.run(llm.extract_candidates("DOC", "S", "paper"))
    assert out == {"candidate_inventions": [], "no_invention_reason": "literature review"}


def test_extract_candidates_parse_failure(monkeypatch):
    _patch(monkeypatch, "not json at all", {})
    out = asyncio.run(llm.extract_candidates("DOC", "S", "disclosure"))
    assert out["candidate_inventions"] == [] and "parsed" in out["no_invention_reason"]


def test_extract_candidates_doc_kind_guidance(monkeypatch):
    seen = {}
    _patch(monkeypatch, {"candidate_inventions": [{"concept": "c"}]}, seen)
    asyncio.run(llm.extract_candidates("DOC", "S", "patent_draft"))
    assert "independent claim" in seen["user"]
    assert "Document kind: patent_draft" in seen["user"]


CANDS = [{"id": "inv1", "concept": "core thing", "level": "core", "cpc_pred": ["G06T7/00"]},
         {"id": "inv2", "concept": "sub thing", "level": "component", "cpc_pred": []}]


def test_extract_elements_shapes_output(monkeypatch):
    seen = {}
    _patch(monkeypatch, {"candidate_inventions": [
        {"id": "inv1",
         "independent_claim_draft": {"method": "A method of aligning, comprising: predicting an offset; and applying it.",
                                     "system": "A system comprising: an offset network."},
         "elements": [
             {"id": "inv1.e0", "text": "A method of aligning widgets", "evidence_quote": "we align widgets",
              "facets": {"thing": ["Widget Alignment Device", "device"], "place": ["printer unit"], "apparatus": []},
              "kind": "structure"},
             {"id": "inv1.e1", "text": "predicting an offset  with a network", "evidence_quote": "",
              "facets": {}, "kind": "weird"},
             {"text": ""},
         ],
         "primary_form": "SYSTEM", "dependent_hints": ["use L2", "", "rotation", "a", "b", "c"]},
        {"id": "inv9", "elements": []},
    ]}, seen)
    out = asyncio.run(llm.extract_elements("DOC TEXT", CANDS))
    inv1, inv2 = out["candidate_inventions"]
    assert inv1["concept"] == "core thing" and inv1["level"] == "core"
    assert inv1["independent_claim_draft"]["method"].startswith("A method of aligning")
    assert [e["id"] for e in inv1["elements"]] == ["inv1.e0", "inv1.e1"]
    assert inv1["elements"][0]["facets"] == {"thing": ["widget alignment"], "place": ["printer"], "apparatus": []}
    assert inv1["elements"][1]["kind"] == "step"
    assert inv1["elements"][1]["text"] == "predicting an offset with a network"
    assert inv1["dependent_hints"] == ["use L2", "rotation", "a", "b"]
    assert inv1["primary_form"] == "system" and inv2["primary_form"] == "method"
    assert inv2["elements"] == [] and inv2["independent_claim_draft"] == {"method": "", "system": ""}
    assert seen["thinking_budget"] == 4096 and seen["max_tokens"] == 16384
    assert "COPY the quote verbatim from the document. DO NOT paraphrase." in seen["user"]
    assert "One limitation per element." in seen["user"]
    assert "primary_form" in seen["user"]
    assert "never use device/member/element/portion/means/unit" in seen["user"]
    assert "PREFILLED" not in seen["user"]


def test_extract_elements_prefill_keeps_texts(monkeypatch):
    seen = {}
    fixed = ["A method of aligning widgets", "predicting an offset", "applying the offset"]
    _patch(monkeypatch, {"candidate_inventions": [
        {"id": "inv1", "independent_claim_draft": {"method": "m", "system": "s"},
         "elements": [
             {"id": "inv1.e0", "text": "REWRITTEN", "evidence_quote": "q0", "kind": "structure"},
             {"id": "inv1.e1", "text": "REWRITTEN too", "evidence_quote": "q1", "kind": "step"},
         ]}]}, seen)
    out = asyncio.run(llm.extract_elements("DOC", CANDS[:1], prefill={"inv1": fixed}))
    els = out["candidate_inventions"][0]["elements"]
    assert [e["text"] for e in els] == fixed
    assert [e["evidence_quote"] for e in els] == ["q0", "q1", ""]
    assert [e["id"] for e in els] == ["inv1.e0", "inv1.e1", "inv1.e2"]
    assert "PREFILLED ELEMENTS (FIXED)" in seen["user"] and "inv1.e2: applying the offset" in seen["user"]


def test_extract_elements_feedback_and_parse_failure(monkeypatch):
    seen = {}
    _patch(monkeypatch, "garbage", seen)
    out = asyncio.run(llm.extract_elements("DOC", CANDS, feedback={"issues": ["quote not in doc"], "suggestion": "copy"}))
    assert out["candidate_inventions"] == [] and "parsed" in out["error"]
    assert "FEEDBACK FROM YOUR PREVIOUS ATTEMPT" in seen["user"] and "quote not in doc" in seen["user"]
    assert asyncio.run(llm.extract_elements("DOC", [])) == {"candidate_inventions": []}
