import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm


def _patch(monkeypatch, reply, seen: dict):
    async def fake_call_llm(system, user, max_tokens=llm.MAX_TOKENS, thinking_budget=0):
        seen["system"], seen["user"] = system, user
        seen["thinking_budget"], seen["max_tokens"] = thinking_budget, max_tokens
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
