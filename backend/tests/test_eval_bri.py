import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm, prompts


def _prompt_seen(monkeypatch):
    seen = {}

    async def fake(system, prompt, **kw):
        seen["prompt"] = prompt
        return '{"checklist_results": {}}'

    monkeypatch.setattr(llm, "call_llm", fake)
    checklist = [{"id": "lim1", "criterion": "a similarity calculator", "weight": 1.0}]
    asyncio.run(llm.evaluate_single_document_text("inv", checklist, "x" * 200, "t", "Patent", doc_mode="full_text"))
    return seen["prompt"]


def test_bri_instruction_only_with_env(monkeypatch):
    monkeypatch.delenv("EVAL_BRI", raising=False)
    assert "BROADEST REASONABLE INTERPRETATION" not in _prompt_seen(monkeypatch)
    monkeypatch.setenv("EVAL_BRI", "1")
    p = _prompt_seen(monkeypatch)
    assert "Do NOT infer beyond what the text states." in p
    assert p.count("BROADEST REASONABLE INTERPRETATION") == 1
    assert "evidence_quotes copied exactly from the document" in p


def test_bri_instruction_is_a_registry_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(prompts, "_store", None)
    prompts._cache.clear()
    prompts.set_overrides(None)
    assert "evaluate.bri_instruction" in prompts.names()
    assert prompts.get("evaluate.bri_instruction")[1] == 0
    prompts.set_overrides({"evaluate.bri_instruction": "\nOVERRIDE BRI TEXT"})
    monkeypatch.setenv("EVAL_BRI", "1")
    p = _prompt_seen(monkeypatch)
    assert "OVERRIDE BRI TEXT" in p and "BROADEST REASONABLE INTERPRETATION" not in p
    prompts.set_overrides(None)
