import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm


def _capture(monkeypatch):
    seen = {}

    async def fake_call_llm(system, user, *a, **kw):
        seen["user"] = user
        return '{"ok": true, "issues": [], "suggestion": "looks good"}'

    monkeypatch.setattr(llm, "call_llm", fake_call_llm)
    return seen


def test_self_check_sends_the_whole_text_layer_not_the_first_20k(monkeypatch):
    seen = _capture(monkeypatch)
    source = ("para %05d lorem ipsum dolor sit amet. " % i for i in range(3000))
    source = "".join(source)
    assert 100_000 < len(source) < llm._EXTRACTION_DOC_CAP
    res = asyncio.run(llm.self_check("claim drafting", source, "A method comprising widgets."))
    assert res["ok"] is True
    assert "para 02999" in seen["user"]
    assert source[20_000:20_040] in seen["user"]


def test_self_check_caps_at_the_extraction_cap(monkeypatch):
    seen = _capture(monkeypatch)
    source = "x" * (llm._EXTRACTION_DOC_CAP + 5000)
    asyncio.run(llm.self_check("claim drafting", source, "draft"))
    assert "x" * llm._EXTRACTION_DOC_CAP in seen["user"]
    assert "x" * (llm._EXTRACTION_DOC_CAP + 1) not in seen["user"]


def test_review_phase_output_sends_the_whole_input(monkeypatch):
    seen = _capture(monkeypatch)
    source = "".join("line %05d of the source document.\n" % i for i in range(3000))
    assert len(source) > 20_000
    asyncio.run(llm.review_phase_output("phase2", "draft claims", source, "output"))
    assert "line 02999" in seen["user"]
