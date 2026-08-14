import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

import app.llm as llm


def test_stage_model_env_overrides_and_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "MODEL", "gemini-2.5-pro")
    for st in llm.STAGES:
        monkeypatch.delenv(f"LLM_MODEL_{st.upper()}", raising=False)
        assert llm.stage_model(st) == "gemini-2.5-pro"
    monkeypatch.setenv("LLM_MODEL_SCREEN", "gemini-3.1-flash-lite")
    monkeypatch.setenv("LLM_MODEL_EXTRACT", "gemini-3.5-flash")
    assert llm.stage_model("screen") == "gemini-3.1-flash-lite"
    assert llm.stage_model("extract") == "gemini-3.5-flash"
    assert llm.stage_model("eval") == "gemini-2.5-pro"
    monkeypatch.setenv("LLM_MODEL_EVAL", "")
    assert llm.stage_model("eval") == "gemini-2.5-pro"


def _fake_client(seen: list):
    async def generate_content(model, contents, config):
        seen.append({"model": model, "config": config})
        part = SimpleNamespace(text='{"ok": true}', thought=False)
        cand = SimpleNamespace(content=SimpleNamespace(parts=[part]))
        return SimpleNamespace(candidates=[cand], text='{"ok": true}',
                               usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=4,
                                                              thoughts_token_count=2))
    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))


def _no_gate(monkeypatch):
    async def _smooth(model=None):
        _smooth.models.append(model)
    _smooth.models = []
    monkeypatch.setattr(llm, "_smooth", _smooth)
    return _smooth


def test_call_llm_model_param_reaches_vertex_and_gate(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "get_client", lambda: _fake_client(seen))
    gate = _no_gate(monkeypatch)
    monkeypatch.setattr(llm, "MODEL", "gemini-2.5-pro")
    llm.usage.clear()
    asyncio.run(llm.call_llm("s", "u"))
    asyncio.run(llm.call_llm("s", "u", model="gemini-3.5-flash", thinking_budget=4096))
    assert [x["model"] for x in seen] == ["gemini-2.5-pro", "gemini-3.5-flash"]
    assert gate.models == ["gemini-2.5-pro", "gemini-3.5-flash"]
    assert llm.usage["gemini-3.5-flash"]["prompt_tokens"] == 10
    assert llm.usage["gemini-3.5-flash"]["thought_tokens"] == 2
    assert llm.usage["gemini-2.5-pro"]["calls"] == 1


def test_gemini3_gets_thinking_level_not_budget():
    import pytest
    from google.genai import types
    if "thinking_level" not in types.ThinkingConfig.model_fields:
        pytest.skip("google-genai too old for thinking_level")
    c = llm._build_config("s", 100, 4096, model="gemini-3.5-flash")
    assert c.thinking_config.thinking_budget is None and str(c.thinking_config.thinking_level.value) == "MEDIUM"
    c = llm._build_config("s", 100, 8192, model="gemini-3.1-flash-lite")
    assert c.thinking_config.thinking_level.value == "HIGH"
    c = llm._build_config("s", 100, 0, model="gemini-3.1-flash-lite")
    assert c.thinking_config.thinking_level.value == "MINIMAL"
    c = llm._build_config("s", 100, 0, model="gemini-3.8-flash")
    assert c.thinking_config.thinking_level.value == "LOW"
    c = llm._build_config("s", 100, 8192, model="gemini-2.5-pro")
    assert c.thinking_config.thinking_budget == 8192 and c.thinking_config.thinking_level is None
    c = llm._build_config("s", 100, 0, model="gemini-2.5-flash")
    assert c.thinking_config is None


def test_stage_functions_pass_stage_model(monkeypatch):
    calls = []

    async def fake(system, user, *a, **k):
        calls.append(k.get("model"))
        return json.dumps({"candidate_inventions": [], "ok": True, "issues": [], "verdicts": []})

    monkeypatch.setattr(llm, "call_llm", fake)
    monkeypatch.setattr(llm, "call_llm_with_pdfs", fake)
    monkeypatch.setattr(llm, "MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("LLM_MODEL_EXTRACT", "m-extract")
    monkeypatch.setenv("LLM_MODEL_EVAL", "m-eval")
    monkeypatch.setenv("LLM_MODEL_IDCA", "m-idca")
    monkeypatch.setenv("LLM_MODEL_SCREEN", "m-screen")

    asyncio.run(llm.extract_candidates("doc", "sum"))
    asyncio.run(llm.self_check("label", "src", "gen"))
    asyncio.run(llm.evaluate_single_document_text("summary", [{"id": "c1", "criterion": "x", "weight": 1}],
                                                  "prior text", "T", "paper"))
    asyncio.run(llm.detect_and_summarize_invention("some document text " * 50))
    from patent_analyzer.agentic import prune
    docs = [{"title": "t", "abstract": "a", "prune_cos": 0.5}]
    asyncio.run(prune.stage2_llm([{"id": "inv1", "concept": "c"}], [{"id": "e1", "text": "t"}], docs, [0]))
    assert calls[:2] == ["m-extract", "m-extract"]
    assert "m-eval" in calls and "m-idca" in calls and calls[-1] == "m-screen"


def test_llm_cache_key_includes_model(monkeypatch, tmp_path):
    import importlib
    import llm_cache
    importlib.reload(llm_cache)
    monkeypatch.setattr(llm_cache, "CACHE_DIR", tmp_path)
    live = []

    async def real_text(system, user, max_tokens, thinking_budget, response_schema=None, model=None):
        live.append(model)
        return f"answer-from-{model}"

    async def real_pdfs(system, user, pdf_paths, max_tokens, thinking_budget, image_parts=None,
                        response_schema=None, model=None):
        live.append(model)
        return f"pdf-from-{model}"

    monkeypatch.setattr(llm, "call_llm", real_text)
    monkeypatch.setattr(llm, "call_llm_with_pdfs", real_pdfs)
    monkeypatch.setattr(llm, "MODEL", "gemini-2.5-pro")
    llm_cache.install()
    assert asyncio.run(llm.call_llm("s", "u")) == "answer-from-gemini-2.5-pro"
    assert asyncio.run(llm.call_llm("s", "u", model="gemini-3.5-flash")) == "answer-from-gemini-3.5-flash"
    assert asyncio.run(llm.call_llm("s", "u", model="gemini-3.5-flash")) == "answer-from-gemini-3.5-flash"
    assert asyncio.run(llm.call_llm("s", "u")) == "answer-from-gemini-2.5-pro"
    assert live == ["gemini-2.5-pro", "gemini-3.5-flash"]
    assert llm_cache.stats == {"hits": 2, "misses": 2, "chars_in": 4, "chars_out": len("answer-from-gemini-2.5-pro") + len("answer-from-gemini-3.5-flash")}
    # default-model key is byte-identical to the historical key (no llm.MODEL → model change)
    legacy = tmp_path / (llm_cache._key("text", "gemini-2.5-pro", "s", "u", llm.MAX_TOKENS, 0) + ".json")
    assert legacy.exists() and json.loads(legacy.read_text())["model"] == "gemini-2.5-pro"
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    assert asyncio.run(llm.call_llm_with_pdfs("s", "u", [str(pdf)], model="gemini-3.1-flash-lite")) == "pdf-from-gemini-3.1-flash-lite"
    assert asyncio.run(llm.call_llm_with_pdfs("s", "u", [str(pdf)])) == "pdf-from-gemini-2.5-pro"
    assert live[-2:] == ["gemini-3.1-flash-lite", "gemini-2.5-pro"]
