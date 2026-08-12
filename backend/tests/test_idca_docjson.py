import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm
from nodes import idca

TXT = """Title: Fast Widget Alignment

Abstract
We align widgets quickly using a learned offset.

1 Introduction
Widgets drift.

2 Related Work
Smith aligned widgets by hand.

3 Method
The offset network predicts a translation from the widget image.
"""

DOC = {"title": "Fast Widget Alignment", "abstract": "We align widgets quickly using a learned offset.",
       "sections": [{"heading": "1 Introduction", "level": 1, "paragraphs": ["Widgets drift."]},
                    {"heading": "2 Related Work", "level": 1, "paragraphs": ["Smith aligned widgets by hand."]},
                    {"heading": "3 Method", "level": 1,
                     "paragraphs": ["The offset network predicts a translation from the widget image."]}],
       "figures": [{"label": "Figure 1", "caption": "The offset network."}], "equations": [], "references_count": 4}

DETECT = {"status_determination": "Present", "has_innovation": True, "reasoning": "r", "doc_type": "invention",
          "input_mode": "academic_paper", "category": "Process", "fields_map": ["CV"], "source_citation": "",
          "cpc_subclass": "G06T", "publication_date": "", "summary": "S"}


def _patch(monkeypatch, doc=DOC, detect=DETECT, calls=None):
    calls = calls if calls is not None else []

    async def fake_detect(document_text, source_pdf_path=None):
        calls.append(("detect", len(document_text)))
        return dict(detect)

    async def fake_docjson(document_text, source_pdf_path=None):
        calls.append(("docjson", len(document_text)))
        if isinstance(doc, Exception):
            raise doc
        return doc

    async def fake_personas(**kw):
        return {"p": "persona"}

    monkeypatch.setattr(llm, "detect_and_summarize_invention", fake_detect)
    monkeypatch.setattr(llm, "build_doc_json", fake_docjson)
    monkeypatch.setattr(llm, "craft_personas", fake_personas)
    return calls


def _run(tmp_path, monkeypatch, state_extra=None, **kw):
    calls = _patch(monkeypatch, **kw)
    f = tmp_path / "in.txt"
    f.write_text(TXT)
    out = asyncio.run(idca.idca_node({"input_local_path": str(f), **(state_extra or {})}))
    return out, calls


def test_doc_json_is_the_text_layer(tmp_path, monkeypatch):
    out, calls = _run(tmp_path, monkeypatch)
    assert [c[0] for c in calls] == ["detect", "docjson"]
    assert out["doc_json"] == DOC
    assert out["document_text"].startswith("Title: Fast Widget Alignment\n\nAbstract\n[S0.P1] We align")
    assert "[S3.P1] The offset network predicts" in out["document_text"]
    assert "# Figures\n[S4.P1] Figure 1: The offset network." in out["document_text"]
    st = out["doc_json_stats"]
    assert st["source"] == "gemini" and st["sections"] == 3 and st["paragraphs"] == 5 and st["figures"] == 1
    assert st["fallback_paragraphs"] == 4     # abstract + 3 body paragraphs from the plain splitter
    assert out["input_mode"] == "academic_paper"
    assert any(e["kind"] == "doc_json" for e in out["events"])


def test_explicit_input_mode_wins_and_manuscript_strips_related_work(tmp_path, monkeypatch):
    out, _ = _run(tmp_path, monkeypatch, state_extra={"input_mode": "manuscript"})
    assert out["input_mode"] == "manuscript"
    assert "Smith aligned" not in out["document_text"]
    assert out["doc_json"]["dropped_sections"] == ["2 Related Work"]
    assert "[S2.P1] The offset network predicts" in out["document_text"]
    out2, _ = _run(tmp_path, monkeypatch, state_extra={"input_mode": "bogus"})
    assert out2["input_mode"] == "academic_paper"   # unknown -> detection result


def test_doc_json_failure_falls_back_to_plain_text(tmp_path, monkeypatch):
    out, _ = _run(tmp_path, monkeypatch, doc=RuntimeError("429"))
    assert out["doc_json"] is None
    assert out["document_text"] == TXT
    assert out["doc_json_stats"]["source"] == "none" and out["doc_json_stats"]["fallback_paragraphs"] == 4
    assert any(e["kind"] == "doc_json_failed" for e in out["events"])


def test_doc_json_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("IDCA_DOC_JSON", "0")
    out, calls = _run(tmp_path, monkeypatch)
    assert [c[0] for c in calls] == ["detect"]
    assert out["doc_json"] is None and out["document_text"] == TXT
