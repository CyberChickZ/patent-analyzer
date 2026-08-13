import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.llm as llm
from graph.extraction_subgraph import (
    build_extraction_subgraph, claim_prefill, independent_claims, resolve_doc_text, should_retry_elements,
)

DOC = ("Title: Widget alignment\n\n# 1 Method\n"
       "[S1.P1] The offset network predicts a translation from the widget image.\n"
       "[S1.P2] The predicted translation is applied to the widget before printing.\n"
       "[S1.P3] Training uses an L1 loss between predicted and measured offsets.\n")

CANDS = {"candidate_inventions": [
    {"id": "inv1", "concept": "Learned widget alignment", "level": "core", "cpc_pred": ["G06T7/00"]},
    {"id": "inv2", "concept": "Offset network training", "level": "component", "cpc_pred": []},
], "no_invention_reason": None}

ELEMENTS = {"candidate_inventions": [
    {"id": "inv1", "concept": "Learned widget alignment", "level": "core", "cpc_pred": ["G06T7/00"],
     "independent_claim_draft": {"method": "A method of aligning a widget, comprising: predicting a translation "
                                           "from the widget image with an offset network; and applying the "
                                           "predicted translation to the widget before printing.",
                                 "system": "A system comprising: an offset network."},
     "elements": [
         {"id": "inv1.e0", "text": "A method of aligning a widget, comprising:",
          "evidence_quote": "predicts a translation from the widget image", "facets": {"thing": ["widget align"], "place": [], "apparatus": []}, "kind": "structure"},
         {"id": "inv1.e1", "text": "predicting a translation from the widget image with an offset network",
          "evidence_quote": "The offset network predicts a translation from the widget image",
          "facets": {"thing": ["offset"], "place": ["widget"], "apparatus": ["network"]}, "kind": "step"},
         {"id": "inv1.e2", "text": "applying the predicted translation to the widget before printing",
          "evidence_quote": "a hallucinated sentence about rotating the widget by ninety degrees",
          "facets": {"thing": [], "place": [], "apparatus": []}, "kind": "step"},
     ], "dependent_hints": ["L1 loss"]},
    {"id": "inv2", "concept": "Offset network training", "level": "component", "cpc_pred": [],
     "independent_claim_draft": {"method": "A method of training, comprising: minimizing an L1 loss.", "system": ""},
     "elements": [
         {"id": "inv2.e0", "text": "minimizing an L1 loss between predicted and measured offsets",
          "evidence_quote": "Training uses an L1 loss between predicted and measured offsets",
          "facets": {"thing": ["l1 loss"], "place": [], "apparatus": []}, "kind": "step"},
     ], "dependent_hints": []},
]}


def _patch(monkeypatch, calls, cands=CANDS, elements=ELEMENTS, checks=None):
    checks = list(checks or [{"ok": True, "issues": [], "suggestion": "looks good"}])

    async def fake_candidates(doc_text, summary, doc_kind):
        calls.append(("A1", doc_kind, doc_text))
        return cands

    async def fake_elements(doc_text, candidates, prefill=None, feedback=None):
        calls.append(("A2", prefill, feedback))
        return elements

    async def fake_self_check(label, source_text, generated_text, source_pdf_path=None):
        calls.append(("A5", label, generated_text))
        return checks.pop(0) if len(checks) > 1 else checks[0]

    monkeypatch.setattr(llm, "extract_candidates", fake_candidates)
    monkeypatch.setattr(llm, "extract_elements", fake_elements)
    monkeypatch.setattr(llm, "self_check", fake_self_check)


def test_happy_path_extraction_checklist_and_locations(monkeypatch):
    calls = []
    _patch(monkeypatch, calls)
    out = asyncio.run(build_extraction_subgraph().ainvoke(
        {"summary": "S", "document_text": DOC, "input_mode": "academic_paper"}))
    assert [c[0] for c in calls] == ["A1", "A2", "A5"]
    assert calls[0][1] == "paper"
    ext = out["extraction"]
    assert ext["doc_kind"] == "paper" and ext["no_invention_reason"] is None
    inv1 = ext["candidate_inventions"][0]
    assert inv1["level"] == "core" and inv1["cpc_pred"] == ["G06T7/00"]
    e0, e1, e2 = inv1["elements"]
    assert not e1["unsupported"] and e1["evidence_loc"]["section"] == "S1" and e1["evidence_loc"]["para"] == 1
    assert e1["evidence_loc"]["method"] == "exact"
    assert DOC[slice(*e1["evidence_loc"]["char"])].startswith("The offset network predicts")
    assert e2["unsupported"] and e2["evidence_loc"] is None
    assert inv1["claim_ratio"] > 0.9
    assert [c["id"] for c in out["checklist"]] == ["inv1.e0", "inv1.e1"]
    assert out["checklist"][0]["criterion"] == "A method of aligning a widget, comprising:"
    assert abs(sum(c["weight"] for c in out["checklist"]) - 1.0) < 1e-9
    err = out["errors"]
    assert err["n_elements"] == 4 and err["n_unsupported"] == 1 and abs(err["quote_survival"] - 0.75) < 1e-6
    assert err["per_candidate"]["inv1"]["unsupported"] == 1
    assert out["llm_calls"] == 3 and out["retry_count"] == 0 and out["self_check_ok"]
    kinds = [e["kind"] for e in out["events"]]
    assert "verified" in kinds and "self_check_pass" in kinds
    assert "METHOD CLAIM: A method of aligning a widget" in calls[2][2]


def test_self_check_failure_retries_elements_once(monkeypatch):
    calls = []
    _patch(monkeypatch, calls, checks=[{"ok": False, "issues": ["claim adds printing"], "suggestion": "drop it"},
                                       {"ok": False, "issues": ["still"], "suggestion": "x"}])
    out = asyncio.run(build_extraction_subgraph().ainvoke({"summary": "S", "document_text": DOC}))
    assert [c[0] for c in calls] == ["A1", "A2", "A5", "A2", "A5"]
    assert calls[3][2]["issues"] == ["claim adds printing"]
    assert out["retry_count"] == 2 and out["llm_calls"] == 5 and not out["self_check_ok"]
    assert out["checklist"]
    assert sum(1 for e in out["events"] if e["kind"] == "retry_applied") == 1


def test_no_invention_skips_elements(monkeypatch):
    calls = []
    _patch(monkeypatch, calls, cands={"candidate_inventions": [], "no_invention_reason": "survey"})
    out = asyncio.run(build_extraction_subgraph().ainvoke({"summary": "S", "document_text": DOC}))
    assert [c[0] for c in calls] == ["A1"]
    assert out["extraction"] == {"doc_kind": "paper", "candidate_inventions": [], "no_invention_reason": "survey"}
    assert out["checklist"] == [] and out["errors"]["llm_calls"] == 1
    assert any(e["kind"] == "no_invention" for e in out["events"])


def test_empty_elements_triggers_retry_without_self_check_call(monkeypatch):
    calls = []
    _patch(monkeypatch, calls, elements={"candidate_inventions": [], "error": "A2 response could not be parsed"})
    out = asyncio.run(build_extraction_subgraph().ainvoke({"summary": "S", "document_text": DOC}))
    assert [c[0] for c in calls] == ["A1", "A2", "A2"]
    assert out["checklist"] == [] and out["retry_count"] == 2 and out["llm_calls"] == 3


CLAIMS = ("1. A method of aligning a widget, the method comprising: predicting a translation from the widget "
          "image with an offset network; and applying the predicted translation to the widget.\n"
          "2. The method of claim 1, wherein the offset network is a convolutional network.\n"
          "3. A system comprising: an offset network; and a printer applying a translation.\n"
          "4. The system according to claim 3 wherein the printer is inkjet.\n")


def test_independent_claims_filter():
    ic = independent_claims(CLAIMS)
    assert len(ic) == 2 and ic[0].startswith("1.") and ic[1].startswith("3.")


def test_claim_prefill_texts():
    cands, prefill = claim_prefill(CLAIMS)
    assert [c["level"] for c in cands] == ["core", "component"]
    assert cands[0]["concept"].startswith("A method of aligning a widget")
    assert prefill["inv1"][0] == "A method of aligning a widget"
    assert "predicting a translation from the widget image with an offset network" in prefill["inv1"]
    assert "and applying the predicted translation to the widget" in prefill["inv1"]
    assert prefill["inv2"] == ["A system comprising:", "an offset network", "and a printer applying a translation"]


def test_claim_mode_prefills_and_skips_a1(monkeypatch):
    calls = []
    cands, prefill = claim_prefill(CLAIMS)
    fixed = prefill["inv1"]
    elements = {"candidate_inventions": [{**cands[0],
        "independent_claim_draft": {"method": "m", "system": "s"},
        "elements": [{"id": f"inv1.e{i}", "text": t, "evidence_quote": t, "facets": {}, "kind": "step"}
                     for i, t in enumerate(fixed)], "dependent_hints": []}]}
    _patch(monkeypatch, calls, elements=elements)
    out = asyncio.run(build_extraction_subgraph().ainvoke(
        {"summary": "S", "document_text": CLAIMS, "input_mode": "claim_text"}))
    assert [c[0] for c in calls] == ["A2", "A5"]
    assert calls[0][1] == prefill
    assert out["extraction"]["doc_kind"] == "patent_draft"
    assert [c["criterion"] for c in out["checklist"]] == fixed
    assert out["llm_calls"] == 2


def test_resolve_doc_text_prefers_longer_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("x" * 100)
    assert resolve_doc_text({"document_text": "short", "input_local_path": str(p)}) == "x" * 100
    assert resolve_doc_text({"document_text": "y" * 200, "input_local_path": str(p)}) == "y" * 200
    assert resolve_doc_text({"document_text": "z", "input_local_path": str(tmp_path / "missing.txt")}) == "z"


def test_should_retry_elements():
    from langgraph.graph import END
    assert should_retry_elements({"self_check_ok": True, "retry_count": 1}) == END
    assert should_retry_elements({"self_check_ok": False, "retry_count": 1}) == "elements"
    assert should_retry_elements({"self_check_ok": False, "retry_count": 2}) == END


def test_doc_json_layer_locates_with_heading_and_falls_back_to_raw_file(tmp_path, monkeypatch):
    from patent_analyzer.adapters.docjson import render_doc_json
    doc_json = {"title": "Widget alignment", "abstract": "",
                "sections": [{"heading": "1 Method", "level": 1, "paragraphs": [
                    "The offset network predicts a translation from the widget image.",
                    "Training uses an L1 loss between predicted and measured offsets."]}],
                "figures": [], "equations": [], "references_count": 0}
    rendered = render_doc_json(doc_json)
    raw = tmp_path / "in.txt"      # the raw file has one sentence Gemini dropped from the Doc JSON
    raw.write_text(rendered.replace("[S1.P1] ", "").replace("[S1.P2] ", "")
                   + "\nThe predicted translation is applied to the widget before printing.\n")
    elements = {"candidate_inventions": [{
        **ELEMENTS["candidate_inventions"][0],
        "elements": ELEMENTS["candidate_inventions"][0]["elements"][:2] + [
            {"id": "inv1.e2", "text": "applying the predicted translation to the widget before printing",
             "evidence_quote": "The predicted translation is applied to the widget before printing",
             "facets": {"thing": [], "place": [], "apparatus": []}, "kind": "step"}]}]}
    calls = []
    _patch(monkeypatch, calls, elements=elements)
    out = asyncio.run(build_extraction_subgraph().ainvoke(
        {"summary": "S", "document_text": rendered, "doc_json": doc_json, "input_local_path": str(raw),
         "input_mode": "manuscript"}))
    assert calls[0][1] == "manuscript" and calls[0][2] == rendered      # A1 read the rendered layer, not the file
    e0, e1, e2 = out["extraction"]["candidate_inventions"][0]["elements"]
    assert e1["evidence_loc"]["section"] == "S1" and e1["evidence_loc"]["para"] == 1
    assert e1["evidence_loc"]["heading"] == "1 Method" and e1["evidence_loc"]["source"] == "doc_json"
    assert not e2["unsupported"] and e2["evidence_loc"]["source"] == "fallback_text"
    assert e2["evidence_loc"]["section"] is None and e2["evidence_loc"]["char"]
    err = out["errors"]
    assert err["doc_json_hits"] == 2 and err["fallback_hits"] == 1 and err["n_unsupported"] == 0
    assert "located in Doc JSON: 2" in next(e["message"] for e in out["events"] if e["kind"] == "verified")


def test_claim_mode_is_decided_on_the_raw_file_not_the_rendered_doc_json(tmp_path, monkeypatch):
    claims = ("1. A method of aligning a widget, comprising: predicting a translation from a widget image; "
              "and applying the translation before printing.\n\n2. The method of claim 1, wherein the translation is learned.\n")
    raw = tmp_path / "claims.txt"
    raw.write_text(claims)
    doc_json = {"title": "", "abstract": "", "sections": [{"heading": "Claims", "level": 1, "paragraphs": claims.split("\n\n")}],
                "figures": [], "equations": [], "references_count": 0}
    from patent_analyzer.adapters.docjson import render_doc_json
    calls = []
    _patch(monkeypatch, calls)
    out = asyncio.run(build_extraction_subgraph().ainvoke(
        {"summary": "S", "document_text": render_doc_json(doc_json), "doc_json": doc_json, "input_local_path": str(raw)}))
    assert calls[0][0] == "A2" and calls[0][1]           # no A1 call: prefilled from the independent claim
    assert out["doc_kind"] == "patent_draft" and out["full_text"] == claims
