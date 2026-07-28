import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.extraction_subgraph import claim_prefill, independent_claims, resolve_doc_text

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


def test_resolve_doc_text_prefers_longer_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("x" * 100)
    assert resolve_doc_text({"document_text": "short", "input_local_path": str(p)}) == "x" * 100
    assert resolve_doc_text({"document_text": "y" * 200, "input_local_path": str(p)}) == "y" * 200
    assert resolve_doc_text({"document_text": "z", "input_local_path": str(tmp_path / "missing.txt")}) == "z"
