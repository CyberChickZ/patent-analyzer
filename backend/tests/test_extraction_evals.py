import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

import extraction_fulldoc_eval as fde
import pap2pat_extraction_eval as p2p


def test_independent_claims_and_gold_claims():
    patent = {"claims": [
        "1. A widget aligner comprising: an offset network configured to predict a translation; and a printer applying the translation.",
        "2. The aligner of claim 1, wherein the network is convolutional.",
        "3. A method of aligning a widget, comprising: predicting a translation with an offset network; and applying the translation.",
        "4. The method according to claim 3 wherein the widget is round.",
        "not a claim",
    ]}
    assert [n for n, _ in p2p.independent_claims(patent["claims"])] == [1, 3]
    gold = p2p.gold_claims(patent)
    assert [g["claim_no"] for g in gold] == [1, 3]
    assert gold[0]["elements"][0]["text"] == "A widget aligner comprising:"
    assert all(e["level"] == "core" for e in gold[0]["elements"])
    assert all(e["level"] is None for e in gold[1]["elements"])
    assert not any(e["text"].startswith("3.") for e in gold[1]["elements"])


def test_coverage_per_claim(monkeypatch):
    vocab = ["offset network", "printer", "predicting", "applying", "round"]

    def fake_embed(texts):
        out = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, w in enumerate(vocab):
                out[i, j] = float(w in t.lower())
            out[i] /= np.linalg.norm(out[i]) or 1.0
        return out
    monkeypatch.setattr(p2p, "embed", fake_embed)
    claims = [{"claim_no": 1, "elements": [{"text": "an offset network"}, {"text": "a printer"}]},
              {"claim_no": 3, "elements": [{"text": "predicting"}, {"text": "applying"}]}]
    cv = p2p.coverage(["the offset network", "the printer", "predicting a translation"], claims)
    assert abs(cv["recall"] - 0.75) < 1e-6 and cv["full"] == 0.5 and cv["n_gold"] == 4 and cv["n_pred"] == 3
    assert p2p.coverage([], claims) == {"recall": 0.0, "full": 0.0, "n_gold": 4, "n_pred": 0}


def test_candidate_and_checklist_elements():
    rec = {"extraction": {"candidate_inventions": [
        {"id": "inv1", "level": "core", "elements": [
            {"id": "inv1.e0", "text": "a", "evidence_quote": "q", "kind": "step", "unsupported": False},
            {"id": "inv1.e1", "text": "b", "evidence_quote": "", "kind": "step", "unsupported": True}]},
        {"id": "inv2", "level": "component", "elements": [
            {"id": "inv2.e0", "text": "c", "evidence_quote": "q", "kind": "structure", "unsupported": False}]},
    ]}, "checklist": [{"criterion": "x"}, {"criterion": ""}]}
    assert [e["id"] for e in p2p.candidate_elements(rec, 1)] == ["inv1.e0"]
    assert [e["id"] for e in p2p.candidate_elements(rec, None, supported_only=False)] == ["inv1.e0", "inv1.e1", "inv2.e0"]
    assert p2p.candidate_elements(rec, 2)[-1]["level"] == "component"
    assert p2p.checklist_elements(rec) == [{"id": "c0", "text": "x"}]
    assert [e["id"] for e in fde.core_elements(rec)] == ["inv1.e0", "inv1.e1"]
    assert fde.core_elements({"extraction": None}) == []


def test_errors_for_dispatch(monkeypatch):
    seen = {}

    def fake_classify(preds, gold, doc, require_quote=True, **kw):
        seen["preds"], seen["rq"] = preds, require_quote
        return {}
    monkeypatch.setattr(fde, "classify_errors", fake_classify)
    fde.errors_for({"checklist": [{"criterion": "x"}]}, ["g"], "doc")
    assert seen["rq"] is False and seen["preds"] == [{"id": "c0", "text": "x"}]
    fde.errors_for({"extractor": "new", "extraction": {"candidate_inventions": [
        {"id": "inv1", "level": "core", "elements": [{"id": "inv1.e0", "text": "t", "evidence_quote": "q", "kind": "step"}]}]},
        "checklist": []}, ["g"], "doc")
    assert seen["rq"] is True and seen["preds"][0]["level"] == "core"
