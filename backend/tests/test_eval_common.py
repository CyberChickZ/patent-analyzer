"""Offline tests for the eval adapters (evals/common.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from common import breakdown_features, disclosed_features, norm_pub, render_doc


def test_norm_pub_fine_spelling():
    assert norm_pub("US 2008/025717 A1") == "US20080025717A1"


def test_norm_pub_bigquery_spelling():
    assert norm_pub("US-2008025717-A1") == "US20080025717A1"


def test_norm_pub_already_normalized():
    assert norm_pub("US20080025717A1") == "US20080025717A1"


def test_norm_pub_granted_and_ep():
    assert norm_pub("US 7,123,456 B2") == "US7123456B2"
    assert norm_pub("EP 1 234 567 A1") == "EP1234567A1"
    assert norm_pub("wo2019/123456a1") == "WO2019123456A1"


def test_norm_pub_empty():
    assert norm_pub("") == ""


def test_render_doc_desc_only_has_no_claims():
    doc = {"title": "T", "abstract": "Abs", "description": ["[0001] one", None, "[0002] two"],
           "claims": ["1. A thing."]}
    txt = render_doc(doc, with_claims=False)
    assert txt.startswith("Title: T\n")
    assert "[0002] two" in txt
    assert "Claims:" not in txt and "A thing" not in txt


def test_render_doc_with_claims_appends_claims():
    doc = {"title": "T", "abstract": "", "description": ["[0001] one"], "claims": ["1. A thing."]}
    txt = render_doc(doc, with_claims=True)
    assert "Abstract" not in txt
    assert txt.rstrip().endswith("Claims:\n1. A thing.")


def _app(label_docs, features, pub="US20080025717A1"):
    return {"cited_patent": {"publication_number": pub},
            "breakdown": {"prior_art_documents": label_docs, "breakdown": features}}


def _ref(doc, kind, numbers=None):
    return {"document": doc, "location": {"reference_type": kind, "numbers": numbers}}


def test_disclosed_features_uses_label_matching_cited_patent():
    app = _app(
        [{"label": "D1", "type": "paper", "identifier": "10.1000/xyz"},
         {"label": "D2", "type": "patent", "identifier": "US 2008/025717 A1"}],
        [{"feature": "a widget with a rotating shaft", "prior_art_references": [_ref("D2", "paragraph", ["[0012]"])]},
         {"feature": "a housing made of aluminium alloy", "prior_art_references": [_ref("D1", "paragraph", [3])]}])
    gold = disclosed_features(app)
    assert [g["feature"] for g in gold] == ["a widget with a rotating shaft"]
    assert gold[0]["passages"] == {("paragraph", 12)}
    assert gold[0]["paragraphs"] == [12]


def test_disclosed_features_drops_figure_only_references():
    app = _app(
        [{"label": "D1", "type": "patent", "identifier": "US20080025717A1"}],
        [{"feature": "a sensor mounted on the bracket", "prior_art_references": [_ref("D1", "figure", [2])]},
         {"feature": "a controller coupled to the sensor",
          "prior_art_references": [_ref("D1", "component", ["10"]), _ref("D1", "paragraph", ["0005-0006"])]}])
    gold = disclosed_features(app)
    assert [g["feature"] for g in gold] == ["a controller coupled to the sensor"]
    assert gold[0]["passages"] == {("paragraph", 5), ("paragraph", 6)}
    universe = breakdown_features(app)
    assert len(universe) == 2 and universe[0]["passages"] == set()


def test_disclosed_features_parses_claim_and_abstract_references():
    app = _app(
        [{"label": "D1", "type": "patent", "identifier": None}],
        [{"feature": "a valve controlled by the processor",
          "prior_art_references": [_ref("D1", "claim", [1, "3"]), _ref("D1", "abstract")]}])
    gold = disclosed_features(app)
    assert gold[0]["passages"] == {("claim", 1), ("claim", 3), ("abstract", None)}
    assert gold[0]["paragraphs"] == []


def test_load_env_yaml_sets_missing_vars_only(tmp_path, monkeypatch):
    from common import load_env_yaml
    f = tmp_path / ".env.yaml"
    f.write_text('A_KEY: "one"\nB_KEY: "two,three"  # comment\n# C_KEY: "no"\nlower: "x"\n')
    monkeypatch.setenv("A_KEY", "keep")
    monkeypatch.delenv("B_KEY", raising=False)
    assert load_env_yaml(f) == ["B_KEY"]
    import os
    assert os.environ["A_KEY"] == "keep" and os.environ["B_KEY"] == "two,three" and "C_KEY" not in os.environ
