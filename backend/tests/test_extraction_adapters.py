import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.adapters.disclosure import doc_from_fields
from patent_analyzer.adapters.manuscript import (
    apply_flat, apply_nested, outline_flat, outline_nested, verdict_lines,
)
from patent_analyzer.adapters.paper import (
    doc_from_sections, doc_from_text, iter_paragraphs, locate_marker, render_doc,
)

PAPER = """Title: Fast Widget Alignment

Abstract
We align widgets quickly using a learned offset.

# 1 Introduction
Widgets drift. This is bad.

Prior fixes were slow.

## 1.1 Contributions
We propose a learned offset.

2 Related Work
Smith et al. aligned widgets by hand.

3 Method
The offset network predicts a translation from the widget image.
It is trained with an L1 loss.

RESULTS
Alignment error drops by 40%.
"""


def test_doc_from_text_sections_and_abstract():
    doc = doc_from_text(PAPER)
    assert doc["title"] == "Fast Widget Alignment"
    assert doc["abstract"].startswith("We align widgets quickly")
    titles = [s["title"] for s in doc["sections"]]
    assert titles == ["1 Introduction", "2 Related Work", "3 Method", "RESULTS"]
    intro = doc["sections"][0]
    assert intro["paragraphs"] == ["Widgets drift. This is bad.", "Prior fixes were slow."]
    assert intro["subsections"][0]["title"] == "1.1 Contributions"
    assert intro["subsections"][0]["paragraphs"] == ["We propose a learned offset."]
    method = doc["sections"][2]
    assert method["paragraphs"] == ["The offset network predicts a translation from the widget image. "
                                    "It is trained with an L1 loss."]
    assert doc["kind"] == "paper"


def test_doc_from_text_per_line_paragraphs_with_markers():
    lines = ["Title: X", "", "Description"] + [f"[{i:04d}] Paragraph number {i} of the description." for i in range(1, 15)]
    doc = doc_from_text("\n".join(lines))
    desc = doc["sections"][0]
    assert desc["title"] == "Description"
    assert len(desc["paragraphs"]) == 14
    assert desc["paragraphs"][2].startswith("[0003]")


def test_doc_from_text_body_sentence_starting_with_number_is_not_heading():
    doc = doc_from_text("Intro\n\n3 layers are stacked. Each layer has 64 channels.\nMore text here.")
    assert [s["title"] for s in doc["sections"]] == ["Intro"]
    assert doc["sections"][0]["paragraphs"][0].startswith("3 layers are stacked")


def test_render_doc_markers_and_locate_marker():
    doc = doc_from_text(PAPER)
    text = render_doc(doc)
    assert "[S0.P1] We align widgets" in text
    assert "[S1.P2] Prior fixes were slow." in text
    assert "[S1.1.P1] We propose a learned offset." in text
    assert "[S3.P1] The offset network" in text
    pos = text.index("trained with an L1 loss")
    assert locate_marker(text, pos) == {"section": "S3", "para": 1}
    assert locate_marker(text, text.index("Title:")) == {"section": None, "para": None}
    paras = list(iter_paragraphs(doc))
    assert paras[0] == ("S0", 1, doc["abstract"])
    assert ("S1.1", 1, "We propose a learned offset.") in paras


def test_doc_from_sections_roundtrip():
    secs = [{"title": "INTRODUCTION", "paragraphs": ["a", "", "b"],
             "subsections": [{"title": "Sub", "paragraphs": ["c"]}]}]
    doc = doc_from_sections("T", "abs", secs)
    assert doc["sections"][0]["paragraphs"] == ["a", "b"]
    assert doc["sections"][0]["subsections"][0]["subsections"] == []
    assert "[S1.1.P1] c" in render_doc(doc)


def _v(verdict, reason="r", paras=()):
    return {"verdict": verdict, "reason": reason, "prior_art_paragraphs": list(paras)}


def test_outline_nested_ids_match_render_markers():
    doc = doc_from_text(PAPER)
    items = outline_nested(doc)
    assert [(i["id"], i["heading"]) for i in items] == [
        ("S1", "1 Introduction"), ("S1.1", "1.1 Contributions"), ("S2", "2 Related Work"),
        ("S3", "3 Method"), ("S4", "RESULTS")]
    text = render_doc(doc)
    for it in items:
        for j in range(1, len(it["paragraphs"]) + 1):
            assert f"[{it['id']}.P{j}] " in text


def test_apply_nested_drops_sections_and_background_paragraphs():
    doc = doc_from_text(PAPER)
    doc["sections"][2]["subsections"].append(
        {"title": "3.1 Background and Related Work", "paragraphs": ["x"], "subsections": []})
    verdicts = {"S1": _v("mixed", "background then contribution", [2]), "S2": _v("prior_art"),
                "S3.1": _v("prior_art"), "S3": _v("own_work"), "S4": _v("own_work")}
    out = apply_nested(doc, verdicts)
    assert [s["title"] for s in out["sections"]] == ["1 Introduction", "3 Method", "RESULTS"]
    assert out["sections"][0]["paragraphs"] == ["Widgets drift. This is bad."]   # P2 was the prior-art one
    assert out["sections"][0]["subsections"][0]["title"] == "1.1 Contributions"  # mixed keeps its subsections
    assert out["sections"][1]["subsections"] == []
    assert out["abstract"] == doc["abstract"] and out["kind"] == "manuscript"    # v2 keeps the abstract
    assert out["dropped_sections"] == ["2 Related Work", "3.1 Background and Related Work"]
    assert out["dropped_paragraphs"] == [["S1", 2]]
    assert len(doc["sections"]) == 4 and len(doc["sections"][0]["paragraphs"]) == 2   # original untouched


def test_apply_nested_unjudged_section_is_kept():
    doc = doc_from_text(PAPER)
    out = apply_nested(doc, {})
    assert [s["title"] for s in out["sections"]] == [s["title"] for s in doc["sections"]]
    assert out["dropped_sections"] == [] and out["dropped_paragraphs"] == []


def test_apply_flat_cascades_into_subsections():
    doc = {"title": "T", "abstract": "a",
           "sections": [{"heading": "Background", "level": 1, "paragraphs": ["b1", "b2"]},
                        {"heading": "Gaze", "level": 2, "paragraphs": ["g"]},
                        {"heading": "Method", "level": 1, "paragraphs": ["m1", "m2"]}]}
    assert [i["id"] for i in outline_flat(doc)] == ["S1", "S1.1", "S2"]   # paths come from the levels
    out = apply_flat(doc, {"S1": _v("prior_art"), "S1.1": _v("own_work"), "S2": _v("mixed", "r", [1])})
    assert [s["heading"] for s in out["sections"]] == ["Method"]
    assert out["sections"][0]["paragraphs"] == ["m2"]
    assert out["dropped_sections"] == ["Background", "Gaze"]      # the level-2 entry went with its parent
    assert out["dropped_paragraphs"] == [["S2", 1]]
    assert out["abstract"] == "a"


def test_verdict_lines_only_reports_judged_sections():
    items = [{"id": "S1", "heading": "Intro", "paragraphs": ["a"]}, {"id": "S2", "heading": "M", "paragraphs": ["b"]}]
    lines = verdict_lines(items, {"S1": _v("mixed", "opens with citations", [1])})
    assert lines == ["S1 Intro -> mixed P[1]: opens with citations"]


def test_doc_from_fields():
    doc = doc_from_fields(problem="Widgets drift.", core_idea="A learned offset network.",
                          how_it_works="Predict translation.\n\nTrain with L1.", novelty="First learned offset.",
                          optional_variants=["Use L2 loss", "Add rotation"])
    assert doc["kind"] == "disclosure"
    assert [s["title"] for s in doc["sections"]] == ["Problem", "Core Idea", "How It Works", "Novelty", "Optional Variants"]
    assert doc["sections"][2]["paragraphs"] == ["Predict translation.", "Train with L1."]
    assert doc["dependent_hints"] == ["Use L2 loss", "Add rotation"]
    assert doc["concept_seed"] == "A learned offset network."
    text = render_doc(doc)
    assert "[S3.P2] Train with L1." in text
    assert doc_from_fields(core_idea="x")["sections"][0]["title"] == "Core Idea"


def test_doc_from_fields_benefits_and_title():
    doc = doc_from_fields(title="Offset nets", core_idea="A learned offset network.", benefits="Less drift.",
                          optional_variants=["Use L2 loss"])
    assert doc["title"] == "Offset nets"
    assert [s["title"] for s in doc["sections"]] == ["Core Idea", "Benefits", "Optional Variants"]
    assert doc["sections"][1]["paragraphs"] == ["Less drift."]
