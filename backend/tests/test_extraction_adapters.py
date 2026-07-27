import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.adapters.disclosure import doc_from_fields
from patent_analyzer.adapters.manuscript import strip_related_work
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


def test_strip_related_work():
    doc = doc_from_text(PAPER)
    doc["sections"][2]["subsections"].append(
        {"title": "3.1 Background and Related Work", "paragraphs": ["x"], "subsections": []})
    out = strip_related_work(doc)
    assert [s["title"] for s in out["sections"]] == ["1 Introduction", "3 Method", "RESULTS"]
    assert out["sections"][1]["subsections"] == []
    assert out["abstract"] == "" and out["kind"] == "manuscript"
    assert out["dropped_sections"] == ["2 Related Work", "3.1 Background and Related Work"]
    assert doc["abstract"]  # original untouched
    assert len(doc["sections"]) == 4


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
