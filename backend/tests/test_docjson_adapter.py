import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.adapters.docjson import (
    doc_json_stats, iter_doc_json_paragraphs, locate_marker, render_doc_json, section_paths,
)

DOC = {
    "title": "Fast Widget Alignment",
    "abstract": "We align widgets quickly using a learned offset.",
    "sections": [
        {"heading": "1 Introduction", "level": 1, "paragraphs": ["Widgets drift.", "Prior fixes were slow."]},
        {"heading": "1.1 Contributions", "level": 2, "paragraphs": ["We propose a learned offset."]},
        {"heading": "3 Method", "level": 1, "paragraphs": ["The offset network predicts a translation from the widget image.",
                                                            "It is trained with an L1 loss $L = |d - \\hat d|$."]},
    ],
    "figures": [{"label": "Figure 1", "caption": "Overview of the offset network."}],
    "equations": [{"label": "1", "latex": "L = |d - \\hat d|"}],
    "references_count": 12,
}


def test_section_paths_from_levels():
    assert section_paths([{"level": 1}, {"level": 2}, {"level": 2}, {"level": 3}, {"level": 1}, {"level": 3}]) == \
        ["S1", "S1.1", "S1.2", "S1.2.1", "S2", "S2.1"]
    assert section_paths([{"level": 2}]) == ["S1"]


def test_render_markers_headings_figures_equations():
    text = render_doc_json(DOC)
    assert text.startswith("Title: Fast Widget Alignment\n\nAbstract\n[S0.P1] We align widgets")
    assert "# 1 Introduction\n[S1.P1] Widgets drift.\n\n[S1.P2] Prior fixes were slow." in text
    assert "## 1.1 Contributions\n[S1.1.P1] We propose" in text
    assert "# 3 Method\n[S2.P1] The offset network" in text
    assert "# Figures\n[S3.P1] Figure 1: Overview of the offset network." in text
    assert "# Equations\n[S4.P1] (1) $L = |d - \\hat d|$" in text
    pos = text.find("predicts a translation")
    assert locate_marker(text, pos) == {"section": "S2", "para": 1}
    plain = render_doc_json(DOC, markers=False)
    assert "[S" not in plain and "The offset network predicts" in plain


def test_iter_paragraphs_and_stats():
    rows = list(iter_doc_json_paragraphs(DOC))
    assert rows[0] == ("S0", "Abstract", 1, DOC["abstract"])
    assert rows[3] == ("S1.1", "1.1 Contributions", 1, "We propose a learned offset.")
    assert rows[-1][1] == "Equations"
    st = doc_json_stats(DOC)
    assert st["sections"] == 3 and st["paragraphs"] == 8 and st["figures"] == 1 and st["equations"] == 1
    assert st["references_count"] == 12
    assert doc_json_stats(None)["paragraphs"] == 0


def test_clean_doc_json_coerces_model_output():
    from app.llm import _clean_doc_json
    raw = {"title": "  A\n B ", "abstract": None,
           "sections": [{"heading": "1 X", "level": "2", "paragraphs": [" p1 ", "", None]},
                        {"heading": "", "level": 9, "paragraphs": []}, "junk"],
           "figures": [{"label": "Fig 1", "caption": ""}, {"label": "Fig 2", "caption": "c"}],
           "equations": [{"label": "1", "latex": " x=y "}], "references_count": "7"}
    d = _clean_doc_json(raw)
    assert d["title"] == "A B" and d["abstract"] == ""
    assert d["sections"] == [{"heading": "1 X", "level": 2, "paragraphs": ["p1"]}]
    assert d["figures"] == [{"label": "Fig 2", "caption": "c"}]
    assert d["equations"] == [{"label": "1", "latex": "x=y"}] and d["references_count"] == 7


def test_cut_flat_drops_section_and_its_subsections():
    import asyncio

    import app.llm as llm
    from patent_analyzer.adapters.manuscript import cut_flat
    doc = {**DOC, "sections": DOC["sections"][:1] + [
        {"heading": "2 Related Work", "level": 1, "paragraphs": ["Smith aligned by hand."]},
        {"heading": "2.1 Prior widgets", "level": 2, "paragraphs": ["Old widgets."]},
    ] + DOC["sections"][2:]}

    async def fake(title, sections):
        assert [s["id"] for s in sections] == ["S1", "S2", "S2.1", "S3"]
        return {"S2": {"verdict": "prior_art", "reason": "Smith's work", "prior_art_paragraphs": []}}

    real, llm.classify_prior_art_sections = llm.classify_prior_art_sections, fake
    try:
        out, verdicts = asyncio.run(cut_flat(doc))
    finally:
        llm.classify_prior_art_sections = real
    assert [s["heading"] for s in out["sections"]] == ["1 Introduction", "3 Method"]
    assert out["dropped_sections"] == ["2 Related Work", "2.1 Prior widgets"]
    assert verdicts["S2"]["verdict"] == "prior_art"
    assert doc["sections"][1]["heading"] == "2 Related Work"   # input untouched
    assert "Smith" not in render_doc_json(out)
