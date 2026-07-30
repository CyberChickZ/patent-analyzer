import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.report_sections import extraction_html, inject_html, inject_md, loop_md, quote_matrix_html

EXT = {"candidate_inventions": [{"id": "inv1", "level": "core", "concept": "kinetic proxy", "cpc_pred": ["H04N7"],
                                 "independent_claim_draft": {"method": "A method comprising: a; b."},
                                 "elements": [{"id": "e1", "text": "a gaze sensor", "evidence_quote": "the sensor tracks gaze",
                                               "evidence_loc": {"para": 3}},
                                              {"id": "e2", "text": "made up", "evidence_quote": "", "unsupported": True}]}]}
STATS = {"loop_rounds": [{"round": 1, "n_queries": 3, "gp_calls": 0, "serpapi_calls": 3, "gp_blocked": 1, "seeds": 5,
                          "expanded": 9, "pool_size": 14, "covered": ["e1"], "uncovered": ["e2"], "cpc_hint": "H04N"}],
         "loop_elements": [{"id": "e1", "text": "a gaze sensor"}, {"id": "e2", "text": "made up"}],
         "serpapi_quota": [{"key": "abcd1234", "used": 3, "cap": 250}]}
SR = [{"pub_num": "US1B2", "title": "Doc one", "checklist_results": {
    "a gaze sensor": {"score": 2, "quote_checks": [{"quote": "q1", "verified": True}, {"quote": "q2", "verified": False}]},
    "made up": {"score": 0}}}]
CL = [{"criterion": "a gaze sensor"}, {"criterion": "made up"}]


def test_extraction_html_marks_unsupported():
    h = extraction_html(EXT)
    assert "Candidate Inventions" in h and "1/2 elements grounded" in h and "unsupported" in h
    assert "the sensor tracks gaze" in h and "Draft method claim" in h


def test_matrix_cells():
    h = quote_matrix_html(SR, CL)
    assert "1/2" in h and "#fef3c7" in h and "does not predict grant" in h


def test_inject_after_summary_and_md_before_assessment():
    base = '<html><div class="sec">\n  <div class="sec-t">Invention Summary</div>\n  <div class="sec-b">S</div>\n</div>\n\n<div class="sec eval-sec">X</div></html>'
    out = inject_html(base, EXT, STATS, SR, CL)
    assert out.index("Candidate Inventions") < out.index('<div class="sec eval-sec">')
    assert out.index("Invention Summary") < out.index("Candidate Inventions")
    md = inject_md("## Invention Summary\n\nS\n\n## Novelty Assessment\n\nN\n", EXT, STATS, SR, CL)
    assert md.index("## Candidate Inventions") < md.index("## Novelty Assessment")
    assert "| 1 | 3 | 0 | 3 | 5 | +9 | 14 | 1/2 |" in "\n".join(loop_md(STATS))
