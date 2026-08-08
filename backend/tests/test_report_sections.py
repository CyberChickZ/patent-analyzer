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


def test_every_query_table_explains_how_each_query_was_built():
    from patent_analyzer.report_sections import loop_html, queries_html
    stats = {"loop_rounds": [{"round": 1, "mode": "wide", "n_queries": 2, "gp_calls": 0, "serpapi_calls": 2, "seeds": 3, "expanded": 4,
                              "cited_total": 40, "pool_size": 9, "covered": [], "uncovered": [], "queries": [
                                  {"n": 1, "kind": "named", "query": '("indocyanine green" OR icg)', "facets_used": {"named": ["indocyanine green", "icg"]},
                                   "elements": ["inv1.e1"], "channel": "serpapi_patents", "total": 3777, "hits": 100, "new": 100, "pubs": ["US1", "US2"]},
                                  {"n": 2, "kind": "thing+place", "query": "((perfusion map)) hindlimb", "facets_used": {"thing": ["perfusion map"], "place": ["hindlimb"]},
                                   "elements": ["inv1.e0", "inv1.e1"], "channel": "serpapi_patents", "total": 120000, "hits": 100, "new": 40, "pubs": ["US2", "US3"]}]}],
             "loop_elements": [{"id": "inv1.e0", "text": "a"}, {"id": "inv1.e1", "text": "b"}],
             "pruned": [{"pub_num": "US2"}], "funnel_docs": [{"pub_num": "US2", "rank": 3}],
             "prune": {"pool": 9, "stage1_out": 5, "stage2_in": 5, "stage2_calls": 1, "stage2_worth": 2, "stage2_out": 1}}
    h = queries_html(stats)
    assert "indocyanine green" in h and "distinctive names only" in h and "from elements inv1.e1" in h
    assert "<td>3777</td><td>100</td><td>100</td><td>1</td><td>1</td>" in h and "<td>120000</td><td>100</td><td>40</td><td>1</td><td>1</td>" in h
    assert "3 seed patents → 40 distinct cited" in h and "LLM read 5 abstracts in 1 calls" in h
    assert h in loop_html(stats)
    md = "\n".join(loop_md(stats))
    assert "| 1 | named | `(\"indocyanine green\" OR icg)` |" in md and "thing: perfusion map" in md
