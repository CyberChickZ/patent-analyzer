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
             "pruned": [{"pub_num": "US2"}], "funnel_docs": [{"pub_num": "US2", "rank": 3, "elements": ["inv1.e1"]}],
             "prune": {"pool": 9, "stage1_out": 5, "stage2_in": 5, "stage2_calls": 1, "stage2_worth": 2, "stage2_out": 1}}
    h = queries_html(stats)
    assert "indocyanine green" in h and "distinctive names only" in h and "from elements inv1.e1" in h
    assert "<td>3777</td><td>100</td><td>100</td><td>1</td><td>1</td><td>inv1.e1</td>" in h and "<td>120000</td><td>100</td><td>40</td><td>1</td><td>1</td>" in h
    assert "3 seed patents → 40 distinct cited" in h and "LLM read 5 abstracts in 1 calls" in h
    assert h in loop_html(stats)
    md = "\n".join(loop_md(stats))
    assert "| 1 | named | `(\"indocyanine green\" OR icg)` |" in md and "thing: perfusion map" in md


def test_reviewer_edits_are_marked_in_extraction_sections():
    from patent_analyzer.report_sections import extraction_md
    import copy
    ext = copy.deepcopy(EXT)
    ext["candidate_inventions"][0]["elements"][0]["edited_by_user"] = True
    assert "edited by reviewer" in extraction_html(ext) and "_(edited by reviewer)_" in "\n".join(extraction_md(ext))


def _docs(E):
    def d(pub, covered, unquoted=()):
        cr = {}
        for e in E:
            if e in covered:
                cr[e] = {"score": 2, "verified_quotes": ["verbatim " + e], "quote_checks": [{"quote": "verbatim " + e, "verified": True}],
                         "analysis": "found"}
            elif e in unquoted:
                cr[e] = {"score": 1, "verified_quotes": [], "quote_checks": [{"quote": "made up", "verified": False}]}
            else:
                cr[e] = {"score": 0}
        return {"pub_num": pub, "title": "Title " + pub, "checklist_results": cr, "patent_link": f"https://x/{pub}"}
    return d


def test_determination_claim_chart_rows_columns_and_cells():
    from patent_analyzer.adjudicate import adjudicate, claim_chart
    from patent_analyzer.report_sections import claim_chart_html, determination_html, determination_md
    E = ["a lens", "a mirror", "a sensor", "a housing"]
    d = _docs(E)
    docs = [d("US-A", E[:3], unquoted=[E[3]]), d("US-B", E[3:]), d("US-C", E[:1])]
    adj = adjudicate(E, docs)
    ch = claim_chart(adj, E, docs)
    h = claim_chart_html(ch)
    assert h.count("<tr>") == 1 + len(E)                                   # header + one row per element
    assert h.count("<th>") + h.count("<th ") == 1 + 3 and 'href="https://x/US-A"' in h        # element + 3 reference columns, linked
    assert "US-A</a><br>" in h and "3/4 elements" in h and "1/4 elements" in h
    rows = h.split("<tbody>")[1].split("</tr>")
    assert "Present · 1✓" in rows[0] and 'title="verbatim a lens"' in rows[0]           # covered cell shows the located quote
    assert "Partial · quote not located" in rows[3] and "#fef3c7" in rows[3]           # scored but unverified: amber, not counted
    assert rows[3].count("Present · 1✓") == 1                                          # only US-B discloses the housing
    full = determination_html(adj, ch, "")
    assert "Obviousness risk (§103)" in full and "US-A</a> — 3/4 elements: a lens; a mirror; a sensor" in full
    assert "US-B</a> — 1/4 elements: a housing" in full and "US-C" not in full.split("§103 combination relied on")[1].split("</ul>")[0]
    md = "\n".join(determination_md(adj, ch))
    assert "| a housing | Partial (quote not located) | Present 1✓ | – |" in md


def test_determination_three_verdict_wordings():
    from patent_analyzer.adjudicate import adjudicate, claim_chart
    from patent_analyzer.report_sections import determination_html
    E = ["a lens", "a mirror", "a sensor", "a housing"]
    d = _docs(E)
    one = [d("US-A", E)]
    adj = adjudicate(E, one)
    h = determination_html(adj, claim_chart(adj, E, one))
    assert "Blocking risk (§102)" in h and "Anticipating reference:" in h and "US-A</a> discloses all 4 elements" in h
    part = [d("US-A", E[:3])]
    adj = adjudicate(E, part, single_partial_103=0.7)
    h = determination_html(adj, claim_chart(adj, E, part), "Routine.")
    assert "primary reference discloses most elements" in h and "Not disclosed by any reference" in h and "a housing" in h and "<p>Routine.</p>" in h
    none = [d("US-A", E[:1]), d("US-B", E[1:2])]
    adj = adjudicate(E, none, single_partial_103=0.7)
    h = determination_html(adj, claim_chart(adj, E, none))
    assert "No blocking art found" in h and "Why no blocking art:" in h and "a sensor; a housing" in h
    assert "§103 combination relied on" not in h and "grant" not in h.split("not a prediction")[0].lower().replace("not a prediction of grant", "")
