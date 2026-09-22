import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.report_sections import draft_html, extraction_html, inject_html, inject_md, loop_md, quote_matrix_html

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
    assert "§103 screening flag" in full and "US-A</a> — 3/4 elements: a lens; a mirror; a sensor" in full
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
    assert "One document already shows everything (§102, anticipation)" in h
    assert "Anticipating reference:" in h and "US-A</a> discloses all 4 elements" in h
    part = [d("US-A", E[:3])]
    adj = adjudicate(E, part, single_partial_103=0.7)
    h = determination_html(adj, claim_chart(adj, E, part), "Routine.")
    assert "Best single document shows most of it (≥70% — screening flag)" in h
    assert "Not disclosed by any reference" in h and "a housing" in h and "<p>Routine.</p>" in h
    none = [d("US-A", E[:1]), d("US-B", E[1:2])]
    adj = adjudicate(E, none, single_partial_103=0.7)
    h = determination_html(adj, claim_chart(adj, E, none))
    assert "No document or combination we read shows all the elements" in h
    assert "What is missing:" in h and "a sensor; a housing" in h
    assert "§103 combination relied on" not in h and "grant" not in h.split("not a prediction")[0].lower().replace("not a prediction of grant", "")


def test_the_verdict_comes_before_the_statutes_not_after_them():
    """Harry, 2026-09-20, on job 99a35c00: the card opened with a paragraph
    carrying BRI, 2131, 2141, 2143, 2143.01, 2143.02 and PANORAMA C.5.3, and the
    answer was underneath it. The MPEP text is still in the report — folded."""
    from patent_analyzer.adjudicate import adjudicate, claim_chart
    from patent_analyzer.report_sections import coverage_lines, determination_html, determination_md
    E = ["a lens", "a mirror", "a sensor", "a housing"]
    d = _docs(E)
    docs = [d("US-A", E[:1]), d("US-B", E[1:2])]
    adj = adjudicate(E, docs, single_partial_103=0.7)
    adj["construction"] = "bri"          # BRI is on by default in production
    ch = claim_chart(adj, E, docs)
    h = determination_html(adj, ch)

    verdict = h.index("No document or combination we read shows all the elements")
    for section in ("MPEP 2111", "MPEP 2131", "MPEP 2141", "2143.01", "2143.02", "C.5.3"):
        assert section in h, section
        assert h.index(section) > verdict, f"{section} sits above the verdict"
        assert h.index(section) > h.index("How this was decided (for attorneys)"), section
    assert "<details" in h and "Rule output, verbatim:" in h
    # the two numbers, on the card itself
    assert coverage_lines(adj) == ["Best single document: 1 of 4 elements.",
                                   "Best combination (up to 3 documents): 2 of 4."]
    for line in coverage_lines(adj):
        assert line in h and h.index(line) < h.index("How this was decided")
    # the machine sentence is not what the reader meets first
    assert h.index("best single coverage") > h.index("How this was decided")

    md = "\n".join(determination_md(adj, ch))
    mv = md.index("No document or combination we read shows all the elements")
    assert md.index("### How this was decided (for attorneys)") > mv
    for line in coverage_lines(adj):
        assert f"- {line}" in md and md.index(line) < md.index("### How this was decided")
    for section in ("MPEP 2111", "MPEP 2131", "2143.02"):
        assert md.index(section) > mv, section


def test_the_cell_legend_is_above_the_table():
    """Under the table it explains "Partial · 1✓" to somebody who has already
    scrolled past every one of them."""
    from patent_analyzer.adjudicate import adjudicate, claim_chart
    from patent_analyzer.report_sections import claim_chart_html
    E = ["a lens", "a mirror"]
    d = _docs(E)
    docs = [d("US-A", E[:1])]
    h = claim_chart_html(claim_chart(adjudicate(E, docs), E, docs))
    assert h.index("How to read this:") < h.index("<table")
    assert h.count("quote not located") >= 1


def _results(E, docs, adj, chart, explanation=""):
    adj = dict(adj, claim_chart=chart, obviousness_explanation=explanation)
    return {"job_id": "t", "source_title": "T", "phase1": {"summary": "S", "doc_mode": "paper", "invention_type": "Process"},
            "phase2": {"checklist": [{"criterion": e, "weight": 0.25} for e in E]},
            "search": {"summary": {"total_patents": 3, "total_papers": 0}}, "adjudication": adj,
            "evaluation": {"scoring_report": [dict(d, similarity_score=0.3, ewss=0.3, css=0.1, similarity_categories=d["checklist_results"])
                                              for d in docs],
                           "summary": "OLD LLM NOVELTY TEXT", "combination_analysis": "OLD COMBO TEXT",
                           "stats": {"top_score": 0.3, "risk_level": "medium"}}}


def test_generate_html_leads_with_the_determination_and_drops_the_old_novelty_chain():
    from patent_analyzer.adjudicate import adjudicate, claim_chart
    from patent_analyzer.report_generator import generate_html
    E = ["a lens", "a mirror", "a sensor", "a housing"]
    d = _docs(E)
    docs = [d("US-A", E[:3]), d("US-B", E[3:])]
    adj = adjudicate(E, docs)
    ch = claim_chart(adj, E, docs)
    h = generate_html(_results(E, docs, adj, ch, "Because.\n\nNothing cuts against it."))
    assert h.index('class="sec det-sec"') < h.index("Invention Summary")
    assert "Two or three documents together show everything" in h and "§103 screen — a flag for review" in h
    assert "<p>Because.</p>" in h and 'class="tbl claim-chart"' in h
    low = h.lower()
    for gone in ("innovation landscape", "novelty score", "novelty assessment", "combination analysis", "ewss", "css=",
                 "old llm novelty text", "old combo text"):
        assert gone not in low, gone
    # The card names both quantities. "100%" next to "does not disclose the
    # invention" read as a contradiction until it said which 100% it was
    # (Harry, 2026-09-20): coverage is the one the determination rests on, so
    # it leads, and relevance is labelled as the judge's confidence.
    assert 'title="3/4 elements disclosed with a located verbatim quote">Covers 3 of 4 elements</span>' in h
    assert 'title="1/4 elements disclosed with a located verbatim quote">Covers 1 of 4 elements</span>' in h
    assert "Relevance " in h and "hit-rel" in h
    assert h.index("US-A") < h.index("US-B")            # ordered by verified coverage
    # the export script reads the determination, not the removed section
    assert ".det-sec .det-verdict" in h and ".eval-sec" not in h
    # no adjudication (e.g. Absent -> Report): page still renders, no determination block
    h0 = generate_html(_results(E, docs, {}, None))
    assert 'class="sec det-sec"' not in h0 and "Invention Summary" in h0


def test_generate_markdown_leads_with_the_determination_and_drops_scores():
    from patent_analyzer.adjudicate import adjudicate, claim_chart
    from patent_analyzer.report_generator import generate_markdown
    from patent_analyzer.report_sections import inject_md
    E = ["a lens", "a mirror", "a sensor", "a housing"]
    d = _docs(E)
    docs = [d("US-A", E)]
    adj = adjudicate(E, docs)
    md = generate_markdown(_results(E, docs, adj, claim_chart(adj, E, docs)))
    assert md.startswith("# Prior Art Search Report: T")
    assert "**Determination:** One document already shows everything (§102, anticipation)" in md
    assert md.index("## Prior-Art Determination") < md.index("## Invention Summary") < md.index("## Evaluation Criteria")
    assert "| 1 | Title US-A | Doc | 4/4 |" in md and "**Elements disclosed (quote located):** 4/4" in md
    low = md.lower()
    for gone in ("risk level", "top score", "css", "ewss", "novelty", "old llm", "old combo"):
        assert gone not in low, gone
    # injected sections land before Evaluation Criteria and the determination is not duplicated
    out = inject_md(md, EXT, STATS, SR, CL, adj)
    assert out.count("## Prior-Art Determination") == 1
    assert out.index("## Candidate Inventions") < out.index("## Evaluation Criteria")


def test_report_node_builds_chart_and_calls_the_explanation_only_for_103(monkeypatch, tmp_path):
    import asyncio
    import json
    import app.llm as llm
    from nodes import report as report_mod
    from patent_analyzer.adjudicate import adjudicate
    calls = []

    async def _fake(system, user, **k):
        calls.append(user)
        return "Reasoning paragraph."
    monkeypatch.setattr(llm, "call_llm", _fake)
    monkeypatch.setattr(report_mod, "_save_to_gcs", lambda *a, **k: None)
    E = ["a lens", "a mirror", "a sensor", "a housing"]
    d = _docs(E)
    docs = [d("US-A", E[:3]), d("US-B", E[3:])]
    adj = adjudicate(E, docs)
    state = {"job_id": "j1", "output_dir": str(tmp_path), "summary": "S", "checklist": [{"criterion": e} for e in E],
             "scoring_report": docs, "eval_stats": {"adjudication": adj}, "overall_summary": "", "combination_analysis": ""}
    out = asyncio.run(report_mod.report_node(state))
    assert out["status"] == "completed" and len(calls) == 1 and "Reference 1: US-A" in calls[0]
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["adjudication"]["label"] == "103" and res["adjudication"]["obviousness_explanation"] == "Reasoning paragraph."
    assert [x["pub_num"] for x in res["adjudication"]["claim_chart"]["docs"]] == ["US-A", "US-B"]
    assert set(res["evaluation"]) >= {"summary", "combination_analysis", "stats"}     # legacy keys kept
    html = (tmp_path / "report.html").read_text()
    assert html.count('class="sec det-sec"') == 1 and "<p>Reasoning paragraph.</p>" in html
    assert "Innovation Landscape" not in html and "novelty score" not in html.lower()
    md = (tmp_path / "report.md").read_text()
    assert md.count("## Prior-Art Determination") == 1 and "Reasoning paragraph." in md
    # §102: no LLM call at all
    calls.clear()
    adj = adjudicate(E, [d("US-A", E)])
    state.update(scoring_report=[d("US-A", E)], eval_stats={"adjudication": adj}, output_dir=str(tmp_path / "b"))
    asyncio.run(report_mod.report_node(state))
    assert calls == [] and json.loads((tmp_path / "b" / "results.json").read_text())["adjudication"]["obviousness_explanation"] == ""


def test_lens_attribution_only_when_lens_was_used():
    from patent_analyzer.report_sections import lens_attribution_html
    h = lens_attribution_html({"loop_rounds": [{"lens": {"bridge_patents": 800, "search_calls": 5}}]})
    assert "Data Sourced from The Lens" in h and 'href="https://www.lens.org"' in h and "<img" in h
    assert lens_attribution_html({"loop_rounds": [{"lens": {"bridge_patents": 0, "search_calls": 0}}]}) == ""
    assert lens_attribution_html(None) == ""


DRAFT = {"candidate_id": "inv1", "primary_form": "method", "strategy": "narrowed",
         "avoidance": {"reason": "US1B2 discloses every element (2/2, §102); the independent claim adds a limitation from dependent_hint[0] that none of the 1 charted references (US1B2) discloses."},
         "claims": [
             {"no": 1, "form": "method", "depends_on": None, "preamble": "A method of tracking gaze, comprising:", "limitations": [
                 {"lid": "c1.l1", "text": "sensing a gaze direction", "origin": "element",
                  "basis": [{"element_id": "e1", "evidence_quote": "the sensor tracks gaze", "evidence_loc": {"para": 3, "section": "2.1"}}],
                  "coverage": {"covered_by": ["US1B2"], "checked_against": ["US1B2"], "verified": True}, "flags": []},
                 {"lid": "c1.l2", "text": "wherein the sensor samples at 120 Hz", "origin": "dependent_hint", "distinguishing": True,
                  "basis": [{"element_id": "hint[0]", "evidence_quote": "sampled at 120 Hz", "evidence_loc": {"char": [10, 30], "method": "exact"}}],
                  "coverage": {"covered_by": [], "checked_against": ["US1B2"], "verified": True,
                               "recheck": {"queried": True, "new_docs": 2, "covered_by_new": []}},
                  "flags": [{"lid": "c1.l2", "category": "relative_term", "span": "high", "rule": "MPEP 2173.05(b)", "fixed": False, "note": "no standard"}]}]},
             {"no": 2, "form": "method", "depends_on": 1, "preamble": "The method of claim 1, further comprising", "limitations": [
                 {"lid": "c2.l1", "text": "logging the gaze direction", "origin": "component_element",
                  "basis": [{"element_id": "inv2.e1", "evidence_quote": "a log of gaze", "evidence_loc": None}],
                  "coverage": {"covered_by": [], "checked_against": [], "verified": False, "status": "unknown"}, "flags": []}]}],
         "definiteness": {"passes": 2, "flags": [{"lid": "c1.l2", "category": "relative_term", "span": "high", "rule": "MPEP 2173.05(b)", "fixed": False, "note": "no standard"},
                                                 {"lid": "c2.l1", "category": "exemplary_phrasing", "span": "such as", "rule": "MPEP 2173.05(d)", "fixed": True}],
                          "open_flags": [{"lid": "c1.l2", "category": "relative_term", "span": "high"}],
                          "llm_advisory": {"1": {"likelihood": "unlikely", "p_indefinite": 0.2, "reasons": [{"category": "undefined_term", "claim_recitations": ["gaze direction"]}]}}},
         "recheck": {"queries": [{"query": "(gaze sensor) (eye tracker)", "channel": "google_patents", "total": 120, "new": 2}],
                     "new_docs": [{"pub_num": "US7B2", "title": "Eye tracker", "url": "http://x", "text_mode": "google_patents_page", "covered": []}], "evaluated": 1}}


def test_draft_html_six_blocks():
    from patent_analyzer.report_sections import draft_html
    h = draft_html(DRAFT, EXT)
    assert "Draft Claims (for attorney review)" in h and "does not predict grant" in h
    assert "narrowed" in h and "adds a limitation from dependent_hint[0]" in h                  # 1 strategy
    assert "<b>1.</b> A method of tracking gaze, comprising:" in h and "sensing a gaze direction;" in h  # 2 claims
    assert "disclosed by US1B2" in h and "not disclosed by charted refs" in h and "unknown" in h and "re-check: 2 new docs, none disclose" in h
    assert "★" in h and 'background:#fee2e2' in h                                                # distinguishing + open flag in red
    assert "<q>the sensor tracks gaze</q>" in h and "section 2.1, para 3" in h and "char" not in h.split("Basis")[1][:400]  # 3 basis
    assert "112(b) self-check" in h and "2 rule flag(s), 1 open after 2 pass(es)" in h and "undefined_term: gaze direction" in h  # 4
    assert "(gaze sensor) (eye tracker)" in h and "US7B2" in h                                    # 5 re-check
    assert "Pre-search draft (A2)" in h and "A method comprising: a; b." in h                     # 6 A2 draft
    assert draft_html({}, EXT) == "" and "No grounded elements" in draft_html({"strategy": "no_elements"}, EXT)


def test_draft_md_and_injection():
    from patent_analyzer.report_sections import draft_md
    md = "\n".join(draft_md(DRAFT, EXT))
    assert md.startswith("## Draft Claims (for attorney review)") and "**Strategy: narrowed**" in md
    assert "- sensing a gaze direction; and [^1]" in md and "not disclosed by charted refs" in md and "112(b) open: relative_term" in md
    assert "| 1 | c1.l1 | e1 | the sensor tracks gaze |" in md and "| c1.l2 | relative_term | high | MPEP 2173.05(b) | OPEN |" in md
    assert "`(gaze sensor) (eye tracker)`" in md
    unresolved = {**DRAFT, "strategy": "unresolved", "avoidance": {"reason": "needs input from the inventor"}}
    assert "Unresolved — every candidate limitation is disclosed" in draft_html(unresolved, EXT)
    base_html = '<div class="sec"><div class="sec-t">Invention Summary</div><div class="sec-b">s</div>\n</div>\n<div class="sec">## Evaluation Criteria</div>'
    assert "Draft Claims (for attorney review)" in inject_html(base_html, EXT, STATS, SR, CL, draft=DRAFT)
    assert "Draft Claims" not in inject_html(base_html, EXT, STATS, SR, CL)
    assert "## Draft Claims (for attorney review)" in inject_md("# R\n## Evaluation Criteria\n", EXT, STATS, SR, CL, draft=DRAFT)


def test_react_steps_show_their_observation_and_decision():
    from patent_analyzer.report_sections import queries_html
    h = queries_html({"loop_rounds": [{"queries": [
        {"n": 1, "kind": "react", "query": "(a) (b) CPC=H04N7/low", "total": 123, "hits": 100, "new": 100, "pubs": [],
         "observation": "titles say swiveling monitor", "decision": "use the learned word", "cpc_group": "H04N7", "cpc_forced": "H04N7"}]}]})
    assert "saw:</b> titles say swiveling monitor" in h and "chose:</b> use the learned word" in h
    assert "H04N7" in h and "forced" in h
