from patent_analyzer.adjudicate import adjudicate, element_covered, greedy_cover

E = ["a first sensor", "a controller coupled to the sensor", "wherein the controller halts the motor",
     "a display showing the halt state"]


def _doc(pub, covered, partial=(), unquoted=()):
    cr = {}
    for e in E:
        if e in covered:
            cr[e] = {"score": 2, "verified_quotes": ["some verbatim text here"]}
        elif e in partial:
            cr[e] = {"score": 1, "verified_quotes": ["partial verbatim text here"]}
        elif e in unquoted:
            cr[e] = {"score": 2, "evidence_quotes": ["made up"], "verified_quotes": []}
        else:
            cr[e] = {"score": 0, "verified_quotes": []}
    return {"pub_num": pub, "title": pub, "checklist_results": cr}


def test_single_doc_full_coverage_is_102():
    out = adjudicate(E, [_doc("A", E), _doc("B", E[:2])])
    assert out["label"] == "102"
    assert out["risk"] == "blocking"
    assert out["best_single"] == "A"
    assert out["per_doc_coverage"][0]["coverage"] == 1.0


def test_union_of_two_docs_is_103_and_single_is_not():
    out = adjudicate(E, [_doc("A", E[:3]), _doc("B", E[2:])])
    assert out["label"] == "103"
    assert set(out["combo"]["docs"]) == {"A", "B"}
    assert out["combo"]["n_covered"] == 4
    assert out["per_doc_coverage"][0]["coverage"] == 0.75


def test_unverified_quote_does_not_count():
    # score 2 but no located quote: not evidence -> the element stays missing
    out = adjudicate(E, [_doc("A", E[:3], unquoted=[E[3]])])
    assert out["label"] == "ALLOW"
    assert out["per_doc_coverage"][0]["missing"] == [E[3]]
    assert not element_covered({"score": 2, "verified_quotes": []})
    assert element_covered({"score": 2, "verified_quotes": []}, require_quotes=False)


def test_allow_missing_and_min_cover_relax_the_single_reference_rule():
    docs = [_doc("A", E[:3])]
    assert adjudicate(E, docs)["label"] == "ALLOW"
    assert adjudicate(E, docs, allow_missing=1)["label"] == "102"
    assert adjudicate(E, docs, min_cover=0.75)["label"] == "102"
    assert adjudicate(E, docs, min_cover=0.8)["label"] == "ALLOW"  # floor(4*0.2)=0 slack
    # partial (score 1) counts by default, not with min_score=2
    part = [_doc("A", E[:3], partial=[E[3]])]
    assert adjudicate(E, part)["label"] == "102"
    assert adjudicate(E, part, min_score=2)["label"] == "ALLOW"
    # PANORAMA rule (a): a strong primary reference alone gives 103
    assert adjudicate(E, docs, single_partial_103=0.7)["label"] == "103"


def test_greedy_cover_prefers_fewest_docs_and_caps_combo():
    sets = [("A", {"1", "2"}), ("B", {"3"}), ("C", {"1", "2", "3"}), ("D", {"4"})]
    docs, covered = greedy_cover(["1", "2", "3", "4"], sets, needed=4, max_combo=3)
    assert docs == ["C", "D"] and covered == {"1", "2", "3", "4"}
    docs, covered = greedy_cover(["1", "2", "3", "4"], [("A", {"1"}), ("B", {"2"}), ("D", {"3"}), ("E", {"4"})],
                                 needed=4, max_combo=3)
    assert len(docs) == 3 and len(covered) == 3
    out = adjudicate(E, [_doc("A", E[:1]), _doc("B", E[1:2]), _doc("C", E[2:3]), _doc("D", E[3:])])
    assert out["label"] == "ALLOW" and out["risk"] == "related"  # 3 docs cap: 3/4 union, 1/4 single
    assert adjudicate([], [_doc("A", E)])["label"] == "ALLOW"
    assert adjudicate(E, [])["label"] == "ALLOW" and adjudicate(E, [])["risk"] == "related"


def test_report_section_states_blocking_risk_only():
    from patent_analyzer.adjudicate import claim_chart
    from patent_analyzer.report_sections import determination_html, determination_md, inject_html, inject_md
    docs = [_doc("US-1", E[:3]), _doc("US-2", E[2:])]
    adj = adjudicate(E, docs)
    ch = claim_chart(adj, E, docs)
    html = determination_html(adj, ch, "Because.\n\nNothing cuts against it.")
    md = "\n".join(determination_md(adj, ch, "Because."))
    assert "Prior-Art Determination" in html and "§103 screening flag" in html and "blocking" in html
    assert "US-1</a>" not in html and "US-1 — 3/4 elements" in html and "US-2 — 2/4 elements" in html   # no url: plain names
    assert "<p>Because.</p><p>Nothing cuts against it.</p>" in html
    assert "§103 screening flag" in md and "US-1 (3/4)" in md and "US-2 (2/4)" in md and "Because." in md
    for text in (html, md):
        low = text.lower()
        assert "grantable" not in low and "would be granted" not in low and "patentable" not in low
        assert "novelty score" not in low and "innovation landscape" not in low
    assert determination_html(None) == "" and determination_md({}) == []
    base = '<div class="sec"><div class="sec-t">Invention Summary</div><div class="sec-b">S</div>\n</div>'
    assert "Prior-Art Determination" in inject_html(base, None, None, None, None, adj)
    already = base + determination_html(adj, ch)
    assert inject_html(already, None, None, None, None, adj).count("Prior-Art Determination") == 1
    out = inject_md("## Invention Summary\n\nS\n\n## Evaluation Criteria\n\nN\n", None, None, None, None, adj)
    assert out.index("Prior-Art Determination") < out.index("## Evaluation Criteria")
    assert "## Prior-Art Determination" in inject_md("## Invention Summary\n\nS\n\n## Novelty Assessment\n", None, None, None, None, adj)


def test_reduce_eval_emits_adjudication_without_touching_scores(monkeypatch):
    import asyncio
    import app.llm as llm
    from graph.eval_subgraph import reduce_eval

    async def _boom(*a, **k):
        raise AssertionError("reduce_eval must not call the LLM")
    monkeypatch.setattr(llm, "call_llm", _boom)
    monkeypatch.setattr(llm, "generate_overall_summary", _boom)
    monkeypatch.setattr(llm, "generate_combination_analysis", _boom)
    checklist = [{"criterion": e, "weight": 0.25} for e in E]
    state = {"checklist": checklist, "summary": "s", "eval_results": [_doc("US-1", E), _doc("US-2", E[:2])]}
    out = asyncio.run(reduce_eval(state))
    assert out["adjudication"]["label"] == "102" and out["adjudication"]["basis"] == "single"
    assert out["overall_summary"] == "" and out["combination_analysis"] == ""   # legacy keys kept, no LLM
    assert out["eval_stats"]["adjudication"]["risk"] == "blocking"
    assert out["novelty_score"] == 0.0 and out["risk_level"] == out["risk_level"]
    assert out["scoring_report"][0]["similarity_score"] == 1.0


def _chart(docs):
    from patent_analyzer.adjudicate import claim_chart
    adj = adjudicate(E, docs, single_partial_103=0.7)
    return adj, claim_chart(adj, E, docs)


def test_claim_chart_102_puts_the_anticipating_reference_first():
    adj, ch = _chart([_doc("B", E[:2]), _doc("A", E)])
    assert adj["label"] == "102"
    assert [d["key"] for d in ch["docs"]] == ["A", "B"]           # best single leads, then next by coverage
    assert ch["n_elements"] == 4 and ch["uncovered"] == []
    assert [r["element"] for r in ch["rows"]] == E                  # rows in element order
    assert all(r["cells"][0]["covered"] and r["cells"][0]["n_verified"] == 1 for r in ch["rows"])
    assert ch["rows"][3]["covered_by"] == ["A"] and ch["rows"][3]["cells"][1]["score"] == 0
    assert ch["rows"][0]["cells"][0]["quote"] == "some verbatim text here"


def test_claim_chart_103_columns_are_the_rule_combination():
    adj, ch = _chart([_doc("C", E[:1]), _doc("A", E[:3]), _doc("B", E[2:]), _doc("D", E[1:2])])
    assert adj["label"] == "103" and set(adj["combo"]["docs"]) == {"A", "B"}
    assert [d["key"] for d in ch["docs"]][:2] == adj["combo"]["docs"] and len(ch["docs"]) == 3
    assert ch["rows"][3]["covered_by"] == ["B"] and ch["rows"][2]["covered_by"] == ["A", "B"]
    assert ch["uncovered"] == []
    # partial primary reference (>=70%, not all): one lead column, the gap is listed
    adj, ch = _chart([_doc("A", E[:3])])
    assert adj["label"] == "103" and adj["combo"] is None or len(adj["combo"]["docs"]) < 2
    assert [d["key"] for d in ch["docs"]] == ["A"] and ch["uncovered"] == [E[3]]


def test_claim_chart_allow_marks_unverified_cells_as_not_covered():
    adj, ch = _chart([_doc("A", E[:1], unquoted=[E[1]]), _doc("B", E[2:3])])
    assert adj["label"] == "ALLOW"
    cell = ch["rows"][1]["cells"][0]
    assert cell["score"] == 2 and cell["n_verified"] == 0 and not cell["covered"]
    assert ch["uncovered"] == [E[1], E[3]] and ch["docs"][0]["n_covered"] == 1
    assert ch["rows"][2]["covered_by"] == ["B"]
    # references that disclose nothing are not chart columns
    adj, ch = _chart([_doc("A", E[:1]), _doc("Z", []), _doc("Y", [])])
    assert [d["key"] for d in ch["docs"]] == ["A"]
    adj, ch = _chart([_doc("Z", [])])
    assert ch["docs"] == [] and ch["rows"][0]["cells"] == [] and ch["uncovered"] == E


def test_explain_obviousness_feeds_the_rule_output_and_only_runs_for_103(monkeypatch):
    import asyncio
    import app.llm as llm
    from patent_analyzer.adjudicate import claim_chart
    seen = {}

    async def _fake(system, user, **k):
        seen["system"], seen["user"] = system, user
        return "Reference 1 supplies the sensor and controller; reference 2 supplies the display.\n\nNothing cuts against it."
    monkeypatch.setattr(llm, "call_llm", _fake)
    docs = [_doc("US-1", E[:3]), _doc("US-2", E[2:])]
    docs[0]["key_teachings"] = "a sensor-driven motor controller"
    adj = adjudicate(E, docs)
    ch = claim_chart(adj, E, docs)
    out = asyncio.run(llm.explain_obviousness(adj, ch, "an invention", docs))
    assert out.startswith("Reference 1 supplies")
    u = seen["user"]
    assert "may NOT change the determination" in u and adj["reason"] in u
    assert "Reference 1: US-1" in u and "Reference 2: US-2" in u and "sensor-driven motor controller" in u
    assert f"- {E[3]} [some verbatim text here]" in u and "(none — every element is disclosed" in u
    seen.clear()
    adj102 = adjudicate(E, [_doc("US-1", E)])
    assert asyncio.run(llm.explain_obviousness(adj102, claim_chart(adj102, E, docs), "x", docs)) == "" and not seen


def test_basis_names_which_rule_fired():
    assert adjudicate(E, [_doc("A", E)])["basis"] == "single"
    assert adjudicate(E, [_doc("A", E[:3]), _doc("B", E[2:])])["basis"] == "combination"
    assert adjudicate(E, [_doc("A", E[:3])], single_partial_103=0.7)["basis"] == "primary_partial"
    assert adjudicate(E, [_doc("A", E[:1])])["basis"] == "none" and adjudicate([], [])["basis"] == "none"


# ── one coverage rule, one source ──
#
# The report's evidence matrix used to call a cell green on its own terms (every
# quote verified) while the determination used element_covered (min_score AND a
# verified quote). Same page, two answers: "best single covers 4/5" beside a
# badge reading "1/5 covered". adjudicate now emits the decision per cell and
# everything downstream reads it.

def test_cells_agree_with_element_covered_and_with_the_covered_list():
    from patent_analyzer.adjudicate import coverage_cells, element_covered
    els = ["e1", "e2", "e3", "e4"]
    cr = {"e1": {"score": 2, "verified_quotes": ["q"]},      # covered
          "e2": {"score": 2},                                 # scored, no quote -> NOT covered
          "e3": {"score": 0, "verified_quotes": ["q"]},       # not disclosed
          }                                                   # e4 absent -> not evaluated
    adj = adjudicate(els, [{"pub_num": "US-1", "checklist_results": cr}])
    row = adj["per_doc_coverage"][0]
    cells = row["cells"]
    assert set(cells) == set(els)
    for name in els:
        assert cells[name]["covered"] is element_covered(cr.get(name)), name
    assert {k for k, v in cells.items() if v["covered"]} == set(row["covered"])
    assert {k for k, v in cells.items() if not v["covered"]} == set(row["missing"])
    assert row["n_covered"] == sum(1 for v in cells.values() if v["covered"]) == 1
    assert coverage_cells(els, cr) == cells


def test_cell_reasons_name_the_test_that_failed():
    from patent_analyzer.adjudicate import coverage_cells
    cells = coverage_cells(["a", "b", "c", "d"], {
        "a": {"score": 2, "verified_quotes": ["q"]},
        "b": {"score": 2},
        "c": {"score": 0},
    }, min_score=2)
    assert cells["a"]["reason"] == ""
    assert cells["b"]["reason"] == "quote_not_verified"
    assert cells["c"]["reason"] == "not_disclosed"
    assert cells["d"]["reason"] == "not_evaluated" and cells["d"]["evaluated"] is False
    low = coverage_cells(["a"], {"a": {"score": 1, "verified_quotes": ["q"]}}, min_score=2)
    assert low["a"]["reason"] == "below_min_score"


def test_matrix_and_determination_cannot_disagree():
    """The regression that started this: a cell the matrix paints green must be a
    cell the determination counts as covered."""
    from patent_analyzer.report_sections import quote_matrix_html
    els = ["e1", "e2"]
    # e2 is scored 2 with all quote_checks verified, but carries no verified_quotes,
    # so element_covered says no. The old matrix painted it green regardless.
    docs = [{"pub_num": "US-1", "title": "D", "checklist_results": {
        "e1": {"score": 2, "verified_quotes": ["q"], "quote_checks": [{"quote": "q", "verified": True}]},
        "e2": {"score": 2, "quote_checks": [{"quote": "x", "verified": True}]}}}]
    adj = adjudicate(els, docs)
    cl = [{"criterion": "e1"}, {"criterion": "e2"}]
    h = quote_matrix_html(docs, cl, adjudication=adj)
    green = h.count("#dcfce7")
    assert adj["per_doc_coverage"][0]["n_covered"] == 1
    assert green == 1, f"matrix painted {green} cells green; the determination counts 1"


def test_rule_trace_names_the_mpep_section_and_admits_what_it_cannot_find():
    """The §103 branches are screening flags, and the trace has to say so
    requirement by requirement: coverage cannot establish a motivation to
    combine (MPEP 2143.01) or a reasonable expectation of success (2143.02)."""
    docs = [_doc("US-1", E[:2]), _doc("US-2", E[2:])]
    adj = adjudicate(E, docs, single_partial_103=0.7)
    by = {t["id"]: t for t in adj["rule_trace"]}
    assert adj["label"] == "103" and adj["prima_facie"] is False
    assert by["graham_a"]["mpep"] == "2141 II (A)" and by["graham_a"]["status"] == "met"
    assert by["graham_b"]["mpep"] == "2141 II (B)" and by["graham_b"]["status"] == "met"
    assert by["graham_c"]["status"] == "not_determined"          # level of ordinary skill
    assert by["graham_objective"]["status"] == "not_determined"  # secondary considerations
    assert by["rationale_a_1"]["mpep"] == "2143 I.A (1)" and by["rationale_a_1"]["status"] == "met"
    for k in ("rationale_a_2", "rationale_a_3", "motivation", "expectation"):
        assert by[k]["status"] == "not_determined", k
    assert by["motivation"]["mpep"] == "2143.01" and by["expectation"]["mpep"] == "2143.02 I"
    assert by["hindsight"]["mpep"] == "2142" and by["hindsight"]["status"] == "met"


def test_a_102_trace_does_not_ask_for_analogous_art():
    """MPEP 2131.05: non-analogous art is not germane to anticipation."""
    adj = adjudicate(E, [_doc("US-1", E)])
    by = {t["id"]: t for t in adj["rule_trace"]}
    assert adj["label"] == "102"
    assert by["anticipation"]["mpep"] == "2131" and by["analogous_not_required"]["mpep"] == "2131.05"
    assert not any(t["id"].startswith("analogous:") for t in adj["rule_trace"])


def test_analogous_art_proxy_never_says_not_analogous():
    """A shared CPC subclass is a proxy for one of the two independent tests in
    MPEP 2141.01(a) I. Failing it is not a finding that a reference is
    non-analogous — "reasonably pertinent to the problem" is a separate test
    that a classification cannot answer."""
    from patent_analyzer.adjudicate import analogous_art
    assert analogous_art({"cpc_codes": ["H04N   7/15"]}, ["H04N   7/14"])[0] == "met"
    assert analogous_art({"cpc_codes": ["G06T   7/00"]}, ["H04N   7/14"])[0] == "not_determined"
    assert analogous_art({}, ["H04N   7/14"])[0] == "not_determined"
    assert analogous_art({"cpc_codes": ["G06T   7/00"]}, None)[0] == "not_determined"


def test_a_103_trace_carries_one_analogous_row_per_reference_relied_on():
    docs = [_doc("US-1", E[:2]), _doc("US-2", E[2:])]
    docs[0]["cpc_codes"] = ["H04N   7/15"]
    adj = adjudicate(E, docs, single_partial_103=0.7, invention_cpc=["H04N   7/14"])
    rows = [t for t in adj["rule_trace"] if t["id"].startswith("analogous:")]
    assert {t["id"] for t in rows} == {"analogous:US-1", "analogous:US-2"}
    assert all(t["mpep"] == "2141.01(a) I" for t in rows)
    assert next(t for t in rows if t["id"] == "analogous:US-1")["status"] == "met"
