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
