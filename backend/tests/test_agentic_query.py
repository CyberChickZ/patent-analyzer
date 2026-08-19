import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.query_gen import boolean_query, next_mode
from patent_analyzer.agentic.validator import validate

EL = {"id": "e1", "facets": {"thing": ["gaze estimation", "eye tracking"], "place": ["video conferencing", "telepresence"],
                              "apparatus": ["camera", "head mount"]}}


def test_strict_query_scoped_with_hint():
    q = boolean_query(EL, "strict")
    assert q.startswith('AB=(((gaze estimation) OR (eye tracking)) ((video conferencing) OR telepresence) (camera OR (head mount)))')
    assert "NEAR/10" not in q  # multi-word first terms: no hint
    q2 = boolean_query({"facets": {"thing": ["gaze"], "place": ["telepresence"], "apparatus": []}}, "strict", cpc="H04N7")
    assert q2 == "AB=(gaze telepresence) (gaze NEAR/10 telepresence) CPC=H04N7/low"   # CPC clause last


def test_loose_and_core():
    assert boolean_query(EL, "loose") == 'AB=(((gaze estimation) OR (eye tracking)) ((video conferencing) OR telepresence))'
    assert boolean_query(EL, "core") == '((gaze estimation) OR (eye tracking))'
    assert boolean_query({"facets": {}}, "strict") == ""


def test_multiword_forms_are_parenthesized_not_quoted():
    # unquoted words get Google's stemming/synonyms; "phrase" would be exact;
    # parentheses stop AND's left associativity from eating the OR
    el = {"facets": {"thing": ["sound damping", "vibration absorber"], "place": ["engine"], "apparatus": []}}
    assert boolean_query(el, "loose", field="") == "((sound damping) OR (vibration absorber)) engine"
    assert '"' not in boolean_query(el, "strict")


def test_field_switch_and_dedupe_cap():
    el = {"facets": {"thing": list("aabcdefghij"), "place": [], "apparatus": []}}
    assert boolean_query(el, "loose", field="CL") == "CL=((a OR b OR c OR d OR e OR f OR g OR h))"


def test_validator_and_mode_walk():
    assert validate(0, 0) == "zero" and validate(None, 0) == "zero"
    assert validate(50_000, 100) == "too_broad"
    assert validate(120, 20) == "ok"
    assert next_mode("strict", "zero") == "loose"
    assert next_mode("loose", "too_narrow") == "core"
    assert next_mode("core", "zero") is None
    assert next_mode("loose", "too_broad") == "strict"
    assert next_mode("strict", "too_broad") is None
    assert next_mode("strict", "ok") is None


def test_term_never_repeated_across_facets():
    el = {"facets": {"thing": ["pressure sensor"], "place": ["housing"], "apparatus": ["pressure sensor", "transducer"]}}
    assert boolean_query(el, "strict", field="") == "(pressure sensor) housing transducer"
    el = {"facets": {"thing": ["accelerometer"], "place": ["housing"], "apparatus": ["Accelerometer"]}}
    assert boolean_query(el, "strict", field="") == "accelerometer housing (accelerometer NEAR/10 housing)"


def test_cpc_clause_needs_a_main_group():
    from patent_analyzer.agentic.query_gen import cpc_clause
    assert cpc_clause("h04n7/15") == "CPC=H04N7/low" and cpc_clause("H04N7") == "CPC=H04N7/low"
    assert cpc_clause("h04n") == "" and cpc_clause("") == ""   # subclass-level CPC= matches nothing (gold probe)


def test_named_terms_are_required_in_strict_and_dropped_in_loose():
    el = {"facets": {"named": ["soluble adenylyl cyclase", "sAC", "sac"], "thing": ["proliferation inhibition", "sac"],
                     "place": ["prostate cancer"], "apparatus": ["kh7"]}}
    q = boolean_query(el, "strict", field="")
    assert q.startswith('("soluble adenylyl cyclase" OR sAC) (proliferation inhibition) (prostate cancer) kh7')
    assert boolean_query(el, "loose", field="") == "(proliferation inhibition) (prostate cancer)"
    el2 = {"facets": {"named": [], "thing": ["gaze"], "place": ["telepresence"], "apparatus": []}}
    assert boolean_query(el2, "strict", field="") == "gaze telepresence (gaze NEAR/10 telepresence)"
