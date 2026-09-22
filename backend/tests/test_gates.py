import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.gates import EDITABLE, PHASES, apply_response, build_interrupt

EXT = {"candidate_inventions": [
    {"id": "inv1", "level": "core", "elements": [
        {"id": "inv1.e0", "text": "A method of X", "facets": {}},
        {"id": "inv1.e1", "text": "a widget", "facets": {"thing": ["widget"]}}]},
    {"id": "inv2", "level": "application", "elements": [{"id": "inv2.e0", "text": "use of X", "facets": {}}]}]}


def test_interrupt_payload_has_editable_args_and_context():
    hi = build_interrupt("extract", {"extraction": EXT, "summary": "s", "search_results": [1]})
    assert hi["action_request"]["action"] == "review_extract"
    assert set(hi["action_request"]["args"]) == {"extraction", "_context"}
    assert hi["action_request"]["args"]["_context"] == {"summary": "s"}
    assert hi["config"]["allow_edit"] and not hi["config"]["allow_ignore"]


def test_accept_and_ignore_change_nothing():
    assert apply_response("extract", {"extraction": EXT}, {"type": "accept"}) == {}
    assert apply_response("extract", {"extraction": EXT}, None) == {}


def test_edit_extraction_marks_and_records_diff():
    import copy
    new = copy.deepcopy(EXT)
    new["candidate_inventions"][0]["elements"][1]["text"] = "a motorized widget"
    del new["candidate_inventions"][1]
    patch = apply_response("extract", {"extraction": EXT}, {"type": "edit", "args": {"extraction": new}})
    assert patch["extraction"]["candidate_inventions"][0]["elements"][1]["edited_by_user"] is True
    ops = {(e["kind"], e.get("id"), e["op"]) for e in patch["user_edits"]}
    assert ("element", "inv1.e1", "edit") in ops and ("element", "inv2.e0", "delete") in ops and ("candidate", "inv2", "delete") in ops
    assert patch["events"][0]["kind"] == "user_edit"
    assert "search_results" not in patch


def test_edit_ignores_keys_outside_the_phase_whitelist():
    patch = apply_response("search", {"ranked_candidates": [{"pub_num": "US1", "title": "a"}, {"pub_num": "US2", "title": "b"}], "search_results": [1, 2]},
                           {"type": "edit", "args": {"ranked_candidates": [{"pub_num": "US2", "title": "b"}], "search_results": [], "extraction": {}}})
    assert set(patch) == {"ranked_candidates", "user_edits", "events"}
    assert [e for e in patch["user_edits"]] == [{**patch["user_edits"][0]}] and patch["user_edits"][0]["pub_num"] == "US1" and patch["user_edits"][0]["op"] == "remove"


def test_score_edit_on_scoring_report():
    before = [{"pub_num": "US1", "title": "t", "checklist_results": {"c1": {"score": 2}, "c2": {"score": 0}}}]
    after = [{"pub_num": "US1", "title": "t", "checklist_results": {"c1": {"score": 1}, "c2": {"score": 0}}}]
    patch = apply_response("evaluate", {"scoring_report": before}, {"type": "edit", "args": {"scoring_report": after}})
    e = patch["user_edits"][0]
    assert (e["kind"], e["criterion_id"], e["before"], e["after"]) == ("score", "c1", 2, 1)
    assert patch["scoring_report"][0]["checklist_results"]["c1"]["edited_by_user"] is True
    assert all(k in EDITABLE for k in ("idca", "extract", "search", "evaluate"))


DRAFT = {"claims": [
    {"no": 1, "form": "method", "depends_on": None, "preamble": "A method, comprising:",
     "limitations": [{"lid": "c1.l1", "text": "receiving a signal"}, {"lid": "c1.l2", "text": "filtering the signal"}]},
    {"no": 2, "form": "method", "depends_on": 1, "preamble": "The method of claim 1, wherein",
     "limitations": [{"lid": "c2.l1", "text": "the signal is ultra-wideband"}]}]}


def test_draft_phase_is_gated_and_limitation_edits_are_recorded():
    import copy
    assert "draft" in PHASES and EDITABLE["draft"] == ("draft_claims",)
    hi = build_interrupt("draft", {"draft_claims": DRAFT, "adjudication": {"label": "102"}, "scoring_report": [1]})
    assert set(hi["action_request"]["args"]) == {"draft_claims", "_context"}
    assert hi["action_request"]["args"]["_context"] == {"adjudication": {"label": "102"}}
    new = copy.deepcopy(DRAFT)
    new["claims"][0]["limitations"][1]["text"] = "low-pass filtering the signal"
    del new["claims"][1]
    patch = apply_response("draft", {"draft_claims": DRAFT}, {"type": "edit", "args": {"draft_claims": new}})
    assert patch["draft_claims"]["claims"][0]["limitations"][1]["edited_by_user"] is True
    ops = {(e["kind"], e.get("id"), e["op"]) for e in patch["user_edits"]}
    assert ("limitation", "c1.l2", "edit") in ops and ("limitation", "c2.l1", "delete") in ops and ("claim", "2", "delete") in ops
    assert apply_response("draft", {"draft_claims": DRAFT}, {"type": "edit", "args": {"draft_claims": copy.deepcopy(DRAFT)}}) == {}
