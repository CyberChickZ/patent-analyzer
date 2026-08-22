import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes.claim_mode import _parse_claim_limitations
from patent_analyzer.draft.assemble import (assemble_independent, dependent_claim, gerund_to_infinitive, mirror_preamble,
                                            recut, render_claim, verify_wording)

CAND = {"id": "inv1", "primary_form": "method", "elements": [
    {"id": "inv1.e0", "text": "A method of coordinating a swarm of drones", "kind": "structure", "evidence_quote": "q0", "evidence_loc": {"para": 1}},
    {"id": "inv1.e1", "text": "broadcasting, by each drone, a state embedding vector via ultra-wideband pulses", "kind": "step",
     "evidence_quote": "q1", "evidence_loc": {"para": 2}},
    {"id": "inv1.e2", "text": "deriving a local phase from an interference pattern of received pulses", "kind": "step", "evidence_quote": "q2"},
    {"id": "inv1.e3", "text": "the embedding vector has 64 dimensions", "kind": "parameter", "evidence_quote": "q3"},
    {"id": "inv1.e4", "text": "made up element", "kind": "step", "evidence_quote": "", "unsupported": True},
]}


def test_assemble_method_from_elements_with_basis_and_recut_count():
    c = assemble_independent(CAND, "method", claim_no=1)
    assert c["preamble"] == "A method of coordinating a swarm of drones, comprising:"
    assert [l["lid"] for l in c["limitations"]] == ["c1.l1", "c1.l2", "c1.l3"]      # unsupported e4 never enters
    assert c["limitations"][2]["text"].startswith("wherein ")
    assert c["limitations"][0]["basis"] == [{"element_id": "inv1.e1", "evidence_quote": "q1", "evidence_loc": {"para": 2}}]
    assert all(l["origin"] == "element" for l in c["limitations"])
    text = render_claim(c)
    assert text.startswith("1. A method of coordinating a swarm of drones, comprising:\n  broadcasting")
    assert "; and\n  wherein" in text and text.endswith(".")
    assert len(_parse_claim_limitations(text)["limitations"]) == 3


def test_mirror_system_form_fallback_wording():
    c = assemble_independent(CAND, "system", claim_no=5)
    assert c["preamble"] == "A system for coordinating a swarm of drones, comprising:"
    assert c["limitations"][0]["text"] == "one or more processors configured to broadcast, by each drone, a state embedding vector via ultra-wideband pulses"
    assert c["limitations"][2]["text"].startswith("wherein ")
    assert mirror_preamble("An apparatus for damping sound", "method") == "A method of damping sound"
    assert gerund_to_infinitive("Computing a hash") == "compute a hash"
    assert gerund_to_infinitive("stopping the motor") == "stop the motor"
    assert gerund_to_infinitive("training a model") == "train a model"


def test_dependent_claim_bridges_and_render():
    parent = assemble_independent(CAND, "method", claim_no=1)
    d = dependent_claim(parent, {"text": "wherein the pulses are emitted at 10 Hz", "kind": "condition", "origin": "dependent_hint", "basis": []}, claim_no=2)
    assert d["depends_on"] == 1 and d["preamble"] == "The method of claim 1, wherein"
    assert render_claim(d) == "2. The method of claim 1, wherein the pulses are emitted at 10 Hz."
    d2 = dependent_claim(parent, {"text": "logging the phase", "kind": "step", "basis": []}, claim_no=3)
    assert render_claim(d2) == "3. The method of claim 1, further comprising logging the phase."


def test_verify_wording_invariant():
    rule = ["broadcasting a vector", "deriving a phase from received pulses", "wherein the vector has 64 dimensions"]
    sims = {("broadcasting a vector", "transmitting the vector"): 0.9,
            ("deriving a phase from received pulses", "deriving a phase, wherein the pulses are received"): 0.95,
            ("wherein the vector has 64 dimensions", "wherein the vector is a 64-dimensional vector and is normalised"): 0.7}
    texts, rep = verify_wording(rule, ["transmitting the vector", "deriving a phase, wherein the pulses are received",
                                       "wherein the vector is a 64-dimensional vector and is normalised"],
                                similarity=lambda a, b: [sims[(x, y)] for x, y in zip(a, b)])
    assert texts == ["transmitting the vector", rule[1], rule[2]]
    assert [e["accepted"] for e in rep["per_limitation"]] == [True, False, False]
    assert rep["per_limitation"][1]["reason"] == "splits on recut" and rep["accepted"] == 1
    texts, rep = verify_wording(rule, ["one", "two"], similarity=lambda a, b: [1.0] * len(a))
    assert texts == rule and not rep["count_ok"]
    assert len(recut(rule)) == 3
