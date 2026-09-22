import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.draft.definiteness import (auto_fix, check, check_antecedent, check_exemplary, check_functional,
                                                check_relative, introduced_nps, open_flags)


def _claim(preamble, *lims, no=1, depends_on=None, quotes=None):
    return {"no": no, "form": "method", "depends_on": depends_on, "preamble": preamble,
            "limitations": [{"lid": f"c{no}.l{i}", "text": t, "basis": [{"evidence_quote": (quotes or {}).get(i, "")}]}
                            for i, t in enumerate(lims, 1)]}


def test_antecedent_mpep_2173_05e_examples():
    # "said lever" with no earlier lever
    flags = check_antecedent("moving said lever", {})
    assert [f["kind"] for f in flags] == ["no_antecedent"] and flags[0]["span"] == "said lever"
    # two levers recited earlier, then "the lever"
    intro = introduced_nps("a first lever and a second lever")
    flags = check_antecedent("rotating the lever", intro)
    assert [f["kind"] for f in flags] == ["ambiguous_antecedent"]
    assert check_antecedent("rotating the first lever", dict(intro)) == []
    # "said aluminum lever" when only "a lever" was recited
    intro = introduced_nps("a lever")
    flags = check_antecedent("pressing said aluminum lever", intro)
    assert flags and flags[0]["kind"] == "no_antecedent" and "only 'lever'" in flags[0]["note"]
    # plural-of quantifiers, 'of' continuation and bare plurals introduce their heads
    intro = introduced_nps("a plurality of sensors coupled to a swarm of drones emitting pulses")
    assert check_antecedent("reading the sensors of the drones from the pulses", intro) == []
    # an introduction later in the same limitation does not count; the claim-reference is whitelisted
    assert check_antecedent("The method of claim 1, wherein the filter is a filter", {})[0]["kind"] == "no_antecedent"


def test_relative_exemplary_functional_rules():
    assert check_relative("heating the fluid to a substantially uniform temperature")[0]["span"] == "substantially"
    assert check_relative("heating the fluid to a substantially uniform temperature", ["within 2% of 300 K"]) == []
    assert check_relative("a threshold of about 5 ms") == []
    assert check_exemplary("a lossy codec, such as JPEG")[0]["span"] == "such as"
    assert check_exemplary("a lossy codec") == []
    f = check_functional("a module for processing the frames")
    assert f and f[0]["category"] == "functional_claiming" and "module" in f[0]["note"]
    assert check_functional("means for damping vibration")[0]["span"] == "means for"
    assert check_functional("a convolutional encoder configured to encode the frames") == []
    assert check_functional("one or more processors configured to encode the frames") == []


def test_check_walks_parents_then_claim_and_auto_fix_repairs_safe_cases():
    parent = _claim("A method of routing packets, comprising:", "receiving a packet at a first node",
                    "forwarding the packet to a second node")
    assert check(parent) == []
    child = _claim("The method of claim 1, wherein", "the packet is encrypted, such as with AES, before the hop", no=2, depends_on=1)
    flags = check(child, parents=[parent])
    cats = {(f["lid"], f["category"]) for f in flags}
    assert ("c2.l1", "antecedent_basis") in cats and ("c2.l1", "exemplary_phrasing") in cats
    child, flags = auto_fix(child, flags)
    assert child["limitations"][0]["text"] == "the packet is encrypted before a hop"
    assert all(f["fixed"] for f in flags) and open_flags(flags) == []
    assert check(child, parents=[parent]) == []
    bad = _claim("A method, comprising:", "a mechanism for sorting the items quickly")
    flags = check(bad)
    assert {f["category"] for f in flags} == {"functional_claiming", "antecedent_basis"}
    bad, flags = auto_fix(bad, flags)
    assert [f["category"] for f in open_flags(flags)] == ["functional_claiming"]
    assert bad["limitations"][0]["text"] == "a mechanism for sorting a items quickly" or "items" in bad["limitations"][0]["text"]
