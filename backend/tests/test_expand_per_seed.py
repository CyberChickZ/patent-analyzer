import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.expand import _per_seed


def _realistic():
    """wg4's shape: ~10,000 seeds, a pool far bigger than the cap, and the gold
    cited by ONE seed while the field's textbooks are cited by thousands."""
    by_seed, cited = {}, Counter()
    by_seed["FOCUSED"] = [f"GOLD{i}" for i in range(4)]
    for i in range(4):
        cited[f"GOLD{i}"] = 1
    for s in range(10_000):                       # noise seeds, each citing shared textbooks
        picks = [f"TEXTBOOK{(s + j) % 8000}" for j in range(6)]
        by_seed[f"N{s}"] = picks
        for p in picks:
            cited[p] += 1
    return by_seed, cited


def test_a_focused_seed_is_not_outvoted_by_ten_thousand_others():
    by_seed, cited = _realistic()
    new, info = _per_seed(by_seed, cited, set(), {}, 5000)
    assert info["per_seed_seeds"] == 10_001
    assert info["per_seed_k"] == 3                        # ceil(5000/10001) -> floor of 3
    # with 10,001 seeds for 5,000 slots the split degenerates to one each for the
    # first 5,000 in caller order — but the focused seed gets that one, which the
    # global vote never gave it
    got = [p for p in new if p.startswith("GOLD")]
    assert len(got) == 1, "the focused seed must still get its slot"
    assert info["per_seed_served"] <= 5000
    # under the old global rule those documents are nowhere near the cut
    old_rank = min(i for i, (p, _) in enumerate(cited.most_common()) if p.startswith("GOLD"))
    assert old_rank > 5000, f"gold ranked {old_rank} globally — inside the cap, test is not realistic"


def test_k_scales_with_the_seed_count_and_stays_inside_its_bounds():
    small = {f"S{i}": [f"P{i}_{j}" for j in range(80)] for i in range(10)}
    cited = Counter({p: 1 for ps in small.values() for p in ps})
    _, info = _per_seed(small, cited, set(), {}, 5000)
    assert info["per_seed_k"] == 50                       # ceil(5000/10)=500, capped at 50

    many = {f"S{i}": [f"P{i}"] for i in range(20_000)}
    _, info = _per_seed(many, Counter({f"P{i}": 1 for i in range(20_000)}), set(), {}, 5000)
    assert info["per_seed_k"] == 3                        # ceil(5000/20000)=1, floored at 3


def test_known_and_already_fetched_documents_are_skipped():
    by_seed = {"S": ["A", "B", "C"]}
    new, _ = _per_seed(by_seed, Counter({"A": 5, "B": 3, "C": 1}), {"A"}, {"B": {}}, 5000)
    assert new == ["C"]


def test_the_callers_seed_order_decides_who_gets_a_slot_when_there_are_too_many():
    """loop.py builds seeds query hits first, then the Reliance bridge, then
    Lens. A query's own hit is a better place to walk from than the eight
    thousandth bridge patent, and with more seeds than slots that ordering is
    the only thing that decides."""
    by_seed, cited = {}, Counter()
    for s in range(6000):
        by_seed[f"N{s}"] = [f"NOISE{s}"]
        cited[f"NOISE{s}"] = 9
    by_seed["FOCUSED"] = ["GOLD"]                     # last by insertion order
    cited["GOLD"] = 1
    assert "GOLD" not in _per_seed(by_seed, cited, set(), {}, 5000)[0]
    assert "GOLD" in _per_seed(by_seed, cited, set(), {}, 5000, order=["FOCUSED"])[0]


def test_no_seeds_is_not_a_crash():
    assert _per_seed({}, Counter(), set(), {}, 5000) == ([], {"per_seed_k": 0, "per_seed_seeds": 0})


def test_budget_scales_with_seeds_instead_of_staying_at_five_thousand():
    from patent_analyzer.agentic.expand import MAX_CITED_LIGHT, budget_for
    assert budget_for(200, MAX_CITED_LIGHT) == 5000        # small jobs unchanged
    assert budget_for(1400, MAX_CITED_LIGHT) == 7000       # 5 per seed
    assert budget_for(43_762, MAX_CITED_LIGHT) == 20_000   # capped, metadata fetch stays in guard


def test_a_query_seeds_own_references_survive_ten_thousand_bridge_seeds():
    """wg4's shape. The gold sits in ONE query seed's references, cited once,
    while 6,000 textbooks are cited nine times each by the bridge seeds. Under
    the old global vote it was nowhere near the 5,000 cut; a query seed now
    takes ten of its own whatever k works out to."""
    by_seed, cited, kind, order = {}, Counter(), {}, []
    for q in range(1200):
        refs = [f"Q{q}_{j}" for j in range(40)]
        by_seed[f"QS{q}"] = refs
        kind[f"QS{q}"] = "query"
        order.append(f"QS{q}")
        for p in refs:
            cited[p] += 1
    by_seed["QS0"] = ["GOLD_A", "GOLD_B"] + by_seed["QS0"]
    cited["GOLD_A"] = cited["GOLD_B"] = 1
    for b in range(9000):
        refs = [f"TEXTBOOK{(b + j) % 6000}" for j in range(6)]
        by_seed[f"BS{b}"] = refs
        kind[f"BS{b}"] = "bridge"
        order.append(f"BS{b}")
        for p in refs:
            cited[p] += 1

    from patent_analyzer.agentic.expand import MAX_CITED_LIGHT, budget_for
    budget = budget_for(len(by_seed), MAX_CITED_LIGHT)
    new, info = _per_seed(by_seed, cited, set(), {}, budget, order=order, kind=kind)
    assert budget == 20_000 and info["per_seed_k"] == 3
    assert {"GOLD_A", "GOLD_B"} <= set(new), "a query seed's own references must survive"
    old_rank = min(i for i, (p, _) in enumerate(cited.most_common()) if p.startswith("GOLD"))
    assert old_rank > 5000, f"gold ranked {old_rank} globally — the old rule would have kept it"
