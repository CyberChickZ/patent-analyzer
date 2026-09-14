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
