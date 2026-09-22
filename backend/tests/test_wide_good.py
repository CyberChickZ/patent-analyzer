import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic import wide_good as WG
from patent_analyzer.recall.pool import Candidate

ELS = [{"id": "e1", "text": "a pair of cameras"}, {"id": "e2", "text": "an infrared marker"}]


def test_delivery_order_puts_strong_first_then_specificity():
    """(strong, elements touched, how many were pinned to a claim number, cosine).
    A paper points with a located quote rather than a number, so the numeric
    count is 'how often a patent said it in its own claims'."""
    docs = [
        {"pub_num": "WEAK", "good": True, "good_touches": {"e1": 3}, "prune_cos": 0.9},
        {"pub_num": "STRONG_QUOTES", "good": True, "good_touches": {"e1": "we used two cameras",
                                                                    "e2": "a reflective marker"}},
        {"pub_num": "STRONG_CLAIMS", "good": True, "good_touches": {"e1": 1, "e2": 4}},
        {"pub_num": "NOT_GOOD", "good": False, "good_touches": {}},
    ]
    assert [d["pub_num"] for d in WG.rank_for_delivery(docs)] == \
        ["STRONG_CLAIMS", "STRONG_QUOTES", "WEAK"]


def test_judge_and_deliver_reads_a_budget_of_the_whole_pool_and_never_prunes_on_abstracts(monkeypatch):
    seen = {}

    def fake_select(elements, cands, n, summary=""):
        seen["offered"] = len(cands)
        return cands[:n], {"in": len(cands), "read": min(n, len(cands)), "tiers": [0, 0, len(cands)],
                           "read_per_tier": [0, 0, min(n, len(cands))], "cut_cos": 0.5}

    async def fake_claims(cands):
        return {c.pub_num: f"1. A camera and a marker ({c.pub_num})." for c in cands}, 1

    async def fake_judge(elements, docs, claims, call=None):
        for i, d in enumerate(docs):
            d["good_touches"] = {"e1": 1, "e2": 2} if i < 2 else {}
            d["good"] = bool(d["good_touches"])
        return {"judged": len(docs), "calls": 1, "good": 2, "with_claims": len(docs), "with_abstract": 0}

    from patent_analyzer.agentic import good as G
    from patent_analyzer.agentic import rounds as R
    monkeypatch.setattr(R, "select_for_reading", fake_select)
    monkeypatch.setattr(R, "_claims_for", fake_claims)
    monkeypatch.setattr(G, "judge", fake_judge)
    monkeypatch.setattr(WG, "READ_BUDGET", 3)

    pool = [Candidate(pub_num=f"US{i}B2", title=f"t{i}", match_type="Patent") for i in range(10)]
    delivered, stats = asyncio.run(WG.judge_and_deliver(ELS, pool))
    assert seen["offered"] == 10                      # the whole pool is offered, nothing pre-pruned
    assert stats["read"] == 3 and stats["good"] == 2 and stats["n_strong"] == 2
    assert [d["pub_num"] for d in delivered] == ["US0B2", "US1B2"]
    assert stats["coverage"] == {"e1": 2, "e2": 2} and stats["uncovered"] == []
    assert stats["read_pubs"] == ["US0B2", "US1B2", "US2B2"]


def test_an_empty_pool_is_not_an_error():
    delivered, stats = asyncio.run(WG.judge_and_deliver(ELS, []))
    assert delivered == [] and "no elements or empty pool" in stats["reason"]
