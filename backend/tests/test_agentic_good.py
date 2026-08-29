import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.good import coverage, independent_claims, judge, rank_good

ELS = [{"id": "inv1.e1", "text": "a pair of cameras"}, {"id": "inv1.e2", "text": "an infrared marker"}]
DOCS = [{"pub_num": "US1111111B2", "title": "Stereo camera rig"},
        {"pub_num": "US2222222B2", "title": "Bread machine"}]
CLAIMS = {"US1111111B2": "1. A system comprising two cameras.\n2. The system of claim 1, wherein a marker is reflective.",
          "US2222222B2": "1. A bread machine."}


def test_independent_claims_drops_the_dependent_ones():
    out = independent_claims(CLAIMS["US1111111B2"])
    assert out.startswith("1. A system") and "claim 1" not in out


def test_an_element_counts_only_with_a_claim_number():
    async def fake_call(system, user, response_schema=None):
        assert "CLAIM NUMBER" in user or "name the NUMBER" in user
        return json.dumps({"verdicts": [
            {"i": 0, "touches": [{"element_id": "inv1.e1", "claim_number": 1, "reason": "two cameras"},
                                 {"element_id": "inv1.e2", "claim_number": 0, "reason": "no number"},
                                 {"element_id": "nope.e9", "claim_number": 3, "reason": "unknown element"}]},
            {"i": 1, "touches": []}]})
    docs = [dict(d) for d in DOCS]
    stats = asyncio.run(judge(ELS, docs, CLAIMS, call=fake_call))
    assert docs[0]["good"] and docs[0]["good_touches"] == {"inv1.e1": 1}      # 0 and the unknown id dropped
    assert docs[1]["good"] is False and docs[1]["good_touches"] == {}
    assert stats == {"judged": 2, "calls": 1, "good": 1, "with_claims": 2}
    assert [d["pub_num"] for d in rank_good(docs)] == ["US1111111B2"]
    assert coverage(docs, ELS) == {"inv1.e1": 1, "inv1.e2": 0}


def test_documents_without_claims_are_not_judged():
    async def fake_call(*a, **k):
        raise AssertionError("must not be called")
    docs = [{"pub_num": "US9999999B2", "title": "No claims here"}]
    assert asyncio.run(judge(ELS, docs, {}, call=fake_call))["judged"] == 0
    assert "good" not in docs[0]
