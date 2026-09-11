import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from patent_analyzer.agentic.prune import prune, stage1_embed, stage2_llm

ELS = [{"id": "e1", "text": "gaze estimation"}, {"id": "e2", "text": "turntable rotation"}]
DOCS = [{"pub_num": "US1", "title": "gaze tracking for video calls", "abstract": ""},
        {"pub_num": "US2", "title": "motorized turntable", "abstract": "rotates a display"},
        {"pub_num": "US3", "title": "bread recipe", "abstract": "flour and water"},
        {"pub_num": "US4", "title": "eye direction estimation", "abstract": ""}]
VEC = {"gaze estimation": [1, 0], "turntable rotation": [0, 1], "gaze tracking for video calls": [0.9, 0.1], "a summary": [0.5, 0.5],
       "motorized turntable rotates a display": [0.1, 0.9], "bread recipe flour and water": [0.5, 0.5],
       "eye direction estimation": [0.8, 0.2]}


def _emb(texts):
    return np.array([VEC[t] for t in texts], dtype=np.float32)


def test_stage1_union_of_per_element_topk():
    docs = [dict(d) for d in DOCS]
    idxs, st = stage1_embed(ELS, docs, topk=1, embed_docs=_emb, embed_queries=_emb)
    assert sorted(idxs) == [0, 1] and st["stage1_in"] == 4 and st["stage1_out"] == 2 and st["stage1_cut_cos"] > 0.9
    assert docs[0]["prune_cos"] > 0.9


def test_stage2_keeps_worth_reading_and_orders_by_elements_then_cosine():
    docs = [dict(d) for d in DOCS]
    idxs, _ = stage1_embed(ELS, docs, topk=3, embed_docs=_emb, embed_queries=_emb)
    prompts = []

    async def fake_call(system, user, response_schema=None):
        prompts.append(user)
        return json.dumps({"verdicts": [{"i": 0, "worth_reading": True, "elements": ["e1"]},
                                        {"i": 1, "worth_reading": True, "elements": ["e2", "e1"]},
                                        {"i": 2, "worth_reading": False, "elements": []},
                                        {"i": 3, "worth_reading": True, "elements": ["e1"]}]})
    kept, st = asyncio.run(stage2_llm([{"id": "inv1", "concept": "c"}], ELS, docs, idxs, batch_size=2, keep=2, call=fake_call))
    assert st["stage2_calls"] == 2 and st["stage2_worth"] == 3 and st["stage2_out"] == 2
    assert kept[0] == 1 and kept[1] in (0, 3) and docs[2]["prune_worth_reading"] is False
    assert docs[2]["prune_reason"] == "" and docs[0]["prune_stage1"] and docs[0]["prune_best_element"] == "e1"
    assert "[0]" in prompts[0] and "e2: turntable rotation" in prompts[0]


def test_prune_end_to_end_reports_every_stage():
    docs = [dict(d) for d in DOCS]

    async def fake_call(system, user, response_schema=None):
        return json.dumps({"verdicts": [{"i": i, "worth_reading": True, "elements": []} for i in range(4)]})
    out, st = asyncio.run(prune([{"id": "inv1"}], ELS, docs, embed_docs=_emb, embed_queries=_emb, call=fake_call, topk=4, keep=3))
    # stage 2 now keeps every document worth reading (the claims screen makes the final cut)
    assert len(out) == 3 and st["pool"] == 4 and st["stage1_out"] == 4 and st["stage2_out"] == 4 and st["unanswered"] == 0


def test_stage1_summary_query_and_cap():
    docs = [dict(d) for d in DOCS]
    idxs, st = stage1_embed(ELS, docs, topk=1, embed_docs=_emb, embed_queries=_emb, summary="a summary")
    assert 2 in idxs and docs[2]["prune_best_element"] == "summary"     # the bread recipe is closest to the summary vector
    idxs, st = stage1_embed(ELS, docs, topk=4, embed_docs=_emb, embed_queries=_emb, cap=2)
    assert len(idxs) == 2 and st["stage1_out"] == 2


def test_graph_sourced_docs_skip_the_embedding_cut():
    docs = [dict(d) for d in DOCS]
    docs[2]["sources"] = ["citation_graph"]          # bread recipe, far from every element
    idxs, st = stage1_embed(ELS, docs, topk=1, embed_docs=_emb, embed_queries=_emb)
    assert 2 in idxs and docs[2]["prune_stage1"]


def test_llm_screen_cannot_drop_a_graph_sourced_doc():
    docs = [dict(d) for d in DOCS]
    docs[2]["sources"] = ["google_similar"]
    idxs, _ = stage1_embed(ELS, docs, topk=4, embed_docs=_emb, embed_queries=_emb)

    async def fake_call(system, user, response_schema=None):
        return json.dumps({"verdicts": [{"i": i, "worth_reading": i != 2, "elements": [], "reason": "r"} for i in range(4)]})
    kept, st = asyncio.run(stage2_llm([{"id": "inv1"}], ELS, docs, idxs, batch_size=4, keep=4, call=fake_call))
    assert 2 in kept and "graph source" in docs[2]["prune_reason"] and "When in doubt, keep it" in __import__("patent_analyzer.agentic.prune", fromlist=["x"])._batch_prompt([{"id": "inv1"}], ELS, [(0, docs[0])])


def test_claims_screen_reorders_by_what_the_claims_touch():
    from patent_analyzer.agentic.prune import stage3_claims
    docs = [dict(d) for d in DOCS]
    for i, d in enumerate(docs):
        d["pub_num"] = f"US{1000 + i}B2"
        d["prune_cos"] = 0.9 - 0.1 * i          # doc 0 is the closest by embedding
        d["prune_elements"] = []

    async def fake_claims(pubs):
        return {p.replace("-", ""): f"1. A thing comprising {p}." for p in pubs}

    async def fake_call(system, user, response_schema=None):
        # only the third document's claims touch two elements
        return json.dumps({"verdicts": [{"i": 0, "worth_reading": False, "elements": [], "reason": "different problem"},
                                        {"i": 1, "worth_reading": True, "elements": ["inv1.e1"], "reason": "one"},
                                        {"i": 2, "worth_reading": True, "elements": ["inv1.e1", "inv1.e2"], "reason": "two"},
                                        {"i": 3, "worth_reading": True, "elements": [], "reason": "maybe"}]})
    kept, st = asyncio.run(stage3_claims(ELS, docs, [0, 1, 2, 3], keep=3, fetch_claims=fake_claims, call=fake_call))
    assert kept[0] == 2 and kept[1] == 1          # ordered by how many elements the claims touch
    assert 0 not in kept                          # the one the claims rule out drops off the 3-cut
    assert st["stage3_with_claims"] == 4 and st["stage3_calls"] == 1 and docs[0]["claims_reason"] == "different problem"


def test_stage1_gives_papers_their_own_share_so_they_cannot_crowd_out_patents():
    """Giving the paper channel named-entity queries took a job's paper returns
    from 100-1,000 to 1,600-3,200 (N5, 2026-09-18). On cosine alone those would
    take the whole shortlist; the gold is measured on the patent side."""
    import numpy as np
    from patent_analyzer.agentic import prune as P
    docs = ([{"title": f"paper {i}", "abstract": "x", "match_type": "Paper"} for i in range(1000)]
            + [{"title": f"patent {i}", "abstract": "x", "match_type": "Patent"} for i in range(200)])

    def ed(texts):      # every paper is a perfect match, every patent a mediocre one
        return np.array([[1.0, 0.0] if t.startswith("paper") else [0.6, 0.8] for t in texts], dtype=np.float32)

    def eq(texts):
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)

    out, info = P.stage1_embed([{"id": "e1", "text": "q"}], docs, topk=1200, cap=300,
                               embed_docs=ed, embed_queries=eq)
    kinds = [docs[i]["match_type"] for i in out]
    assert len(out) == 300
    assert kinds.count("Patent") == 200          # every patent survives despite the worse cosine
    assert kinds.count("Paper") == 100           # papers take their 30% share plus the unused rest
    assert info["stage1_papers_out"] == 100


def test_doi_backfill_fills_only_empty_dois_and_rejects_a_different_title():
    """41 of the 88 unreadable papers in a real job had no DOI at all — no DOI
    means no Unpaywall, no Europe PMC, no cache key and nothing to put on a
    manual-download list (N7, 2026-09-18). A wrong DOI is worse than none, so a
    result whose title is not the same title is rejected."""
    import asyncio
    from unittest import mock
    from patent_analyzer.recall import openalex as oa
    from patent_analyzer.recall.pool import Candidate

    async def fake_get(client, url, params, *a, **k):
        t = params["filter"].split("title.search:")[1]
        if "Sourdough" in t:
            return {"results": [{"title": "A completely different work", "doi": "https://doi.org/10.9/wrong"}]}, None
        return {"results": [{"title": "CO-FISH strand orientation", "doi": "https://doi.org/10.1/right"}]}, None

    cands = [Candidate(title="CO-FISH strand orientation", match_type="Paper"),
             Candidate(title="Sourdough starters and wild yeast cultures", match_type="Paper"),
             Candidate(title="Has one already", match_type="Paper", doi="10.5/keep"),
             Candidate(title="US Patent thing", match_type="Patent", pub_num="US1A1")]
    class _Mem:                       # the real KV persists across runs, which would hide the call
        def __init__(self):
            self.d = {}

        def get(self, ns, key, max_age_days=None):
            return self.d.get((ns, key))

        def put(self, ns, key, value):
            self.d[(ns, key)] = value
    import patent_analyzer.cache as _cache
    with mock.patch.object(oa, "_get", fake_get), mock.patch.object(_cache, "kv", lambda: _Mem()):
        out = asyncio.run(oa.backfill_dois(cands))
    assert out == {"asked": 2, "filled": 1, "cached": 0, "mismatched": 1}
    assert cands[0].doi == "10.1/right"
    assert cands[1].doi == ""            # different title -> rejected
    assert cands[2].doi == "10.5/keep"   # never overwritten
    assert cands[3].doi == ""            # patents are not asked about
