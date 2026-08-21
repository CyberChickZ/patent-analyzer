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
    assert len(out) == 3 and st["pool"] == 4 and st["stage1_out"] == 4 and st["stage2_out"] == 3 and st["unanswered"] == 0


def test_stage1_summary_query_and_cap():
    docs = [dict(d) for d in DOCS]
    idxs, st = stage1_embed(ELS, docs, topk=1, embed_docs=_emb, embed_queries=_emb, summary="a summary")
    assert 2 in idxs and docs[2]["prune_best_element"] == "summary"     # the bread recipe is closest to the summary vector
    idxs, st = stage1_embed(ELS, docs, topk=4, embed_docs=_emb, embed_queries=_emb, cap=2)
    assert len(idxs) == 2 and st["stage1_out"] == 2
