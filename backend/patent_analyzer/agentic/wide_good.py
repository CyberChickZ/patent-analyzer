"""Delivery by the claims judge, on top of the wide loop's recall.

Five tags and four rounds of fixes said the same thing: M1's recall strategy —
expand only from documents the claims judge called GOOD, with a per-round cap —
reaches less gold than the wide loop it was meant to replace. On the two
comparable papers, pool reach was .267 against the wide loop's .400 and the
ReAct wide loop's .600, and by the last round the pool was no longer small
(7,788-15,063 candidates) — it was full of the wrong documents. The GOOD judge
cannot see a bridge: a document that cites the gold need not disclose any
element of the invention itself, so requiring GOOD before expanding throws the
bridges away, and the citation expansion is where 19 of h1h's 30 first-reaches
came from.

What M1 did prove is the other half. Reading claims and making the model point
at a claim number (or, for a paper, at a quote we can locate) is a far better
basis for DELIVERY than any abstract-level ranking: on h1o the abstract screen
called 5,434 documents worth reading and the two examiner-cited families sat at
rank 598 and 1,841 of every abstract ordering tried, while reading claims
judged both True with one at rank 51.

So this module keeps that half and drops the other: the wide loop recalls,
and the claims judge decides what is delivered and in what order.
"""

from __future__ import annotations

import os
import time

from . import good as G
from . import rounds as R
from ..recall.pool import Candidate

READ_BUDGET = int(os.environ.get("WIDE_GOOD_READ", "900"))
DELIVER = int(os.environ.get("M1_DELIVER", "120"))


def rank_for_delivery(docs: list[dict]) -> list[dict]:
    """Strong first, then by how much of the invention the document touches,
    then by how specifically it pointed (a claim number per element), then by
    cosine. `good_touches` maps an element id to a claim number for a patent
    and to a located quote for a paper, so counting the numeric ones is
    counting the places a patent said it in its own claims."""
    def key(d: dict):
        touches = d.get("good_touches") or {}
        numbered = sum(1 for v in touches.values() if isinstance(v, int))
        return (not G.is_strong(d), -len(touches), -numbered, -float(d.get("prune_cos") or 0.0))
    return sorted([d for d in docs if d.get("good")], key=key)


async def judge_and_deliver(elements: list[dict], pool: list[Candidate], summary: str = "",
                            event=None) -> tuple[list[dict], dict]:
    """(delivered docs, stats). Reads the claims of the documents the budget
    selects out of the whole pool, judges them, and delivers the GOOD ones in
    order. Nothing is pruned on an abstract."""
    t0 = time.monotonic()
    if not elements or not pool:
        return [], {"mode": "wide_good", "reason": "no elements or empty pool"}
    read, sel = R.select_for_reading(elements, list(pool), READ_BUDGET, summary)
    claims, bq_calls = await R._claims_for(read)
    docs = [{"pub_num": c.pub_num, "title": c.title, "sources": c.sources,
             "abstract": c.abstract or c.snippet, "raw": c.raw,
             "prune_cos": float((c.raw or {}).get("cos") or 0.0)} for c in read]
    judge = await G.judge(elements, docs, claims)
    ranked = rank_for_delivery(docs)
    cover = G.coverage(ranked, elements)
    stats = {
        "mode": "wide_good", "pool": len(pool), "select": sel, "read": len(read),
        "bq_calls": bq_calls, "judge": judge,
        "good": len([d for d in docs if d.get("good")]),
        "n_strong": sum(1 for d in ranked if G.is_strong(d)),
        "delivered": min(len(ranked), DELIVER),
        "coverage": cover, "uncovered": G.uncovered(cover),
        "read_pubs": [d["pub_num"] for d in docs if d.get("pub_num")],
        "seconds": round(time.monotonic() - t0, 1),
    }
    if event:
        event("wide_good", f"claims read for {len(read)} of {len(pool)} pooled; {stats['good']} GOOD "
                           f"({stats['n_strong']} strong); delivering {stats['delivered']}", stats)
    return ranked[:DELIVER], stats
