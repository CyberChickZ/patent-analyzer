"""LOOP_MODE=moves — the M1 entry point.

Round 0 is the only part that touches the input paper: the predicted CPC
groups enumerated through USPTO ODP, the paper's own neighbourhood bridged to
patents, and the ReAct queries (still on 2.5-pro, whose gate the flash models
lost: H1-01 pool reach 4/5 against 1/5 and 0/5). Everything after that is
rounds.run_rounds walking out from the documents whose claims were read.
"""

from __future__ import annotations

import asyncio
import os

from ..recall.pool import Candidate
from . import moves as M
from . import rounds as R
from .elements import attach_facets, candidates_from_state
from .wide import broad_terms, title_terms

MAX_ELEMENTS = int(os.environ.get("LOOP_MAX_ELEMENTS", "12"))


def _title_terms(elements: list[dict], cap: int = 12) -> list[str]:
    """Single words from the elements' own 'thing' forms — what the ODP title
    index can actually match (it has no abstract and no claims)."""
    out: list[str] = []
    for e in elements:
        for form in ((e.get("facets") or {}).get("thing") or [])[:3]:
            for w in str(form).lower().split():
                if len(w) > 3 and w not in out:
                    out.append(w)
    return out[:cap]


async def run_moves(state: dict, serpapi_left, serpapi_take, event) -> tuple[list[Candidate], dict]:
    cands = candidates_from_state(state)
    if not cands:
        return [], {"rounds": [], "reason": "no elements", "mode": "moves"}
    core = cands[0]
    elements = [e for c in cands for e in c["elements"]][:4 * MAX_ELEMENTS]
    await attach_facets(elements, state.get("summary", ""))
    cutoff = str(state.get("date_cutoff") or "")
    cutoff = cutoff if cutoff.isdigit() and len(cutoff) == 8 else None

    pred = [str(c).split("/")[0] for c in (core.get("cpc_pred") or []) if len(str(c).split("/")[0]) >= 5]
    neigh = ((core.get("elements") or [{}])[0].get("cpc_groups") if core.get("elements") else None) or []
    groups = list(dict.fromkeys(pred + neigh))[:4]
    terms = _title_terms(elements)

    async def round0():
        """S2 (predicted CPC enumerated) + S3 (ReAct) + the paper bridge. The wide loop runs in
        seed-only mode: its expansion / similar / CPC round are moves here, run from GOOD."""
        os.environ.setdefault("WIDE_SEED_ONLY", "1")
        from .loop import run_wide                       # reuse the neighbourhood + ReAct machinery
        results: list[M.MoveResult] = []
        s2, wide = await asyncio.gather(
            M.p5_cpc_enum(groups, terms, cutoff, set(), cap=M.CAPS["S2_predicted_cpc"],
                          round_no=0, name="S2_predicted_cpc"),
            run_wide(state, serpapi_left, serpapi_take, event),
            return_exceptions=True)
        if isinstance(s2, Exception):
            s2 = M.MoveResult("S2_predicted_cpc", 0, error=f"{type(s2).__name__}: {s2}"[:200])
        results.append(s2)
        cands0 = list(s2.candidates)
        if isinstance(wide, Exception):
            results.append(M.MoveResult("S3_react", 0, error=f"{type(wide).__name__}: {wide}"[:200]))
        else:
            wide_cands, wide_stats = wide
            r = M.MoveResult("S3_react", 0, list(wide_cands)[:M.CAPS["S3_react"]])
            rnd = (wide_stats.get("rounds") or [{}])[0]
            r.calls = len(rnd.get("queries") or [])
            r.note = f"wide loop: {len(wide_cands)} candidates, {r.calls} queries"
            results.append(r)
            cands0 += r.candidates
        return cands0, results

    out = await R.run_rounds(elements, groups, terms, cutoff, round0, event=event)
    good = out["good"]
    stats = {
        "mode": "moves", "rounds": out["rounds"], "stop": out["stop"], "coverage": out["coverage"],
        "move_rows": out["rows"], "seconds": out["seconds"],
        "good": [{"pub_num": d.get("pub_num"), "title": (d.get("title") or "")[:100],
                  "sources": d.get("sources"), "touches": d.get("good_touches"),
                  "reasons": d.get("good_reasons")} for d in good],
        "elements": [{"id": e["id"], "text": e["text"], "facets": e.get("facets"),
                      "candidate": c["id"]} for c in cands for e in c["elements"]],
        "candidates": [{"id": c["id"], "level": c["level"], "n_elements": len(c["elements"])} for c in cands],
        "title_terms": terms, "cpc_groups": groups,
    }
    return out["pool"], stats
