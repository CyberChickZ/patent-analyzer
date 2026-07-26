"""The agentic search loop (channel `agentic_loop` inside search_node).

round 1: every element, strict → validator-driven relax/tighten (≤2)
round 2: uncovered elements, loose + CL= field (claims)
round 3: uncovered elements, strict + CPC= from seeds
Each round: search (Google direct first, SerpAPI only when direct is
blocked/failed) → seeds → BQ expansion (examiner citations) → coverage
tag → per-round stats event.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from ..recall import google_patents as gp
from ..recall import serpapi as sp
from ..recall.pool import Candidate, candidates_to_legacy_docs, pool_and_dedupe
from .coverage import tag_coverage
from .elements import attach_facets, elements_from_state
from .expand import expand
from .query_gen import boolean_query, next_mode
from .validator import validate

MAX_ROUNDS = int(os.environ.get("LOOP_MAX_ROUNDS", "3"))
SEEDS_PER_ELEMENT = 10
GP_CALLS_PER_JOB = int(os.environ.get("LOOP_GP_MAX_CALLS", "30"))


class Budget:
    def __init__(self, serpapi_left, serpapi_take):
        self.gp_calls = 0
        self.serp_calls = 0
        self.gp_blocked = 0
        self._serp_left = serpapi_left
        self._serp_take = serpapi_take

    def gp_ok(self) -> bool:
        return self.gp_calls < GP_CALLS_PER_JOB and not gp.is_blocked()

    def serp_left(self) -> int:
        return self._serp_left()

    def serp_ok(self) -> bool:
        return self._serp_left() > 0 and self._serp_take()


async def _search(query: str, before: str | None, budget: Budget) -> tuple[list[Candidate], int | None, str]:
    """Direct first; SerpAPI only when direct is blocked or errors."""
    if query and budget.gp_ok():
        budget.gp_calls += 1
        cands, err = await gp.search(query, num=20, before=before)
        if not err:
            return cands, gp.last_total.get(query), "google_patents"
        if "blocked" in (err or ""):
            budget.gp_blocked += 1
    if query and budget.serp_ok():
        budget.serp_calls += 1
        cands, err = await sp.search_patents(query, max_pages=1)
        if not err:
            return cands, sp.last_total.get(query), "serpapi_patents"
    return [], None, "none"


async def run_loop(state: dict, serpapi_left, serpapi_take, event, embed=None) -> tuple[list[Candidate], dict]:
    """Returns (candidates for the pool, loop_stats)."""
    elements = elements_from_state(state)
    if not elements:
        return [], {"rounds": [], "reason": "no elements"}
    elements = await attach_facets(elements, state.get("summary", ""))
    before = state.get("date_cutoff")
    before = f"priority:{before}" if before and str(before).isdigit() else None
    budget = Budget(serpapi_left, serpapi_take)

    pool: dict[str, Candidate] = {}
    known: set[str] = set()
    uncovered = list(elements)
    cpc_hint: str | None = None
    rounds = []

    for rnd in range(1, MAX_ROUNDS + 1):
        if not uncovered:
            break
        field = "CL" if rnd == 2 else "AB"
        start_mode = "loose" if rnd == 2 else "strict"
        new_this_round: list[Candidate] = []
        queries_log = []
        for el in uncovered:
            mode = start_mode
            for attempt in range(3):
                q = boolean_query(el, mode, field=field, cpc=cpc_hint if rnd == 3 else None)
                if not q:
                    break
                cands, total, chan = await _search(q, before, budget)
                verdict = validate(total, len(cands))
                queries_log.append({"element": el["id"], "round": rnd, "mode": mode, "query": q[:160],
                                    "channel": chan, "total": total, "hits": len(cands), "verdict": verdict})
                for c in cands[:SEEDS_PER_ELEMENT]:
                    c.raw.setdefault("loop", {})["element"] = el["id"]
                    new_this_round.append(c)
                if verdict == "ok" or chan == "none":
                    break
                mode = next_mode(mode, verdict)
                if mode is None:
                    break
        seeds = [c.pub_num for c in new_this_round if c.pub_num]
        expanded, info = await expand(seeds, known) if seeds else ([], {})
        if info.get("cpc_subclasses"):
            cpc_hint = next(iter(info["cpc_subclasses"]))
        for c in new_this_round + expanded:
            key = (c.pub_num or c.title).upper()
            if key not in pool:
                pool[key] = c
            known.add(key)
        docs = candidates_to_legacy_docs(pool_and_dedupe({"agentic_loop": list(pool.values())}))
        cov = tag_coverage(elements, docs, embed=embed)
        covered_ids = {eid for eid, hits in cov.items() if hits}
        uncovered = [e for e in elements if e["id"] not in covered_ids]
        stats = {"round": rnd, "n_queries": len(queries_log), "gp_calls": budget.gp_calls,
                 "serpapi_calls": budget.serp_calls, "gp_blocked": budget.gp_blocked,
                 "seeds": len(seeds), "expanded": len(expanded), "pool_size": len(pool),
                 "new_in_pool": len(new_this_round) + len(expanded),
                 "covered": sorted(covered_ids), "uncovered": [e["id"] for e in uncovered],
                 "cpc_hint": cpc_hint, "queries": queries_log,
                 "ts": datetime.now(timezone.utc).isoformat()}
        rounds.append(stats)
        event("round_done", f"loop round {rnd}: pool {len(pool)}, covered {len(covered_ids)}/{len(elements)}, "
                            f"gp {budget.gp_calls} serp {budget.serp_calls}", stats)
        if not budget.gp_ok() and budget.serp_left() <= 0:
            break
    return list(pool.values()), {"rounds": rounds, "elements": [{"id": e["id"], "text": e["text"], "facets": e.get("facets")}
                                                              for e in elements],
                                 "coverage_by_element": cov if rounds else {}}
