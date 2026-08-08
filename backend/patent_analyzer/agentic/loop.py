"""The agentic search loop (channel `agentic_loop` inside search_node).

round 1: every element, strict (thing ∧ place ∧ apparatus, full text);
         zero → loose → core; too_broad is accepted (engine ranking)
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
from .elements import attach_facets, candidates_from_state, elements_from_state
from .expand import MAX_CITED_LIGHT, expand
from .query_gen import boolean_query, next_mode
from .validator import validate
from .wide import wide_queries

MAX_ROUNDS = int(os.environ.get("LOOP_MAX_ROUNDS", "3"))
MAX_ELEMENTS = int(os.environ.get("LOOP_MAX_ELEMENTS", "12"))
SEEDS_PER_ELEMENT = 10
GP_CALLS_PER_JOB = int(os.environ.get("LOOP_GP_MAX_CALLS", "30"))
LOOP_MODE = os.environ.get("LOOP_MODE", "wide")          # wide (recall-first) | elements (per-element walk)
WIDE_MAX_QUERIES = int(os.environ.get("LOOP_WIDE_MAX_QUERIES", "10"))


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


GP_WAIT_S = int(os.environ.get("LOOP_GP_WAIT_S", "0"))   # evals: wait out a Google block instead of dropping the query


async def _wait_for_gp(budget: Budget) -> None:
    """With no SerpAPI credit left, an eval run may sit out Google's soft
    block (15 min breaker) rather than log the query as `none`."""
    import asyncio
    waited = 0
    while GP_WAIT_S and gp.is_blocked() and budget.gp_calls < GP_CALLS_PER_JOB and budget.serp_left() <= 0 and waited < GP_WAIT_S:
        await asyncio.sleep(30)
        waited += 30


async def _search(query: str, before: str | None, budget: Budget, num: int = 20,
                  scholar: bool = False) -> tuple[list[Candidate], int | None, str]:
    """Direct first; SerpAPI only when direct is blocked or errors."""
    if query:
        await _wait_for_gp(budget)
    if query and budget.gp_ok():
        budget.gp_calls += 1
        cands, err = await gp.search(query, num=num, before=before)
        if not err:
            return cands, gp.last_total.get(query), "google_patents"
        if "blocked" in (err or ""):
            budget.gp_blocked += 1
    if query and budget.serp_ok():
        budget.serp_calls += 1
        cands, err = await sp.search_patents(query, max_pages=1, before=before, scholar=scholar)
        if not err:
            return cands, sp.last_total.get(query), "serpapi_patents"
    return [], None, "none"


async def run_wide(state: dict, serpapi_left, serpapi_take, event) -> tuple[list[Candidate], dict]:
    """Recall-first: ≤WIDE_MAX_QUERIES broad queries over every candidate
    invention, 100 results each (patents + scholar), every hit is a seed,
    citation expansion up to MAX_CITED_LIGHT. No validator walk: the pool is
    narrowed afterwards by prune.py."""
    cands = candidates_from_state(state)
    if not cands:
        return [], {"rounds": [], "reason": "no elements"}
    all_els = [e for c in cands for e in c["elements"]][:4 * MAX_ELEMENTS]
    await attach_facets(all_els, state.get("summary", ""))
    cutoff = str(state.get("date_cutoff") or "")
    cutoff = cutoff if cutoff.isdigit() and len(cutoff) == 8 else None
    before = f"priority:{cutoff}" if cutoff else None
    budget = Budget(serpapi_left, serpapi_take)
    queries = wide_queries(cands, max_total=WIDE_MAX_QUERIES)
    pool: dict[str, Candidate] = {}
    log = []
    seeds: list[str] = []
    for i, q in enumerate(queries, 1):
        hits, total, chan = await _search(q["query"], before, budget, num=100, scholar=True)
        returned, new_keys = [], []
        for c in hits:
            c.raw.setdefault("loop", {})["candidate"] = q["candidate"]
            key = (c.pub_num or c.title).upper()
            returned.append(c.pub_num or c.title[:80])
            if key not in pool:
                pool[key] = c
                new_keys.append(c.pub_num or c.title[:80])
            if c.pub_num and c.match_type == "Patent":
                seeds.append(c.pub_num)
        log.append({"n": i, "candidate": q["candidate"], "kind": q["kind"], "query": q["query"],
                    "facets_used": q.get("facets_used", {}), "elements": q.get("elements", []),
                    "channel": chan, "total": total, "hits": len(hits), "new": len(new_keys),
                    "papers": sum(1 for c in hits if c.match_type != "Patent"),
                    "pubs": returned, "new_pubs": new_keys})
    seeds = list(dict.fromkeys(seeds))
    expanded, info = await expand(seeds, set(pool), max_cited=MAX_CITED_LIGHT, before=cutoff, light=True) if seeds else ([], {})
    dropped = set(info.get("seeds_after_cutoff") or [])
    for k in list(pool):
        if k in dropped:
            del pool[k]
    for c in expanded:
        pool.setdefault((c.pub_num or c.title).upper(), c)
    stats = {"round": 1, "mode": "wide", "n_queries": len(log), "gp_calls": budget.gp_calls,
             "serpapi_calls": budget.serp_calls, "gp_blocked": budget.gp_blocked,
             "seeds": len(seeds), "seeds_after_cutoff": sorted(dropped), "expanded": len(expanded),
             "cited_total": info.get("cited_total", 0), "cited_light": info.get("cited_light", 0),
             "cited_by_seed": info.get("cited_by_seed", {}), "expanded_pubs": sorted((c.pub_num or c.title).upper() for c in expanded),
             "pool_size": len(pool), "queries": log, "pool_pubs": sorted(pool), "seed_pubs": sorted(set(seeds) - dropped),
             "covered": [], "uncovered": [], "cpc_hint": next(iter(info.get("cpc_subclasses") or {}), None),
             "ts": datetime.now(timezone.utc).isoformat()}
    event("round_done", f"wide: {len(log)} queries, {len(seeds)} seeds, +{len(expanded)} cited, pool {len(pool)}, "
                        f"gp {budget.gp_calls} serp {budget.serp_calls}", stats)
    return list(pool.values()), {"rounds": [stats], "mode": "wide",
                                 "elements": [{"id": e["id"], "text": e["text"], "facets": e.get("facets"),
                                               "candidate": c["id"]} for c in cands for e in c["elements"]],
                                 "candidates": [{"id": c["id"], "level": c["level"], "n_elements": len(c["elements"])} for c in cands],
                                 "coverage_by_element": {}}


async def run_loop(state: dict, serpapi_left, serpapi_take, event, embed=None) -> tuple[list[Candidate], dict]:
    """Returns (candidates for the pool, loop_stats)."""
    if LOOP_MODE == "wide":
        return await run_wide(state, serpapi_left, serpapi_take, event)
    elements = elements_from_state(state)[:MAX_ELEMENTS]
    if not elements:
        return [], {"rounds": [], "reason": "no elements"}
    elements = await attach_facets(elements, state.get("summary", ""))
    cutoff = str(state.get("date_cutoff") or "")
    cutoff = cutoff if cutoff.isdigit() and len(cutoff) == 8 else None
    before = f"priority:{cutoff}" if cutoff else None
    budget = Budget(serpapi_left, serpapi_take)

    pool: dict[str, Candidate] = {}
    known: set[str] = set()
    uncovered = list(elements)
    cpc_hint: str | None = None
    rounds = []

    for rnd in range(1, MAX_ROUNDS + 1):
        if not uncovered:
            break
        # round 1: three facets on full text, relax on zero (pilot: AND of
        # facets, drop one when nothing matches); round 2: claims-scoped
        # two facets; round 3: strict + CPC from the seeds.
        field = "CL" if rnd == 2 else ""
        start_mode = "loose" if rnd == 2 else "strict"
        new_this_round: list[Candidate] = []
        queries_log = []
        for el in uncovered:
            mode, tried = start_mode, set()
            for attempt in range(3):
                q = boolean_query(el, mode, field=field, cpc=cpc_hint if rnd == 3 else None)
                if not q or q in tried:   # zero → core → too_broad → loose would repeat the query
                    break
                tried.add(q)
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
        expanded, info = await expand(seeds, known, before=cutoff) if seeds else ([], {})
        dropped = set(info.get("seeds_after_cutoff") or [])
        new_this_round = [c for c in new_this_round if (c.pub_num or "").upper() not in dropped]
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
                 "pool_pubs": sorted(k for k in pool),
                 "new_pubs": sorted({(c.pub_num or c.title).upper() for c in new_this_round + expanded}),
                 "seed_pubs": sorted(set(seeds) - dropped), "seeds_after_cutoff": sorted(dropped),
                 "ts": datetime.now(timezone.utc).isoformat()}
        rounds.append(stats)
        event("round_done", f"loop round {rnd}: pool {len(pool)}, covered {len(covered_ids)}/{len(elements)}, "
                            f"gp {budget.gp_calls} serp {budget.serp_calls}", stats)
        if not budget.gp_ok() and budget.serp_left() <= 0:
            break
    return list(pool.values()), {"rounds": rounds, "elements": [{"id": e["id"], "text": e["text"], "facets": e.get("facets")}
                                                              for e in elements],
                                 "coverage_by_element": cov if rounds else {}}
