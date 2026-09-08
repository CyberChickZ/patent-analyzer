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

import asyncio
import os
from datetime import datetime, timezone

from ..recall import google_patents as gp
from ..recall import serpapi as sp
from ..recall.pool import Candidate, candidates_to_legacy_docs, pool_and_dedupe
from .coverage import tag_coverage
from .elements import attach_facets, candidates_from_state, elements_from_state
from .expand import MAX_CITED_LIGHT, expand, similar_neighbours
from .neighbourhood import paper_neighbourhood
from .query_gen import boolean_query, next_mode
from .validator import validate
from .react_search import run_react
from .wide import broad_terms, cpc_queries, title_terms

MAX_ROUNDS = int(os.environ.get("LOOP_MAX_ROUNDS", "3"))
MAX_ELEMENTS = int(os.environ.get("LOOP_MAX_ELEMENTS", "12"))
SEEDS_PER_ELEMENT = 10
GP_CALLS_PER_JOB = int(os.environ.get("LOOP_GP_MAX_CALLS", "30"))
LOOP_MODE = os.environ.get("LOOP_MODE", "wide")          # moves (M1 rounds) | wide (recall-first) | elements
WIDE_MAX_QUERIES = int(os.environ.get("LOOP_WIDE_MAX_QUERIES", "10"))
CPC_QUERIES = int(os.environ.get("LOOP_CPC_QUERIES", "2"))
LENS_CALLS = int(os.environ.get("LOOP_LENS_CALLS", "6"))      # Lens trial: 1000 req/month


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
    # two of the budget are held back for the CPC queries issued after expansion
    pool: dict[str, Candidate] = {}
    log = []
    seeds: list[str] = []

    async def _run(qs: list[dict]) -> list[str]:
        new_seeds: list[str] = []
        for q in qs:
            if not budget.gp_ok() and budget.serp_left() <= 0:
                log.append({"n": len(log) + 1, "candidate": q["candidate"], "kind": q["kind"], "query": q["query"],
                            "facets_used": q.get("facets_used", {}), "elements": q.get("elements", []),
                            "channel": "skipped:budget", "total": None, "hits": 0, "new": 0, "papers": 0, "pubs": [], "new_pubs": []})
                continue
            hits, total, chan = await _search(q["query"], before, budget, num=100, scholar=True)
            returned, new_keys = [], []
            for rank, c in enumerate(hits):
                loop_meta = c.raw.setdefault("loop", {})
                loop_meta["candidate"] = q["candidate"]
                # first query that found it, and where in that query's results — M1 reads every
                # query's top 10 whatever the embedding says (h1i reached 2 of its 4 gold families
                # straight off a query, and a cosine cut would have to keep them to see them)
                loop_meta.setdefault("q", q["query"][:160])
                loop_meta.setdefault("rank", rank)
                key = (c.pub_num or c.title).upper()
                returned.append(c.pub_num or c.title[:80])
                if key not in pool:
                    pool[key] = c
                    new_keys.append(c.pub_num or c.title[:80])
                if c.pub_num and c.match_type == "Patent":
                    new_seeds.append(c.pub_num)
            log.append({"n": len(log) + 1, "candidate": q["candidate"], "kind": q["kind"], "query": q["query"],
                        "facets_used": q.get("facets_used", {}), "elements": q.get("elements", []),
                        "channel": chan, "total": total, "hits": len(hits), "new": len(new_keys),
                        "papers": sum(1 for c in hits if c.match_type != "Patent"),
                        "pubs": returned, "new_pubs": new_keys})
        return new_seeds

    # paper neighbourhood runs alongside the patent queries (Harry: parallel, no ordering)
    async def _neigh():
        if os.environ.get("LOOP_NEIGHBOURHOOD", "1") != "1":
            return [], {"skipped": True}
        try:
            return await paper_neighbourhood(state.get("source_title", ""), cands, cutoff=cutoff,
                                             doi=state.get("source_doi", ""), arxiv_id=state.get("source_arxiv_id", ""),
                                             summary=state.get("summary", ""))
        except Exception as exc:
            return [], {"error": f"{type(exc).__name__}: {exc}"[:200]}

    # ReAct loop (Harry: "做 query 本身就是用 agent 能力去更好地搜索"): the neighbourhood runs
    # first so its title terms are in the broad group; then the model drives ≤10 queries.
    # The template query path (LOOP_REACT=0, wide_queries + terms_query) was removed
    # 2026-09-18; _run below still serves the CPC round.
    papers, neigh_info = await _neigh()
    nterms0 = title_terms([c.title for c in papers]) if papers else []
    core = cands[0]
    pred_groups = [str(c).split("/")[0] for c in (core.get("cpc_pred") or []) if len(str(c).split("/")[0]) >= 5]
    neigh_groups = ((core.get("elements") or [{}])[0].get("cpc_groups") if core.get("elements") else None) or []
    groups = list(dict.fromkeys(pred_groups + [g for g in neigh_groups]))[:8]
    bterms = broad_terms(core, nterms0)

    async def _react_search(query: str):
        hits, total, chan = await _search(query, before, budget, num=100, scholar=True)
        return hits, total, chan

    steps = await run_react(core.get("elements") or [], bterms, groups, _react_search,
                            lambda: min(budget.serp_left(), max(0, WIDE_MAX_QUERIES - CPC_QUERIES - len(log))) if not budget.gp_ok() else WIDE_MAX_QUERIES - CPC_QUERIES - len(log),
                            event=event)
    q_seeds = []
    for st in steps:
        hits = st.pop("_hits", [])
        new_keys = []
        for rank, c in enumerate(hits):
            loop_meta = c.raw.setdefault("loop", {})
            loop_meta["candidate"] = core.get("id")
            # Where this hit sat in its query's results, so M1's claims budget can reserve every
            # query's top ten. The same three lines exist in _run above, but _run serves the
            # template queries — THIS is the path ReAct actually takes, and m1d's selection came
            # back with an empty top-ten tier because only _run had been given them.
            loop_meta.setdefault("q", str(st.get("query") or "")[:160])
            loop_meta.setdefault("rank", rank)
            key = (c.pub_num or c.title).upper()
            if key not in pool:
                pool[key] = c
                new_keys.append(c.pub_num or c.title[:80])
            if c.pub_num and c.match_type == "Patent":
                q_seeds.append(c.pub_num)
        st["new_pubs"] = new_keys
        st["candidate"] = core.get("id")
        log.append({k: v for k, v in st.items() if k != "top"} | {"top_titles": [t["title"][:80] for t in st.get("top", [])[:10]]})
    seeds += q_seeds
    for c in papers:
        pool.setdefault((c.pub_num or c.title).upper(), c)
    # bridge: patents that cite the neighbourhood papers (Reliance on Science, amie_patents.pcs_oa)
    bridge_seeds: list[str] = []
    bridge_info: dict = {"oa_ids": 0, "patents": 0, "by_paper": {}}
    oa_ids = [(c.raw.get("neigh") or {}).get("oa_id") for c in papers]
    oa_ids = [o for o in dict.fromkeys(oa_ids) if o]
    bridge_info["oa_ids"] = len(oa_ids)
    if oa_ids:
        try:
            from ..recall.bigquery_patents import fetch_citing_patents
            citing = await fetch_citing_patents(oa_ids)
            for oid, rows in citing.items():
                pubs = [r.get("patent_pub") or r.get("publication_number") or "" for r in rows]
                pubs = [x for x in pubs if x]
                if pubs:
                    bridge_info["by_paper"][oid] = pubs
                    bridge_seeds += pubs
        except ImportError:
            bridge_info["error"] = "fetch_citing_patents not available"
        except Exception as exc:
            bridge_info["error"] = f"{type(exc).__name__}: {exc}"[:200]
    bridge_seeds = list(dict.fromkeys(bridge_seeds))
    bridge_info["patents"] = len(bridge_seeds)
    seeds += bridge_seeds
    # Lens bridge + Lens patent search (trial: 1000 req/month, patent 10/min): ≤LENS_CALLS per job
    lens_info: dict = {"bridge_patents": 0, "search_calls": 0, "searches": [], "error": None}
    lens_pubs: list[str] = []
    if LENS_CALLS and os.environ.get("LENS_API_TOKEN"):
        try:
            from ..recall import lens
            dois = [c.doi for c in papers if c.doi][:400]
            sch, err = await lens.scholarly_by_ids(dois, oa_ids[:400])
            lens_ids = list(dict.fromkeys(l for v in (sch or {}).values() for l in (v.get("patent_citations") or [])))
            if lens_ids:
                lcands, err2 = await lens.patents_by_lens_ids(lens_ids[:800])
                for c in lcands:
                    key = (c.pub_num or c.title).upper()
                    if c.pub_num:
                        lens_pubs.append(c.pub_num)
                        seeds.append(c.pub_num)
                    pool.setdefault(key, c)
                lens_info["bridge_patents"] = len(lcands)
                lens_info["bridge_error"] = err2
            lens_info["bridge_papers"] = len(sch or {})
            lens_info["bridge_error"] = lens_info.get("bridge_error") or err
            # searches: the NEIGHBOURING main groups (facet call's cpc_groups, minus the ≤2 Google
            # already used) × the core candidate's thing forms — Lens is free during the trial,
            # so the extra groups cost no SerpAPI (leader H7 (a)); then per-element forms if budget remains
            core = cands[0]
            core_els = core.get("elements") or []
            pred = [str(c).split("/")[0] for c in (core.get("cpc_pred") or []) if len(str(c).split("/")[0]) >= 5]
            neigh_groups = [g for g in ((core_els[0].get("cpc_groups") if core_els else None) or []) if g not in pred[:2]]
            core_forms = []
            for e in core_els[:4]:
                for t in ((e.get("facets") or {}).get("thing") or [])[:2]:
                    t = " ".join(str(t).lower().split())
                    if t and t not in core_forms:
                        core_forms.append(t)
            lens_queries = [{"element": "core", "terms": core_forms[:6], "cpc": g} for g in neigh_groups[:LENS_CALLS - 2]]
            grp = pred[0] if pred else None
            for e in core_els[1:]:
                if len(lens_queries) >= LENS_CALLS - 1:
                    break
                forms = [" ".join(str(t).lower().split()) for t in ((e.get("facets") or {}).get("thing") or [])][:3]
                if forms:
                    lens_queries.append({"element": e.get("id"), "terms": forms, "cpc": grp})
            lens_info["neigh_groups"] = neigh_groups
            for i, lq in enumerate(lens_queries[:LENS_CALLS - 1]):
                lc, lerr = await lens.search_patents(lq["terms"], cpc=lq["cpc"], before=cutoff, size=100)
                lens_info["search_calls"] += 1
                new_keys = []
                for c in lc:
                    key = (c.pub_num or c.title).upper()
                    if key not in pool:
                        pool[key] = c
                        new_keys.append(c.pub_num)
                    if c.pub_num:
                        lens_pubs.append(c.pub_num)
                        seeds.append(c.pub_num)
                lens_info["searches"].append({"element": lq["element"], "terms": lq["terms"], "cpc": lq["cpc"], "returned": len(lc),
                                              "new": len(new_keys), "pubs": [c.pub_num for c in lc if c.pub_num], "error": lerr})
        except Exception as exc:
            lens_info["error"] = f"{type(exc).__name__}: {exc}"[:200]
    lens_pubs = list(dict.fromkeys(lens_pubs))
    seeds = list(dict.fromkeys(seeds))
    if os.environ.get("WIDE_SEED_ONLY") == "1":
        # M1 round 0 wants the queries and the paper neighbourhood only: the citation expansion,
        # Google similar and the CPC round are moves of their own, run from GOOD documents rather
        # than from every query hit, and repeating them here would spend the BigQuery budget twice.
        # The Reliance bridge patents are seeds here and nothing else, so in this mode they would
        # never reach M1's pool at all — and the bridges first reached 8 of h1h's 30 families.
        for pub in bridge_seeds:
            pool.setdefault(pub.upper(), Candidate(pub_num=pub, match_type="Patent",
                                                   sources=["reliance_bridge"], raw={"move": "reliance_bridge"}))
        return list(pool.values()), {"rounds": [{"round": 1, "queries": log, "n_queries": len(log),
                                                 "seeds": len(seeds), "pool_size": len(pool),
                                                 "neighbourhood": neigh_info, "bridge": bridge_info,
                                                 "lens": {k: v for k, v in lens_info.items() if k != "searches"},
                                                 "seed_only": True}],
                                     "mode": "wide", "elements": all_els,
                                     "candidates": [{"id": c["id"], "level": c["level"],
                                                     "n_elements": len(c["elements"])} for c in cands],
                                     "coverage_by_element": {}}
    expanded, info = await expand(seeds, set(pool), max_cited=MAX_CITED_LIGHT, before=cutoff, light=True, forward=True) if seeds else ([], {})
    dropped = set(info.get("seeds_after_cutoff") or [])
    for k in list(pool):
        if k in dropped:
            del pool[k]
    for c in expanded:
        pool.setdefault((c.pub_num or c.title).upper(), c)
    # Google's semantic neighbours of the seeds (channel "google_similar")
    try:
        sim_cands, sim_info = await similar_neighbours(seeds, set(pool), before=cutoff) if seeds else ([], {})
    except Exception as exc:
        sim_cands, sim_info = [], {"error": f"{type(exc).__name__}: {exc}"[:160]}
    for c in sim_cands:
        pool.setdefault((c.pub_num or c.title).upper(), c)
    # CPC round: the subclasses the citation neighbourhood is classified in
    # (expansion head carries cpc_codes), AND the core candidate's thing forms
    # main groups (H04N7), not subclasses: CPC=<subclass>/low returns nothing (gold probe)
    cpc_counts: dict[str, int] = {}
    for c in expanded:
        for code in ((c.raw or {}).get("bigquery") or {}).get("cpc_codes") or []:
            grp = str(code).split("/")[0]
            if len(grp) >= 5:
                cpc_counts[grp] = cpc_counts.get(grp, 0) + 1
    pred = list(dict.fromkeys(str(c).split("/")[0] for cand in cands for c in (cand.get("cpc_pred") or []) if len(str(c).split("/")[0]) >= 5))
    seed_top = [k for k, _ in sorted(cpc_counts.items(), key=lambda kv: -kv[1])]
    top_cpc = list(dict.fromkeys(pred[:1] + seed_top + pred[1:]))[:CPC_QUERIES]
    cpc_seeds = await _run(cpc_queries(cands, top_cpc, max_total=CPC_QUERIES)) if top_cpc else []
    cpc_seeds = [p for p in dict.fromkeys(cpc_seeds) if p not in set(seeds)]
    if cpc_seeds:
        more, info2 = await expand(cpc_seeds, set(pool), max_cited=MAX_CITED_LIGHT // 4, before=cutoff, light=True)
        dropped |= set(info2.get("seeds_after_cutoff") or [])
        for k in list(pool):
            if k in dropped:
                del pool[k]
        for c in more:
            pool.setdefault((c.pub_num or c.title).upper(), c)
        expanded = expanded + more
        seeds = seeds + [p for p in cpc_seeds if p not in dropped]
        info["cited_by_seed"] = {**(info.get("cited_by_seed") or {}), **(info2.get("cited_by_seed") or {})}
        info["cited_total"] = info.get("cited_total", 0) + info2.get("cited_total", 0)
    stats = {"round": 1, "mode": "wide", "n_queries": len(log), "gp_calls": budget.gp_calls,
             "serpapi_calls": budget.serp_calls, "gp_blocked": budget.gp_blocked,
             "seeds": len(seeds), "seeds_after_cutoff": sorted(dropped), "expanded": len(expanded),
             "cited_total": info.get("cited_total", 0), "cited_light": info.get("cited_light", 0),
             "cited_by_seed": info.get("cited_by_seed", {}), "expanded_pubs": sorted((c.pub_num or c.title).upper() for c in expanded),
             "cpc_top": top_cpc,
             "neighbourhood": {k: v for k, v in neigh_info.items()}, "neighbourhood_papers": len(papers),
             "bridge": {"oa_ids": bridge_info["oa_ids"], "patents": bridge_info["patents"], "error": bridge_info.get("error")},
             "bridge_pubs": sorted(bridge_seeds), "bridge_by_paper": bridge_info["by_paper"], "neigh_oa_ids": oa_ids,
             "lens": {k: v for k, v in lens_info.items() if k != "searches"}, "lens_searches": lens_info.get("searches", []),
             "lens_pubs": sorted(lens_pubs),
             "similar_total": sim_info.get("similar_total", 0), "similar_added": len(sim_cands),
             "similar_by_seed": sim_info.get("by_seed", {}),
             "similar_pubs": sorted((c.pub_num or c.title).upper() for c in sim_cands),
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
    if LOOP_MODE == "moves":
        from .loop_moves import run_moves
        return await run_moves(state, serpapi_left, serpapi_take, event)
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
