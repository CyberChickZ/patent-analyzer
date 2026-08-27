"""Phase 3: Multi-channel prior art recall + Phase 3b: semantic ranking.

For P0 this is a single node wrapping the original search + ranking logic.
In P1 this becomes a subgraph with Map-Reduce for eval (Phase 4 only).
Phase 3 recall stays as asyncio.gather in a single node (channels are not homogeneous).
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from state import GraphState


async def search_node(state: GraphState) -> dict:
    """Phase 3 + 3b: multi-channel search, pool, dedup, rerank, download PDFs."""
    from app.llm import refine_search_query, summarize_failure
    from patent_analyzer.recall import arxiv as ch_arxiv
    from patent_analyzer.recall import openalex as ch_oa
    from patent_analyzer.recall import pool as recall_pool
    from patent_analyzer.recall import semantic_scholar as ch_ss
    from patent_analyzer.recall import serpapi as ch_serpapi
    from patent_analyzer.searcher import download_pdf

    summary = state["summary"]
    source_title = state.get("source_title", "")
    source_arxiv_id = state.get("source_arxiv_id", "")
    source_doi = state.get("source_doi", "")
    delegation = state.get("delegation", {})
    checklist = state.get("checklist", [])
    input_path = state.get("input_local_path", "")
    output_dir = state.get("output_dir", "/tmp/outputs/default")

    events = []

    def _event(kind: str, message: str, payload: dict | None = None):
        evt = {"ts": datetime.now(timezone.utc).isoformat(),
               "phase": "phase3", "kind": kind, "message": message}
        if payload:
            evt["payload"] = payload
        events.append(evt)

    _event("start", "Multi-channel prior art recall")

    queries = delegation
    n_query_groups = len(queries.get("groups", []))
    _event("info", f"Built queries for {n_query_groups} groups")

    # Short/long queries for different channels
    def _short_query() -> str:
        if source_title and len(source_title) > 8:
            return " ".join(source_title.split())[:200]
        first_sent = (summary or "").split(".")[0]
        return " ".join((first_sent[:200] if first_sent else (summary or "")[:200]).split())

    recall_query_short = _short_query()

    # SerpAPI throttling + per-job budget (free tier is 250 searches/month)
    serpapi_lock = asyncio.Semaphore(1)
    SERPAPI_COOLDOWN = 1.5
    serpapi_budget = {"left": int(os.environ.get("SERPAPI_MAX_CALLS_PER_JOB", "8"))}

    def _serpapi_take() -> bool:
        if serpapi_budget["left"] <= 0:
            return False
        serpapi_budget["left"] -= 1
        return True

    def _serpapi_quota_status():
        try:
            return ch_serpapi.quota_status()
        except Exception:
            return []

    async def _serpapi_throttled_patent(q: str):
        async with serpapi_lock:
            res = await ch_serpapi.search_patents(q, max_pages=1)
            await asyncio.sleep(SERPAPI_COOLDOWN)
            return res

    async def _serpapi_throttled_scholar(q: str):
        async with serpapi_lock:
            res = await ch_serpapi.search_scholar(q, max_pages=1)
            await asyncio.sleep(SERPAPI_COOLDOWN)
            return res

    patent_queries = [q for g in queries.get("groups", [])
                      for q in (g.get("patent_queries") or [])[:2]]
    gp_failed: list[str] = []
    gp_done = asyncio.Event()

    async def run_google_patents():
        """Direct Google Patents XHR (free). Queries it cannot serve (blocked
        or errored) are handed to the SerpAPI channel via gp_failed."""
        from patent_analyzer.recall import google_patents as ch_gp
        out, errs = [], []
        try:
            for q in patent_queries:
                if ch_gp.is_blocked():
                    errs.append({"query": q, "error": "google_patents: blocked"})
                    gp_failed.append(q)
                    continue
                cands, err = await ch_gp.search(q, num=20)
                if err or not cands:
                    errs.append({"query": q, "error": err or "no results"})
                    gp_failed.append(q)
                out.extend(cands)
        finally:
            gp_done.set()
        return out, errs

    async def run_serpapi_patents():
        """SerpAPI patents, budgeted; spends calls only on queries the direct
        channel could not serve."""
        await gp_done.wait()
        out, errs = [], []
        for q in gp_failed:
            if not _serpapi_take():
                errs.append({"query": q, "error": "serpapi budget exhausted"})
                return out, errs
            cands, err = await _serpapi_throttled_patent(q)
            if err:
                errs.append({"query": q, "error": err})
            out.extend(cands)
        return out, errs

    async def run_serpapi_scholar():
        out, errs = [], []
        for group in queries.get("groups", []):
            for q in (group.get("paper_queries") or [])[:2]:
                if not _serpapi_take():
                    errs.append({"query": q, "error": "serpapi budget exhausted"})
                    return out, errs
                cands, err = await _serpapi_throttled_scholar(q)
                if err:
                    errs.append({"query": q, "error": err})
                out.extend(cands)
        return out, errs

    def _paper_queries() -> list[str]:
        """Plain-keyword queries for the paper APIs (free, unlimited): the
        title/summary query plus ≤2 per candidate invention from the loop's
        facets (names + things; things + places)."""
        out = [recall_query_short]
        by_cand: dict[str, list[dict]] = {}
        for e in loop_stats.get("elements") or []:
            by_cand.setdefault(e.get("candidate") or "inv1", []).append(e)
        for els in by_cand.values():
            def _u(k, n):
                seen = []
                for e in els:
                    for t in (e.get("facets") or {}).get(k) or []:
                        if t not in seen:
                            seen.append(t)
                return seen[:n]
            names, things, places = _u("named", 2), _u("thing", 3), _u("place", 2)
            if names or things:
                out.append(" ".join(names + things)[:200])
            if things and places:
                out.append(" ".join(things + places)[:200])
        return list(dict.fromkeys(q for q in out if q.strip()))[:int(os.environ.get("PAPER_QUERIES_MAX", "5"))]

    async def run_semantic_scholar():
        # S2 free tier 429s under bursts; each 429 costs up to 80 s of backoff in
        # ch_ss._get, so the wide queries are capped (PAPER_QUERIES_MAX) and a
        # rate-limited query is not retried here
        out, errs = [], []
        for q in _paper_queries():
            cands, err = await ch_ss.search(q, limit=50)
            out.extend(cands)
            if err:
                errs.append({"query": q, "error": err})
        return out, errs

    async def run_openalex():
        # OpenAlex free tier rate-limits bursts (429 on the 2nd query in the
        # h1d smoke): one query per second, one retry after a pause
        out, errs = [], []
        for i, q in enumerate(_paper_queries()):
            if i:
                await asyncio.sleep(1.0)
            cands, err = await ch_oa.search_works(q, limit=50)
            if err and "429" in err:
                await asyncio.sleep(6.0)
                cands, err = await ch_oa.search_works(q, limit=50)
            out.extend(cands)
            if err:
                errs.append({"query": q, "error": err})
        return out, errs

    async def run_arxiv():
        cands, err = await ch_arxiv.search(recall_query_short, limit=50)
        return cands, ([{"query": recall_query_short, "error": err}] if err else [])

    async def run_bigquery_patents():
        """SEARCH-indexed recall over our own abstracts table, one call per
        query group using its quoted terms (the old LIKE scan cost 327 GiB)."""
        import traceback
        try:
            from patent_analyzer.recall.bigquery_patents import search_abstracts
            # one ranked call per job (each call reads 20-90 GiB); terms = the
            # anchor phrases across groups, longest first as a rarity proxy
            terms = []
            for group in queries.get("groups", []):
                for q in (group.get("patent_queries") or [])[:1]:
                    terms += re.findall(r'"([^"]{3,60})"', q)
            terms = sorted(dict.fromkeys(t.strip().rstrip("*") for t in terms if t.strip()), key=len, reverse=True)[:10]
            if not terms:
                terms = [w for w in recall_query_short.split() if len(w) > 4][:8]
            cands, err = await search_abstracts(terms, limit=40, before=state.get("date_cutoff"))
            return cands, ([{"query": " | ".join(terms), "error": err}] if err else [])
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[BQ CRASH] {exc}\n{tb}")
            _event("channel_crashed", f"bigquery_patents CRASH: {exc}")
            return [], [{"query": "crash", "error": f"{type(exc).__name__}: {exc}"}]

    # 8th channel first: the agentic loop (per-element boolean search + BQ
    # expansion) gets SerpAPI budget priority over the broad legacy channels
    loop_stats: dict = {}
    loop_cands: list = []
    try:
        from patent_analyzer.agentic.loop import run_loop
        loop_cands, loop_stats = await run_loop(
            state, lambda: serpapi_budget["left"], _serpapi_take,
            lambda kind, msg, payload=None: _event(kind, msg, payload))
    except Exception as exc:
        _event("channel_crashed", f"agentic_loop: {type(exc).__name__}: {exc}")

    async def run_agentic_loop():
        return loop_cands, []

    # Launch all channels in parallel (7 channels)
    channel_specs = [
        ("agentic_loop", run_agentic_loop),
        ("google_patents", run_google_patents),
        ("serpapi_patents", run_serpapi_patents),
        ("serpapi_scholar", run_serpapi_scholar),
        ("semantic_scholar", run_semantic_scholar),
        ("openalex", run_openalex),
        ("arxiv", run_arxiv),
        ("bigquery_patents", run_bigquery_patents),
    ]
    import time as _time

    async def _timed(name, fn):
        t0 = _time.monotonic()
        try:
            return await fn(), _time.monotonic() - t0
        except Exception as exc:
            return exc, _time.monotonic() - t0

    gathered = await asyncio.gather(*(_timed(n, f) for n, f in channel_specs))

    channel_results: dict[str, list] = {}
    for (name, _), (result, secs) in zip(channel_specs, gathered):
        if isinstance(result, Exception):
            _event("channel_crashed", f"{name}: {type(result).__name__}: {result}", {"channel": name, "seconds": round(secs, 1)})
            channel_results[name] = []
            continue
        cands, errs = result
        channel_results[name] = cands
        _event("channel_done", f"{name}: {len(cands)} raw candidates in {secs:.0f}s",
               {"channel": name, "n": len(cands), "seconds": round(secs, 1), "errors": errs[:5]})
        for e in errs:
            if any(k in str(e.get("error", "")) for k in ("blocked", "429", "budget", "Sorry")):
                _event("channel_limited", f"{name}: {str(e.get('error', ''))[:120]}")

    # Pool & dedupe
    pooled = recall_pool.pool_and_dedupe(channel_results)
    _event("info", f"Pool: {len(pooled)} unique candidates")

    if not pooled:
        return {
            "phase": "phase3",
            "status": "failed_recall",
            "error": "All recall channels returned 0 candidates",
            "search_results": [],
            "ranked_candidates": [],
            "events": events,
            "phase_results": {"phase3": {"status": "completed", "data": {"total": 0}}},
        }

    # Citation chaining: hop 1 from the top 5 papers (refs + cits), hop 2
    # (refs only) from the 3 most-cited hop-1 papers — free S2 calls
    if len(pooled) >= 3:
        top_for_chaining = sorted(pooled, key=lambda c: c.source_score, reverse=True)[:5]
        chain_results: dict[str, list] = {}
        hop1: list = []
        for cand in top_for_chaining:
            ss_id = ((cand.raw or {}).get("semantic_scholar") or {}).get("paperId") or cand.arxiv_id or ""
            if not ss_id:
                continue
            try:
                ref_cands, _ = await ch_ss.references(ss_id, limit=20)
                cit_cands, _ = await ch_ss.citations(ss_id, limit=20)
                chain_results[f"ref_{ss_id[:16]}"] = ref_cands
                chain_results[f"cit_{ss_id[:16]}"] = cit_cands
                hop1 += ref_cands + cit_cands
            except Exception:
                pass
        def _cites(c):
            return int(((c.raw or {}).get("semantic_scholar") or {}).get("citationCount") or 0)
        for cand in sorted(hop1, key=_cites, reverse=True)[:3]:
            ss_id = ((cand.raw or {}).get("semantic_scholar") or {}).get("paperId") or ""
            if not ss_id or f"ref_{ss_id[:16]}" in chain_results:
                continue
            try:
                ref_cands, _ = await ch_ss.references(ss_id, limit=20)
                chain_results[f"ref2_{ss_id[:16]}"] = ref_cands
            except Exception:
                pass
        if chain_results:
            merged = dict(channel_results)
            merged.update(chain_results)
            before = len(pooled)
            pooled = recall_pool.pool_and_dedupe(merged)
            _event("info", f"Citation chaining added {len(pooled) - before} candidates")

    # Convert to legacy doc dicts
    all_docs = recall_pool.candidates_to_legacy_docs(pooled)

    # ── Phase 3b: Semantic ranking ──
    _event("info", "Phase 3b: semantic ranking")
    _STOP = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "by", "to",
             "with", "from", "is", "at", "as", "its", "via", "using", "based"}

    def title_similarity(a: str, b: str) -> float:
        ta = set(a.lower().split()) - _STOP
        tb = set(b.lower().split()) - _STOP
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta | tb), 1)

    # Filter self-citations
    before_filter = len(all_docs)
    if source_title:
        all_docs = [d for d in all_docs
                    if title_similarity(d.get("title", ""), source_title) < 0.75]
    if source_arxiv_id:
        all_docs = [d for d in all_docs
                    if source_arxiv_id not in (d.get("arxiv_id") or "")]
    if source_doi:
        all_docs = [d for d in all_docs
                    if source_doi not in (d.get("doi") or "")]
    filtered = before_filter - len(all_docs)
    if filtered:
        _event("info", f"Filtered {filtered} self-citations")

    # Precision stage (wide mode): embedding shortlist + LLM screen → ≤60,
    # then the semantic rerank orders what survived
    prune_stats: dict = {}
    pruned_docs: list[dict] = []
    if os.environ.get("PRUNE", "1") == "1" and loop_stats.get("mode") == "wide" and loop_stats.get("elements"):
        try:
            from patent_analyzer.agentic.prune import prune as _prune
            _t0 = _time.monotonic()
            pruned_docs, prune_stats = await _prune(loop_stats.get("candidates") or [], loop_stats["elements"], all_docs, summary=summary)
            prune_stats["seconds"] = round(_time.monotonic() - _t0, 1)
            _event("prune_done", f"prune: pool {prune_stats.get('pool')} → embed {prune_stats.get('stage1_out')} "
                                 f"→ llm {prune_stats.get('stage2_worth')} worth reading, kept {len(pruned_docs)} "
                                 f"({prune_stats.get('stage2_calls')} calls)", prune_stats)
        except Exception as exc:
            _event("channel_crashed", f"prune: {type(exc).__name__}: {exc}")
            pruned_docs, prune_stats = [], {"error": str(exc)[:200]}

    # Semantic rerank. Everything the prune kept is deep-read and reported (leader, 2026-09-18:
    # h1h lost 8 of the 16 gold families that survived the screen to the old top-30 cut); the
    # embedding order is only a tie-break among documents the claims screen already ranked.
    from patent_analyzer.semantic_search import rerank_docs
    rank_limit = int(os.environ.get("RERANK_LIMIT", "60")) if pruned_docs else 30
    ranked = rerank_docs(summary, pruned_docs or all_docs, limit=rank_limit)
    rank_of = {id(d): i + 1 for i, d in enumerate(ranked)}

    # Ensure BigQuery patent candidates aren't lost after rerank
    ranked_titles = {d.get("title", "").lower() for d in ranked}
    bq_missed = [d for d in all_docs
                 if "bigquery_patents" in (d.get("sources") or [])
                 and d.get("title", "").lower() not in ranked_titles]
    if bq_missed:
        ranked.extend(bq_missed)
        _event("info", f"Injected {len(bq_missed)} BigQuery patents missed by rerank")

    _event("info", f"Ranked: top {len(ranked)} candidates")

    # Enrich ranked papers missing abstracts via OpenAlex title lookup
    # (SerpAPI scholar results only carry a 1-2 sentence snippet)
    from patent_analyzer.fetch_abstracts import search_paper as _oa_lookup
    need_abs = [d for d in ranked
                if d.get("match_type") != "Patent"
                and not (d.get("abstract") or "").strip()
                and d.get("title")][:20]
    if need_abs:
        abs_sem = asyncio.Semaphore(4)

        async def _fill_abstract(d):
            async with abs_sem:
                try:
                    info = await asyncio.to_thread(_oa_lookup, d["title"])
                except Exception:
                    return
                if info and info.get("abstract"):
                    d["abstract"] = info["abstract"]
                    for k in ("authors", "doi"):
                        if not d.get(k) and info.get(k):
                            d[k] = info[k]

        await asyncio.gather(*(_fill_abstract(d) for d in need_abs))
        filled = sum(1 for d in need_abs if (d.get("abstract") or "").strip())
        _event("info", f"OpenAlex abstract enrichment: {filled}/{len(need_abs)} filled")

    # Download top PDFs
    job_dir = Path(output_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    MAX_DOWNLOADS = 30
    download_count = 0
    for i, doc in enumerate(ranked[:MAX_DOWNLOADS]):
        pdf_url = recall_pool.resolve_pdf_url(doc)
        if not pdf_url:
            continue
        try:
            fname = f"prior_art_{i:03d}.pdf"
            local = download_pdf(pdf_url, job_dir, fname)
            if local:
                doc["local_pdf"] = local
                download_count += 1
        except Exception:
            pass
    _event("info", f"Downloaded {download_count}/{min(len(ranked), MAX_DOWNLOADS)} PDFs")

    patent_count = sum(1 for d in all_docs if d.get("match_type") == "Patent")
    paper_count = sum(1 for d in all_docs if d.get("match_type") != "Patent")

    return {
        "phase": "phase3",
        "search_results": [{"title": d.get("title", ""), "match_type": d.get("match_type", "")}
                           for d in all_docs],
        "ranked_candidates": ranked,
        "search_stats": {
            "total_patents": patent_count,
            "total_papers": paper_count,
            "total_unique": len(all_docs),
            "active_channels": len([v for v in channel_results.values() if v]),
            "downloaded": download_count,
            "pool": [{"pub_num": d.get("pub_num", ""), "sources": d.get("sources", []),
                      "match_type": d.get("match_type", "")} for d in all_docs],
            "loop_rounds": loop_stats.get("rounds", []),
            "loop_elements": loop_stats.get("elements", []),
            "loop_mode": loop_stats.get("mode", "elements"),
            "coverage_by_element": loop_stats.get("coverage_by_element", {}),
            "prune": prune_stats,
            "pruned": [{"pub_num": d.get("pub_num", ""), "sources": d.get("sources", []),
                        "match_type": d.get("match_type", ""), "elements": d.get("prune_elements", [])}
                       for d in pruned_docs],
            # per-document funnel: embedding score / shortlist / LLM verdict + reason / final rank
            "funnel_docs": [{"pub_num": d.get("pub_num", ""), "title": (d.get("title") or "")[:100],
                             "match_type": d.get("match_type", ""), "sources": d.get("sources", []),
                             "cos": round(float(d.get("prune_cos", 0.0)), 4), "best_element": d.get("prune_best_element", ""),
                             "stage1": bool(d.get("prune_stage1")), "worth_reading": d.get("prune_worth_reading"),
                             "elements": d.get("prune_elements", []), "reason": d.get("prune_reason", ""),
                             "rank": rank_of.get(id(d))}
                            for d in all_docs] if prune_stats else [],
            "serpapi_quota": _serpapi_quota_status(),
        },
        "events": events,
        "phase_results": {"phase3": {
            "status": "completed",
            "data": {"patents": patent_count, "papers": paper_count, "total": len(all_docs), "ranked": len(ranked), "downloaded": download_count},
        }},
    }
