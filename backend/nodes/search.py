"""Phase 3: Multi-channel prior art recall + Phase 3b: semantic ranking.

For P0 this is a single node wrapping the original search + ranking logic.
In P1 this becomes a subgraph with Map-Reduce for eval (Phase 4 only).
Phase 3 recall stays as asyncio.gather in a single node (channels are not homogeneous).
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from state import GraphState

# Wall-clock budget per recall channel, in seconds. A channel that blows it is
# dropped (empty result + a channel_timeout event + a "timeout" row in
# search_stats["channel_health"]); the job goes on with the other channels.
# Override one channel with SEARCH_TIMEOUT_<CHANNEL>, all of them with
# SEARCH_CHANNEL_TIMEOUT_S.
_CHANNEL_TIMEOUT_DEFAULT = 600.0
_CHANNEL_TIMEOUT_OVERRIDES = {
    "agentic_loop": 2400.0,      # many rounds × (LLM + boolean search + BQ expansion)
    "bigquery_patents": 900.0,   # dry-run + SEARCH-indexed scan
    "fulltext_bq": 900.0,        # ceil(delivered/300) bucket-pruned claims lookups
}

# Total time allowed for the prior-art PDF downloads, all documents together.
_PDF_DOWNLOAD_BUDGET_S = float(os.environ.get("PDF_DOWNLOAD_BUDGET_S", "300"))


def _channel_timeout(name: str) -> float:
    env = os.environ.get(f"SEARCH_TIMEOUT_{name.upper()}") or os.environ.get("SEARCH_CHANNEL_TIMEOUT_S")
    if env:
        return float(env)
    return _CHANNEL_TIMEOUT_OVERRIDES.get(name, _CHANNEL_TIMEOUT_DEFAULT)


async def search_node(state: GraphState) -> dict:
    """Phase 3 + 3b: multi-channel search, pool, dedup, rerank, download PDFs."""
    from app.llm import refine_search_query, summarize_failure
    from patent_analyzer.recall import arxiv as ch_arxiv
    from patent_analyzer.recall import openalex as ch_oa
    from patent_analyzer.recall import pool as recall_pool
    from patent_analyzer.recall import semantic_scholar as ch_ss
    from patent_analyzer.recall import serpapi as ch_serpapi
    from patent_analyzer.searcher import download_pdf
    from patent_analyzer import fulltext as ft_oa

    summary = state["summary"]
    source_title = state.get("source_title", "")
    source_arxiv_id = state.get("source_arxiv_id", "")
    source_doi = state.get("source_doi", "")
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

    # Short/long queries for different channels
    def _short_query() -> str:
        if source_title and len(source_title) > 8:
            return " ".join(source_title.split())[:200]
        first_sent = (summary or "").split(".")[0]
        return " ".join((first_sent[:200] if first_sent else (summary or "")[:200]).split())

    recall_query_short = _short_query()

    # SerpAPI per-job budget (free tier is 250 searches/month). The agentic
    # loop is the only spender left and does its own serialisation.
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
            # one ranked call per job (each call reads 20-90 GiB). The terms used
            # to come from the delegation groups; nothing has written delegation
            # since the extraction subgraph replaced generate_search_queries, so
            # only the fallback ever ran.
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
    loop_timed_out = False
    try:
        from patent_analyzer.agentic.loop import run_loop
        loop_cands, loop_stats = await asyncio.wait_for(run_loop(
            state, lambda: serpapi_budget["left"], _serpapi_take,
            lambda kind, msg, payload=None: _event(kind, msg, payload)),
            timeout=_channel_timeout("agentic_loop"))
    except asyncio.TimeoutError:
        loop_timed_out = True
        _event("channel_timeout", f"agentic_loop: no result within {_channel_timeout('agentic_loop'):.0f}s "
                                  "— continuing with the remaining channels")
    except Exception as exc:
        _event("channel_crashed", f"agentic_loop: {type(exc).__name__}: {exc}")

    async def run_agentic_loop():
        return loop_cands, []

    # Launch all channels in parallel
    channel_specs = [
        ("agentic_loop", run_agentic_loop),
        ("semantic_scholar", run_semantic_scholar),
        ("openalex", run_openalex),
        ("arxiv", run_arxiv),
        ("bigquery_patents", run_bigquery_patents),
    ]
    import time as _time

    async def _timed(name, fn):
        """Every channel gets a wall-clock budget. Without one a single hung
        socket (BigQuery `job.result()`, a stalled TLS handshake) holds the
        whole `gather` — and therefore the job — open forever; the point of
        eight channels is that losing one is survivable."""
        t0 = _time.monotonic()
        try:
            return await asyncio.wait_for(fn(), timeout=_channel_timeout(name)), _time.monotonic() - t0
        except asyncio.TimeoutError:
            return asyncio.TimeoutError(f"no result within {_channel_timeout(name):.0f}s"), _time.monotonic() - t0
        except Exception as exc:
            return exc, _time.monotonic() - t0

    gathered = await asyncio.gather(*(_timed(n, f) for n, f in channel_specs))

    # Per-channel health, carried in search_stats so the *report* can say which
    # channels degraded (events are a side channel and never reach results.json).
    channel_health: list[dict] = []
    channel_results: dict[str, list] = {}
    from patent_analyzer import metering
    for (name, _), (result, secs) in zip(channel_specs, gathered):
        if isinstance(result, Exception):
            kind = "channel_timeout" if isinstance(result, asyncio.TimeoutError) else "channel_crashed"
            detail = f"{type(result).__name__}: {result}"
            _event(kind, f"{name}: {detail}", {"channel": name, "seconds": round(secs, 1)})
            # A channel lost whole is the biggest degradation the pipeline can
            # suffer and results.json never said so — only the event stream did.
            metering.incident(name, metering.FAILED, f"{kind} after {secs:.0f}s: {detail}")
            channel_results[name] = []
            channel_health.append({"channel": name, "status": "timeout" if kind == "channel_timeout" else "crashed",
                                   "n": 0, "seconds": round(secs, 1), "detail": detail[:200], "errors": []})
            continue
        cands, errs = result
        channel_results[name] = cands
        _event("channel_done", f"{name}: {len(cands)} raw candidates in {secs:.0f}s",
               {"channel": name, "n": len(cands), "seconds": round(secs, 1), "errors": errs[:5]})
        limited = [str(e.get("error", "")) for e in errs
                   if any(k in str(e.get("error", "")) for k in ("blocked", "429", "budget", "quota", "Sorry"))]
        for msg in limited:
            _event("channel_limited", f"{name}: {msg[:120]}")
            metering.incident(name, metering.DEGRADED, msg[:160])
        status = "limited" if limited else ("ok" if cands else ("errored" if errs else "empty"))
        channel_health.append({"channel": name, "status": status, "n": len(cands), "seconds": round(secs, 1),
                               "detail": (limited[0][:200] if limited else
                                          (str(errs[0].get("error", ""))[:200] if errs and not cands else "")),
                               "errors": [str(e.get("error", ""))[:160] for e in errs[:5]]})
    if loop_timed_out:
        for h in channel_health:
            if h["channel"] == "agentic_loop":
                h.update(status="timeout", detail=f"no result within {_channel_timeout('agentic_loop'):.0f}s")

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
            # still report *why* every channel came back empty
            "search_stats": {"total_patents": 0, "total_papers": 0, "total_unique": 0,
                             "active_channels": 0, "downloaded": 0, "channel_health": channel_health,
                             "serpapi_quota": _serpapi_quota_status()},
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

    # M1 (LOOP_MODE=moves): the GOOD set is the selection — documents whose claims a model could
    # point at, element by element. No pool ranking, no prune; delivery is the GOOD order, capped.
    if loop_stats.get("mode") == "moves":
        good = loop_stats.get("good") or []
        by_pub = {(d.get("pub_num") or "").upper(): d for d in all_docs}
        deliver = int(os.environ.get("M1_DELIVER", "120"))
        ranked = []
        for g in good[:deliver]:
            d = by_pub.get((g.get("pub_num") or "").upper())
            if d:
                d["good_touches"] = g.get("touches")
                d["good_strong"] = bool(g.get("strong"))
                ranked.append(d)
        _event("info", f"M1: delivering {len(ranked)} GOOD documents of {len(good)} "
                       f"({loop_stats.get('n_strong', 0)} strong, {loop_stats.get('rounds')} rounds, "
                       f"stop: {loop_stats.get('stop')})")
        prune_stats = {"mode": "moves", "good": len(good), "strong": loop_stats.get("n_strong", 0),
                       "delivered": len(ranked), "uncovered": loop_stats.get("uncovered", [])}
        rank_of = {id(d): i + 1 for i, d in enumerate(ranked)}
        pruned_docs = ranked

    # Semantic rerank. Everything the prune kept is deep-read and reported (leader, 2026-09-18:
    # h1h lost 8 of the 16 gold families that survived the screen to the old top-30 cut); the
    # embedding order is only a tie-break among documents the claims screen already ranked.
    if loop_stats.get("mode") != "moves":
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

    # ── Deep-read text ──
    # Our own BigQuery copy first. A prior-art PDF download succeeded 0-6 times
    # out of 30 on the M2 e2e runs (mostly HTTP 503), and every document it
    # missed reached Phase 4 as `no_content` — counted as "checked" while
    # nothing had been read. The patents in `amie_patents` do not depend on a
    # third party being up: title + abstract from `pubs`, claims_text from
    # `claims` (US only — there is no description column anywhere in the
    # dataset, see bigquery_patents.hydrate_full_text).
    ft_t0 = _time.monotonic()
    ft_stats: dict = {}
    try:
        from patent_analyzer.recall.bigquery_patents import hydrate_full_text
        ft_stats = await asyncio.wait_for(hydrate_full_text(ranked),
                                          timeout=_channel_timeout("fulltext_bq"))
    except asyncio.TimeoutError:
        ft_stats = {"errors": [f"no result within {_channel_timeout('fulltext_bq'):.0f}s"]}
        _event("channel_timeout", "fulltext_bq: timed out — the deep read falls back to PDFs and abstracts")
    except Exception as exc:
        ft_stats = {"errors": [f"{type(exc).__name__}: {exc}"[:200]]}
        _event("channel_crashed", f"fulltext_bq: {type(exc).__name__}: {exc}")
    ft_stats["seconds"] = round(_time.monotonic() - ft_t0, 1)
    n_claims = int(ft_stats.get("with_claims", 0))
    _event("info", f"BigQuery full text: claims for {n_claims}/{ft_stats.get('asked', 0)} patent candidates "
                   f"in {ft_stats.get('chunks', 0)} chunk(s), {ft_stats['seconds']:.0f}s")
    channel_health.append({
        "channel": "fulltext_bq",
        "status": "errored" if ft_stats.get("errors") and not n_claims else ("ok" if n_claims else "empty"),
        "n": n_claims, "seconds": ft_stats["seconds"],
        "detail": (ft_stats.get("errors") or [""])[0][:200] if ft_stats.get("errors") else
                  f"claims for {n_claims} of {ft_stats.get('asked', 0)} patent candidates "
                  f"(amie_patents.claims is US-only and has no description column)",
        "errors": [str(e)[:160] for e in (ft_stats.get("errors") or [])[:5]]})

    # PDFs for what BigQuery cannot serve: papers, and patents with no claims
    # text. Downloading a PDF for a patent whose claims we already hold buys
    # nothing and costs up to 60 s of the shared budget.
    job_dir = Path(output_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    MAX_DOWNLOADS = 30
    download_count = 0
    # download_pdf retries once with a 30 s socket timeout, so 30 documents can
    # cost 30 min on their own. Downloads are an optimisation (evaluation falls
    # back to the abstract), so they get a shared budget and stop when it runs out.
    dl_t0 = _time.monotonic()
    dl_skipped = 0
    dl_not_needed = 0
    dl_no_url = 0
    dl_failed = 0
    dl_cached = 0
    need_pdf = []
    for i, doc in enumerate(ranked):
        if (doc.get("claims_text") or "").strip():
            dl_not_needed += 1
            continue
        need_pdf.append((i, doc))

    # Papers get an open-access resolution pass first. `resolve_pdf_url` only
    # repeats what a channel happened to put on the document; this walks the
    # tiers (arXiv, the OpenAlex/S2 OA fields, then Unpaywall) and records which
    # one answered, so the report can say per document where its text came from
    # instead of leaving a title-only evaluation looking like a read one. There
    # is no paywalled tier on purpose — see patent_analyzer/fulltext.py.
    papers = [(i, doc) for i, doc in need_pdf[:MAX_DOWNLOADS] if doc.get("match_type") != "Patent"]
    ft_oa_t0 = _time.monotonic()
    oa_patches: list[dict] = []
    ft_rows: list[dict] = []
    try:
        oa_patches = await ft_oa.resolve_many([d for _, d in papers])
        for (_, doc), patch in zip(papers, oa_patches):
            doc.update(patch)
    except Exception as exc:
        _event("channel_crashed", f"fulltext_oa: {type(exc).__name__}: {exc}")
    for i, doc in enumerate(ranked):
        if (doc.get("claims_text") or "").strip():
            doc["fulltext_download"] = "not_needed"
    for i, doc in need_pdf[:MAX_DOWNLOADS]:
        pdf_url = doc.get("fulltext_url") or recall_pool.resolve_pdf_url(doc)
        if not pdf_url:
            dl_no_url += 1
            doc["fulltext_download"] = "no_url"
            continue
        if _time.monotonic() - dl_t0 > _PDF_DOWNLOAD_BUDGET_S:
            dl_skipped += 1
            doc["fulltext_download"] = "skipped_budget"
            continue
        try:
            fname = f"prior_art_{i:03d}.pdf"
            # GCS, keyed by DOI, is shared across jobs: the same examiner-cited
            # paper comes back run after run, and the second run should not ask
            # the publisher again.
            doi = doc.get("doi") or ft_oa.normalise_doi(doc.get("pub_num") or "")
            local = None
            cached = await asyncio.to_thread(ft_oa.cache_get, doi) if doi else None
            if cached:
                (job_dir / fname).write_bytes(cached)
                local = str(job_dir / fname)
                dl_cached += 1
            else:
                local = await asyncio.to_thread(download_pdf, pdf_url, job_dir, fname)
                if local and doi:
                    await asyncio.to_thread(ft_oa.cache_put, doi, Path(local).read_bytes())
            if local:
                doc["local_pdf"] = local
                doc["fulltext_download"] = "cached" if cached else "ok"
                download_count += 1
            else:
                dl_failed += 1
                doc["fulltext_download"] = "failed"
        except Exception:
            dl_failed += 1
            doc["fulltext_download"] = "failed"
    _event("info", f"Downloaded {download_count}/{min(len(need_pdf), MAX_DOWNLOADS)} PDFs "
                   f"({dl_cached} served from the GCS cache, {dl_not_needed} documents already had "
                   f"BigQuery claims, {dl_no_url} offered no URL, {dl_failed} failed)")

    # Resolution and possession are two different numbers, and the report gets
    # both. Measured on the 17 examiner-cited NPL gold: 6 resolved to an
    # open-access URL and 0 of them returned a PDF — publisher hosts answer a
    # plain HTTP client with a 403 (Cloudflare) and PMC now puts a proof-of-work
    # challenge in front of the file. `fulltext_tier` therefore keeps saying
    # which tier answered, and `fulltext_download` says what came back, so a
    # resolved-but-unfetchable paper cannot be counted as read.
    if papers:
        docs_only = [d for _, d in papers]
        tiers = ft_oa.tier_counts(oa_patches)
        read = ft_oa.read_counts(docs_only, oa_patches)
        n_resolved = tiers.get("arxiv", 0) + tiers.get("oa", 0)
        n_read = sum(read.values())
        _event("info", f"Open-access full text: {n_resolved}/{len(papers)} papers resolved to a copy "
                       f"(arXiv {tiers.get('arxiv', 0)}, OA {tiers.get('oa', 0)}, "
                       f"abstract-only {tiers.get('abstract_only', 0)}); {n_read} returned a PDF")
        channel_health.append({
            "channel": "fulltext_oa",
            "status": "ok" if n_read else ("limited" if n_resolved else "empty"),
            "n": n_read, "seconds": round(_time.monotonic() - ft_oa_t0, 1),
            "detail": f"{n_resolved} of {len(papers)} papers resolved to an open-access copy "
                      f"(arXiv {tiers.get('arxiv', 0)}, OA {tiers.get('oa', 0)}) and {n_read} of those "
                      f"returned a readable PDF. Paywalled copies are not fetched: OSU Libraries' "
                      f"Responsible Use policy forbids programmatic downloading of licensed content, "
                      f"so the rest are listed for manual download instead.", "errors": []})
        ft_rows = ft_oa.manifest_rows(docs_only, oa_patches)
        if ft_rows:
            try:
                (job_dir / "manual_fulltext_manifest.md").write_text(ft_oa.manifest_markdown(ft_rows))
            except Exception:
                pass
    if dl_skipped:
        _event("channel_limited", f"pdf_download: {_PDF_DOWNLOAD_BUDGET_S:.0f}s budget spent, "
                                  f"{dl_skipped} PDFs not fetched (those documents are evaluated from their abstract)")
        channel_health.append({"channel": "pdf_download", "status": "limited", "n": download_count,
                               "seconds": round(_time.monotonic() - dl_t0, 1),
                               "detail": f"{dl_skipped} downloads skipped after the "
                                         f"{_PDF_DOWNLOAD_BUDGET_S:.0f}s budget", "errors": []})
    elif need_pdf and not download_count:
        # Silence here used to read exactly like a clean run. It is not one:
        # every one of these documents reaches Phase 4 with an abstract at best.
        channel_health.append({"channel": "pdf_download", "status": "errored", "n": 0,
                               "seconds": round(_time.monotonic() - dl_t0, 1),
                               "detail": f"0 of {min(len(need_pdf), MAX_DOWNLOADS)} PDF downloads succeeded "
                                         f"({dl_failed} failed, {dl_no_url} offered no URL) — those documents "
                                         f"are evaluated from their abstract or not at all", "errors": []})

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
            "delivered": len(ranked),
            # what the deep read will actually have to read, per document
            "text_source": {
                "bq_claims": sum(1 for d in ranked if (d.get("claims_text") or "").strip()),
                "pdf": download_count,
                "abstract_only": sum(1 for d in ranked
                                     if not (d.get("claims_text") or "").strip()
                                     and not d.get("local_pdf")
                                     and len((d.get("abstract") or d.get("snippet") or "").strip()) >= 120),
                "nothing": sum(1 for d in ranked
                               if not (d.get("claims_text") or "").strip()
                               and not d.get("local_pdf")
                               and len((d.get("abstract") or d.get("snippet") or "").strip()) < 120),
            },
            "fulltext_bq": ft_stats,
            "pdf_download": {"attempted": min(len(need_pdf), MAX_DOWNLOADS), "ok": download_count,
                             "failed": dl_failed, "no_url": dl_no_url, "from_cache": dl_cached,
                             "skipped_budget": dl_skipped, "not_needed": dl_not_needed},
            "fulltext_oa": {"papers": len(papers), **ft_oa.tier_counts(oa_patches),
                            "read": ft_oa.read_counts([d for _, d in papers], oa_patches),
                            "manifest": ft_rows},
            "pool": [{"pub_num": d.get("pub_num", ""), "sources": d.get("sources", []),
                      "match_type": d.get("match_type", "")} for d in all_docs],
            "loop_rounds": loop_stats.get("rounds", []) if loop_stats.get("mode") != "moves" else [],
            "move_rows": loop_stats.get("move_rows", []),
            "good": loop_stats.get("good", []),
            "coverage": loop_stats.get("coverage", {}),
            "uncovered": loop_stats.get("uncovered", []),
            "n_strong": loop_stats.get("n_strong", 0),
            "read": loop_stats.get("read", []),
            "stop": loop_stats.get("stop"),
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
                             "claims_read": "claims_worth_reading" in d, "claims_worth_reading": d.get("claims_worth_reading"),
                             "claims_elements": d.get("claims_elements", []), "claims_reason": d.get("claims_reason", ""),
                             "rank": rank_of.get(id(d))}
                            for d in all_docs] if prune_stats else [],
            "serpapi_quota": _serpapi_quota_status(),
            "channel_health": channel_health,
        },
        "events": events,
        "phase_results": {"phase3": {
            "status": "completed",
            "data": {"patents": patent_count, "papers": paper_count, "total": len(all_docs), "ranked": len(ranked), "downloaded": download_count},
        }},
    }
