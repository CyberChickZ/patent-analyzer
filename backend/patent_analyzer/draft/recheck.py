"""Re-check the draft's new limitations against a small search (leader_draft §2.5).

Runs inside the draft node, never loops the main graph (search_results /
eval_results are operator.add channels — a second pass would accumulate):

  1. the new / distinguishing limitations (origin dependent_hint, component_element,
     refinement) in the independent claim and the first 3 dependents
  2. facets for them: agentic.elements.attach_facets (1 LLM call; fallback = no LLM)
  3. queries: agentic.wide.candidate_queries on a pseudo-candidate {e0 + new
     limitations}, `element:` rows only, <= DRAFT_MAX_QUERIES (3)
  4. agentic.loop._search — Google direct first, SerpAPI only with credit left
     (search_stats.serpapi_left when the search node wrote it, else a quarter
     of SERPAPI_MAX_CALLS_PER_JOB)
  5. new = not in search_stats.pool; agentic.prune.stage1_embed keeps the top 5
  6. each new doc: full text from Google Patents (recall.google_patents.fetch_patent)
     else title + abstract, evaluated on the new limitations only with the
     Phase 4 evaluator + quote verifier (<= 5 LLM calls)
  7. coverage.recheck written on every rechecked limitation; when the limitation
     added to the independent claim is disclosed by a new document the next
     rechecked-and-undisclosed dependent limitation takes its place once, else
     strategy = unresolved.
The new documents never enter ranked_candidates / scoring_report.
"""

from __future__ import annotations

import os

MAX_QUERIES = int(os.environ.get("DRAFT_MAX_QUERIES", "3"))
MAX_EVAL = 5
NEW_ORIGINS = ("dependent_hint", "component_element", "refinement")


def new_limitations(draft: dict) -> list[dict]:
    """(claim, limitation) pairs to re-check: new-origin limitations in the
    primary independent claim and its first three dependents."""
    claims = draft.get("claims") or []
    if not claims:
        return []
    primary = claims[0]
    deps = [c for c in claims if c.get("depends_on") == primary.get("no")][:3]
    out = []
    for c in [primary] + deps:
        for l in c.get("limitations") or []:
            if l.get("origin") in NEW_ORIGINS:
                out.append({"claim": c, "lim": l})
    return out


def _serp_budget(state: dict) -> int:
    left = (state.get("search_stats") or {}).get("serpapi_left")
    if isinstance(left, int):
        return max(0, min(left, MAX_QUERIES))
    return max(0, int(os.environ.get("SERPAPI_MAX_CALLS_PER_JOB", "8")) // 4)


def _cutoff(state: dict) -> str | None:
    cutoff = str(state.get("date_cutoff") or "")
    return f"priority:{cutoff}" if cutoff.isdigit() and len(cutoff) == 8 else None


async def _fetch_text(doc: dict) -> tuple[str, str]:
    """Full text for a new Google Patents hit, else what the hit carried."""
    pub = doc.get("pub_num") or ""
    if pub and doc.get("match_type") == "Patent":
        try:
            from patent_analyzer.recall import google_patents as gp
            page = await gp.fetch_patent(pub)
        except Exception:
            page = None
        if page and (page.get("claims") or page.get("description")):
            text = "\n".join(x for x in (page.get("abstract", ""), "[CLAIMS]", "\n".join(page.get("claims") or []),
                                         "[DESCRIPTION]", "\n".join(page.get("description") or [])) if x)
            return text[:150_000], "google_patents_page"
    text = " ".join(x for x in (doc.get("title") or "", doc.get("abstract") or doc.get("snippet") or "") if x)
    return text, "abstract"


async def recheck(draft: dict, state: dict, event=None) -> dict:
    from app.llm import evaluate_single_document_text
    from patent_analyzer.adjudicate import element_covered
    from patent_analyzer.agentic.elements import attach_facets
    from patent_analyzer.agentic.loop import Budget, _search
    from patent_analyzer.agentic.prune import stage1_embed
    from patent_analyzer.agentic.wide import candidate_queries
    from patent_analyzer.quote_verify import verify_checklist_results
    from patent_analyzer.recall.pool import candidates_to_legacy_docs

    def _ev(kind, msg):
        if event:
            event(kind, msg)

    out = {"queries": [], "new_docs": [], "evaluated": 0, "limitation_coverage": {}, "llm_calls": 0, "gp_calls": 0,
           "serp_calls": 0, "skipped": False}
    targets = new_limitations(draft)
    if not targets:
        out.update(skipped=True, reason="no new limitations to re-check")
        return out

    # pseudo-candidate: the core preamble element (facets already there) + the new limitations
    extraction = state.get("extraction") or {}
    cands = extraction.get("candidate_inventions") or []
    core = next((c for c in cands if c.get("level") == "core"), cands[0] if cands else {}) or {}
    els = [e for e in core.get("elements") or [] if not e.get("unsupported")]
    e0 = {"id": els[0]["id"], "text": els[0]["text"], "facets": dict(els[0].get("facets") or {})} if els else {"id": "e0", "text": "", "facets": {}}
    new_els = [{"id": t["lim"]["lid"], "text": t["lim"]["text"], "facets": {}} for t in targets]
    try:
        await attach_facets(new_els, state.get("summary", ""))
        out["llm_calls"] += 1
    except Exception as e:
        _ev("warn", f"Re-check facets skipped: {type(e).__name__}")
    pseudo = {"id": "draft", "level": "core", "concept": core.get("concept", ""), "cpc_pred": list(core.get("cpc_pred") or []),
              "elements": [e0] + new_els}
    queries = [q for q in candidate_queries(pseudo, cpc_groups=pseudo["cpc_pred"]) if str(q.get("kind", "")).startswith("element:")]
    queries = queries[:MAX_QUERIES]
    if not queries:
        out.update(skipped=True, reason="no query could be formed for the new limitations")
        return out

    serp = {"left": _serp_budget(state)}

    def _take():
        if serp["left"] <= 0:
            return False
        serp["left"] -= 1
        return True

    budget = Budget(lambda: serp["left"], _take)
    before = _cutoff(state)
    known = {(p.get("pub_num") or "").upper() for p in (state.get("search_stats") or {}).get("pool") or []}
    known |= {(d.get("pub_num") or "").upper() for d in state.get("ranked_candidates") or []}
    pool: dict[str, dict] = {}
    for q in queries:
        hits, total, chan = await _search(q["query"], before, budget, num=50)
        new_keys = []
        for c in hits:
            key = (c.pub_num or c.title).upper()
            if key in known or key in pool:
                continue
            pool[key] = candidates_to_legacy_docs([c])[0]
            new_keys.append(c.pub_num or c.title[:80])
        out["queries"].append({"query": q["query"], "elements": q.get("elements", []), "channel": chan, "total": total,
                               "hits": len(hits), "new": len(new_keys), "new_pubs": new_keys[:20]})
    out["gp_calls"], out["serp_calls"] = budget.gp_calls, budget.serp_calls
    docs = list(pool.values())
    if not docs:
        out["reason"] = "no new documents beyond the search pool"
        for t in targets:
            t["lim"]["coverage"]["recheck"] = {"queried": True, "new_docs": 0, "covered_by_new": []}
        return out

    try:
        idx, _ = stage1_embed(new_els, docs, topk=MAX_EVAL, summary="")
        docs = sorted((docs[i] for i in idx), key=lambda d: -float(d.get("prune_cos", 0.0)))[:MAX_EVAL]
    except Exception as e:
        _ev("warn", f"Re-check embedding rank failed ({type(e).__name__}); taking the first {MAX_EVAL}")
        docs = docs[:MAX_EVAL]

    items = [{"id": t["lim"]["lid"], "criterion": t["lim"]["text"], "weight": 1.0 / len(targets)} for t in targets]
    summary = state.get("summary", "")
    coverage: dict[str, list[str]] = {t["lim"]["lid"]: [] for t in targets}
    for d in docs:
        text, mode = await _fetch_text(d)
        row = {"pub_num": d.get("pub_num", ""), "title": (d.get("title") or "")[:120], "url": d.get("url", ""),
               "cos": round(float(d.get("prune_cos", 0.0)), 4), "text_mode": mode, "covered": []}
        if len(text) < 120:
            row["skipped"] = "no text"
            out["new_docs"].append(row)
            continue
        res = await evaluate_single_document_text(summary, items, text, d.get("title") or d.get("pub_num", ""),
                                                  d.get("match_type") or "Patent", doc_mode="full_text")
        out["llm_calls"] += 1
        out["evaluated"] += 1
        cr = res.get("checklist_results") or {}
        stats = verify_checklist_results(cr, text)
        row["verified"] = stats.get("verified", 0)
        for t in targets:
            lid, crit = t["lim"]["lid"], t["lim"]["text"]
            if element_covered(cr.get(crit) or cr.get(lid)):
                coverage[lid].append(d.get("pub_num") or d.get("title", ""))
                row["covered"].append(lid)
        out["new_docs"].append(row)
    out["limitation_coverage"] = coverage
    for t in targets:
        t["lim"]["coverage"]["recheck"] = {"queried": True, "new_docs": len(docs), "covered_by_new": coverage[t["lim"]["lid"]]}

    # the limitation that narrowed the independent claim must survive the re-check
    claims = draft.get("claims") or []
    primary = claims[0] if claims else None
    added = next((t for t in targets if t["claim"] is primary), None)
    if added and coverage.get(added["lim"]["lid"]):
        hit = coverage[added["lim"]["lid"]]
        swap = next((t for t in targets if t["claim"] is not primary and not coverage.get(t["lim"]["lid"])
                     and not t["lim"]["coverage"].get("covered_by")), None)
        if swap:
            old, new = added["lim"], swap["lim"]
            old_text, new_text = old["text"], new["text"]
            for k in ("text", "rule_text", "kind", "origin", "pid", "basis", "coverage", "flags", "distinguishing"):
                old[k], new[k] = new.get(k), old.get(k)
            old["distinguishing"], new["distinguishing"] = True, False
            mirror = next((c for c in claims if c.get("depends_on") is None and c is not primary), None)
            if mirror and mirror.get("limitations"):
                m = mirror["limitations"][-1]
                m["text"], m["rule_text"], m["basis"], m["coverage"], m["origin"], m["pid"] = (new_text, new_text, old["basis"], old["coverage"],
                                                                                              old["origin"], old["pid"])
            out["swapped"] = {"out": old_text, "in": new_text, "disclosed_by_new": hit}
            out["reason"] = (f"the limitation added to claim 1 is disclosed by {', '.join(hit)} (re-check); replaced by the next "
                             f"re-checked limitation no evaluated document discloses")
        else:
            out["strategy"] = "unresolved"
            out["reason"] = (f"the limitation added to claim 1 is disclosed by {', '.join(hit)} found in the re-check and no other "
                             f"re-checked limitation is undisclosed; the independent claim is not narrowed by a verified limitation")
    return out
