"""Phase 4b: draft claims for attorney review (leader_draft.md).

Linear node between the evaluate gate and the report; never loops back into
search. Order: claim chart -> rule assembly of both statutory forms from the
core candidate's supported elements -> one LLM call for wording + pool quotes
-> snap every new quote (unsupported ones are dropped, not drafted) ->
wording invariant (assemble.verify_wording) -> pool limitations re-evaluated
against the charted references' full text (same evaluator + quote verifier as
Phase 4) -> avoidance plan (avoid.plan) -> dependent claims -> rule 112(b)
check / auto_fix / <=2 reword calls -> LLM advisory (DRAFT_ADVISORY) ->
re-check search on the new limitations (DRAFT_RECHECK, draft/recheck.py).

Output: {"draft_claims": {...}, "adjudication": adj (+ claim_chart), events,
phase_results.phase4b}. Every limitation carries basis[] (element id +
verbatim quote + loc). The report wording is blocking risk only.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from state import GraphState

MAX_REWORD_ROUNDS = 2
POOL_WORDING_TAU = 0.75
REFINE_WINDOW = 600
MAX_REFINE_TARGETS = 4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _core(extraction: dict | None) -> dict | None:
    cands = (extraction or {}).get("candidate_inventions") or []
    if not cands:
        return None
    return next((c for c in cands if c.get("level") == "core"), cands[0])


def _doc_full_text(doc: dict | None) -> tuple[str, str]:
    """(text, mode) for a ranked candidate: PDF text, else abstract + claims, else abstract/snippet."""
    if not doc:
        return "", "none"
    from pathlib import Path
    pdf = doc.get("local_pdf") or ""
    if pdf and Path(pdf).exists():
        from patent_analyzer.quote_verify import pdf_text
        t = pdf_text(pdf)
        if t:
            return t, "pdf"
    claims = (doc.get("claims_text") or "").strip()
    abstract = (doc.get("abstract") or "").strip() or (doc.get("snippet") or "").strip()
    if claims:
        return f"[ABSTRACT] {abstract}\n\n[CLAIMS]\n{claims}", "claims"
    return abstract, "abstract"


def _lim_from_pool(p: dict, lid: str, form: str, primary_form: str, disclosed_by: dict, cols: list[str]) -> dict:
    from patent_analyzer.draft.assemble import limitation_text
    kind = p.get("kind") or ("condition" if p.get("text", "").lower().startswith("wherein") else "step")
    text = limitation_text({"text": p.get("text", ""), "kind": kind}, form, primary_form) if form != primary_form else p.get("text", "")
    return {"lid": lid, "text": text, "rule_text": text, "kind": kind, "origin": p.get("origin"), "pid": p.get("pid"),
            "basis": [{"element_id": p.get("element_id", ""), "evidence_quote": p.get("evidence_quote", ""), "evidence_loc": p.get("evidence_loc")}],
            "coverage": {"covered_by": sorted(disclosed_by.get(p.get("pid"), set()) & set(cols)), "checked_against": list(cols),
                         "verified": bool(cols)} if cols else {"covered_by": [], "checked_against": [], "verified": False, "status": "unknown"},
            "flags": []}


def _renumber(claims: list[dict]) -> None:
    for i, c in enumerate(claims, 1):
        c["no"] = i
    for c in claims:
        for j, l in enumerate(c.get("limitations") or [], 1):
            l["lid"] = f"c{c['no']}.l{j}"
    for c in claims:
        if c.get("depends_on") is not None:
            noun = "method" if c.get("form") == "method" else "system"
            bridge = c.get("preamble", "").rsplit(",", 1)[-1].strip() or "wherein"
            c["preamble"] = f"The {noun} of claim {c['depends_on']}, {bridge}"


async def draft_node(state: GraphState) -> dict:
    from evals.extraction_errors import snap_quote
    from graph.extraction_subgraph import raw_file_text
    from patent_analyzer.adjudicate import claim_chart, element_covered
    from patent_analyzer.draft import assemble as A
    from patent_analyzer.draft import avoid as V
    from patent_analyzer.draft import definiteness as D

    events: list[dict] = []

    def _event(kind: str, message: str, payload: dict | None = None):
        evt = {"ts": _now(), "phase": "phase4b", "kind": kind, "message": message}
        if payload:
            evt["payload"] = payload
        events.append(evt)

    _event("start", "Drafting claims from the grounded elements and the evidence matrix")
    extraction = state.get("extraction") or {}
    core = _core(extraction)
    supported = [e for e in (core or {}).get("elements") or [] if not e.get("unsupported") and e.get("text")]
    if not core or len(supported) < 2:
        _event("info", "No grounded elements to draft from")
        return {"draft_claims": {"candidate_id": (core or {}).get("id"), "strategy": "no_elements", "claims": [], "llm_calls": 0},
                "events": events, "phase_results": {"phase4b": {"status": "completed", "data": {"strategy": "no_elements"}}}}

    checklist = state.get("checklist") or []
    scoring_report = state.get("scoring_report") or []
    ranked = {(d.get("pub_num") or d.get("title") or ""): d for d in state.get("ranked_candidates") or []}
    summary = state.get("summary", "")
    document_text = state.get("document_text") or ""
    fallback_text = raw_file_text(state) if state.get("doc_json") else ""
    llm_calls = 0

    # ── claim chart (same evidence the determination used) ──
    adj = dict(state.get("adjudication") or (state.get("eval_stats") or {}).get("adjudication") or {})
    has_prior_art = bool(scoring_report) and bool(adj.get("n_elements"))
    chart = adj.get("claim_chart") if has_prior_art else None
    if has_prior_art and not chart:
        chart = claim_chart(adj, checklist, scoring_report)
        adj["claim_chart"] = chart
    cols = [d.get("key") or d.get("pub_num") or d.get("title") or "" for d in (chart or {}).get("docs") or []]
    covered = V.covered_sets(chart, checklist)

    # ── rule assembly ──
    primary_form = core.get("primary_form") or "method"
    mirror_form = "system" if primary_form == "method" else "method"
    primary = A.assemble_independent(core, primary_form, claim_no=1)
    mirror = A.assemble_independent(core, mirror_form, claim_no=None)
    for i, l in enumerate(mirror["limitations"], 1):
        l["lid"] = f"m.l{i}"
    for claim in (primary, mirror):
        for l in claim["limitations"]:
            l["rule_text"] = l["text"]
            eid = l["basis"][0]["element_id"]
            l["coverage"] = ({"covered_by": sorted(k for k, ids in covered.items() if eid in ids), "checked_against": list(cols), "verified": True}
                             if cols else {"covered_by": [], "checked_against": [], "verified": False, "status": "unknown"})
    pool = V.candidate_limitations(extraction, core)

    # refinement targets: condition / parameter elements with a located span -> the passage around it
    refine_targets = []
    for e in supported[1:]:
        loc = e.get("evidence_loc") or {}
        span = loc.get("char") if isinstance(loc, dict) else None
        if e.get("kind") in ("condition", "parameter", "step") and span and len(refine_targets) < MAX_REFINE_TARGETS:
            src = document_text if loc.get("source") != "fallback_text" else (fallback_text or document_text)
            refine_targets.append({"element_id": e["id"], "text": e["text"],
                                   "passage": src[max(0, span[0] - REFINE_WINDOW): span[1] + REFINE_WINDOW]})
    coverage_lines = [f"{k} discloses: {', '.join(sorted(ids)) or 'nothing verified'}" for k, ids in covered.items()]
    if chart:
        coverage_lines.append(f"not disclosed by any charted reference: {', '.join(V.uncovered_elements(chart, checklist)) or 'none'}")

    # ── LLM 1: wording + pool quotes + refinements ──
    from app.llm import draft_claims as llm_draft
    _event("llm", "Rewording the rule-assembled claims and copying quotes for the candidate limitations (one call)")
    llm_out = await llm_draft(primary_form, [{"lid": l["lid"], "text": l["text"]} for l in primary["limitations"]],
                              [{"lid": l["lid"], "text": l["text"]} for l in mirror["limitations"]],
                              [{"pid": p["pid"], "text": p["text"], "evidence_quote": p.get("evidence_quote", "")} for p in pool],
                              refine_targets, coverage_lines, document_text)
    llm_calls += 1
    if llm_out.get("error"):
        _event("warn", llm_out["error"])
    for p in pool:
        got = (llm_out.get("pool") or {}).get(p["pid"]) or {}
        if got.get("text"):
            p["draft_text"] = got["text"]
        if not p.get("evidence_quote") and got.get("evidence_quote"):
            p["evidence_quote"] = got["evidence_quote"]
    for i, r in enumerate(llm_out.get("refinements") or []):
        pool.append({"pid": f"ref{i}", "origin": "refinement", "source": f"refine({r.get('element_id') or '?'})", "text": r["text"],
                     "element_id": f"refine({r.get('element_id') or '?'})", "evidence_quote": r.get("evidence_quote", ""),
                     "evidence_loc": None, "kind": ""})

    # ── every new quote must snap to the document (37 CFR 1.75(d)(1)); else the limitation is dropped ──
    n_dropped = 0
    for p in pool:
        if p["origin"] not in ("dependent_hint", "refinement"):
            continue
        q = p.get("evidence_quote") or ""
        found, loc = snap_quote(q, document_text) if q else (False, None)
        source = "doc_json" if state.get("doc_json") else "text"
        if not found and q and fallback_text:
            found, loc = snap_quote(q, fallback_text)
            source = "fallback_text"
        if found:
            p["evidence_loc"] = {"char": (loc or {}).get("char"), "method": (loc or {}).get("method"), "sim": (loc or {}).get("sim"), "source": source}
        else:
            p["dropped"] = "unsupported" if q else "no_quote"
            n_dropped += 1
    # pool wording: LLM's claim-language text must stay close to the hint / element text it came from
    similarity = A.embed_similarity
    sim_method = "te005"
    try:
        similarity(["a"], ["a"])
    except Exception as e:
        similarity, sim_method = A.difflib_similarity, f"difflib ({type(e).__name__})"
        _event("warn", f"Embedding unavailable for the wording invariant; using difflib ratio ({type(e).__name__})")
    for p in pool:
        if p.get("draft_text") and not p.get("dropped"):
            # a hint is plain language; its claim wording may drift further than an element's (quote still anchors it)
            texts, rep = A.verify_wording([p["text"]], [p["draft_text"]], similarity, tau=POOL_WORDING_TAU)
            p["text"] = texts[0]
            p["wording"] = rep["per_limitation"][0]
    usable = [p for p in pool if not p.get("dropped")]

    # ── wording invariant on the independent claims ──
    for claim, key in ((primary, "primary"), (mirror, "mirror")):
        rule = [l["text"] for l in claim["limitations"]]
        got = llm_out.get(key) or {}
        llm_texts = [got.get(l["lid"]) for l in claim["limitations"]] if got else None
        texts, rep = A.verify_wording(rule, [t or "" for t in llm_texts] if llm_texts else None, similarity)
        A.apply_wording(claim, texts, rep)
        claim["wording_check"] = {**{k: v for k, v in rep.items() if k != "per_limitation"}, "method": sim_method}
    _event("draft_written", f"Independent claims assembled: {len(primary['limitations'])} limitations ({primary_form} + {mirror_form} mirror); "
                            f"wording accepted {primary['wording_check']['accepted']}/{primary['wording_check']['n']} (primary), "
                            f"{mirror['wording_check']['accepted']}/{mirror['wording_check']['n']} (mirror); pool {len(usable)} usable, {n_dropped} dropped")

    # ── pool limitations vs the charted references' full text (Phase 4 evaluator + quote verifier) ──
    disclosed_by: dict[str, set[str]] = {p["pid"]: set() for p in usable}
    pool_eval = []
    if chart and usable:
        from app.llm import evaluate_single_document_text
        from patent_analyzer.quote_verify import verify_checklist_results
        items = [{"id": p["pid"], "criterion": p["text"], "weight": 1.0 / len(usable)} for p in usable]
        for d in chart["docs"]:
            key = d.get("key") or d.get("pub_num") or d.get("title") or ""
            src = ranked.get(key) or ranked.get(d.get("pub_num") or "") or {}
            text, mode = _doc_full_text(src)
            if len(text) < 120:
                pool_eval.append({"doc": key, "mode": mode, "skipped": "no text"})
                continue
            _event("llm", f"Checking {len(items)} candidate limitations against {key} ({mode})")
            res = await evaluate_single_document_text(summary, items, text[:150_000], d.get("title") or key,
                                                      src.get("match_type") or "Patent", doc_mode="full_text")
            llm_calls += 1
            cr = res.get("checklist_results") or {}
            stats = verify_checklist_results(cr, text)
            row = {"doc": key, "mode": mode, "verified": stats.get("verified", 0), "quotes": stats.get("quotes", 0), "covered": []}
            for p in usable:
                if element_covered(cr.get(p["text"]) or cr.get(p["pid"])):
                    disclosed_by[p["pid"]].add(key)
                    row["covered"].append(p["pid"])
            pool_eval.append(row)

    # ── avoidance plan ──
    pl = V.plan(adj, chart, checklist, pool, disclosed_by, has_prior_art=has_prior_art)
    _event("draft_plan", f"Strategy {pl['strategy']}: {pl['reason']}", {"label": pl.get("label"), "basis": pl.get("basis")})
    by_pid = {p["pid"]: p for p in pool}
    uncovered = set(pl.get("uncovered_elements") or [])
    for claim in (primary, mirror):
        for l in claim["limitations"]:
            if l["basis"][0]["element_id"] in uncovered:
                l["distinguishing"] = True
    if pl.get("independent_add"):
        p = by_pid[pl["independent_add"]]
        primary["limitations"].append(_lim_from_pool(p, f"c1.l{len(primary['limitations']) + 1}", primary_form, primary_form, disclosed_by, cols))
        mirror["limitations"].append(_lim_from_pool(p, f"m.l{len(mirror['limitations']) + 1}", mirror_form, primary_form, disclosed_by, cols))
        for claim in (primary, mirror):
            claim["limitations"][-1]["distinguishing"] = True

    # ── dependent claims (further-limitation check) ──
    dep_items = [by_pid[pid] for pid in pl.get("dependents") or [] if pid in by_pid and not by_pid[pid].get("dropped")]
    kept, rejected = V.dependent_claims(dep_items, primary, similarity)
    claims = [primary]
    for p in kept:
        claims.append(A.dependent_claim(primary, _lim_from_pool(p, "x", primary_form, primary_form, disclosed_by, cols)))
    mirror_no = len(claims) + 1
    mirror["no"] = mirror_no
    claims.append(mirror)
    for p in kept:
        claims.append(A.dependent_claim(mirror, _lim_from_pool(p, "x", mirror_form, primary_form, disclosed_by, cols)))
    _renumber(claims)

    # ── 112(b): rules -> auto_fix -> reword (<= 2 calls) -> rules ──
    from app.llm import reword_limitations
    all_flags: list[dict] = []
    rounds = 0
    open_flags: list[dict] = []
    for rnd in range(MAX_REWORD_ROUNDS + 1):
        rounds = rnd + 1
        by_no = {c["no"]: c for c in claims}
        all_flags = []
        for c in claims:
            parents = []
            d = c.get("depends_on")
            while d is not None and d in by_no:
                parents.insert(0, by_no[d])
                d = by_no[d].get("depends_on")
            c, flags = D.auto_fix(c, D.check(c, parents))
            for f in flags:
                if f.get("lid") == "preamble":
                    f["lid"] = f"c{c['no']}.pre"
                f["round"] = rounds
            all_flags += flags
        open_flags = [f for f in all_flags if not f.get("fixed") and not f["lid"].endswith(".pre")]
        for c in claims:
            for l in c["limitations"]:
                l["flags"] = [f for f in all_flags if f["lid"] == l["lid"]]
        if not open_flags or rnd == MAX_REWORD_ROUNDS:
            break
        lims = {l["lid"]: l for c in claims for l in c["limitations"]}
        flagged = []
        for lid in sorted({f["lid"] for f in open_flags}):
            l = lims[lid]
            flagged.append({"lid": lid, "text": l["text"], "flags": [{k: f.get(k) for k in ("category", "span", "note")} for f in open_flags if f["lid"] == lid],
                            "quotes": [b.get("evidence_quote", "") for b in l.get("basis") or []]})
        _event("llm", f"Rewording {len(flagged)} flagged limitation(s) under 112(b) (round {rnd + 1})")
        new = await reword_limitations(flagged)
        llm_calls += 1
        if not new:
            break
        for lid, text in new.items():
            l = lims.get(lid)
            if not l:
                continue
            texts, rep = A.verify_wording([l.get("rule_text") or l["text"]], [text], similarity)
            if rep["per_limitation"][0]["accepted"]:
                l["text"] = texts[0]
                l["reworded"] = True
            else:
                l["reword_rejected"] = rep["per_limitation"][0].get("reason")
    n_fixed = sum(1 for f in all_flags if f.get("fixed"))
    _event("draft_flags", f"112(b) rules: {len(all_flags)} flag(s), {n_fixed} auto-fixed, {len(open_flags)} open after {rounds} pass(es)",
           {"open": [{"lid": f["lid"], "category": f["category"], "span": f["span"]} for f in open_flags]})

    # ── LLM advisory (categories the rules cannot check) ──
    advisory: dict = {}
    if os.environ.get("DRAFT_ADVISORY", "1") != "0":
        from app.llm import definiteness_advisory
        _event("llm", "Definiteness advisory over the claim set (PEDANTIC examination prompt, one call)")
        advisory = await definiteness_advisory([{"no": c["no"], "text": A.render_claim(c), "depends_on": c.get("depends_on")} for c in claims],
                                               document_text[:60000])
        llm_calls += 1
        if advisory.get("_error"):
            _event("warn", f"Advisory skipped: {advisory['_error']}")

    draft = {
        "candidate_id": core.get("id"), "primary_form": primary_form, "strategy": pl["strategy"],
        "basis_adjudication": {"label": adj.get("label"), "basis": adj.get("basis"), "best_single": adj.get("best_single"),
                               "combo_docs": list((adj.get("combo") or {}).get("docs") or []), "chart_docs": cols},
        "claims": claims,
        "avoidance": {"covered_set": pl.get("covered_set"), "uncovered_elements": pl.get("uncovered_elements"),
                      "candidates_tried": pl.get("candidates_tried"), "independent_add": pl.get("independent_add"),
                      "first_dependent": pl.get("first_dependent"), "reason": pl["reason"], "pool_eval": pool_eval,
                      "rejected_dependents": [{k: r.get(k) for k in ("pid", "source", "text", "rejected", "note", "sim")} for r in rejected],
                      "pool": [{k: p.get(k) for k in ("pid", "origin", "source", "text", "evidence_quote", "evidence_loc", "dropped", "wording")} for p in pool]},
        "definiteness": {"passes": rounds, "flags": all_flags, "open_flags": open_flags,
                         "llm_advisory": {str(k): v for k, v in advisory.items() if not str(k).startswith("_")}},
        "wording_check": {"primary": primary.get("wording_check"), "mirror": mirror.get("wording_check")},
        "recheck": {"skipped": True},
        "llm_calls": llm_calls, "search_calls": {"gp": 0, "serpapi": 0},
    }

    # ── re-check search on the new limitations (never loops the main graph) ──
    if os.environ.get("DRAFT_RECHECK", "1") != "0":
        try:
            from patent_analyzer.draft.recheck import recheck
            _event("draft_recheck", "Re-checking the new limitations against a small search")
            rc = await recheck(draft, state, _event)
            draft["recheck"] = rc
            draft["llm_calls"] += rc.get("llm_calls", 0)
            draft["search_calls"] = {"gp": rc.get("gp_calls", 0), "serpapi": rc.get("serp_calls", 0)}
            if rc.get("strategy"):
                draft["strategy"] = rc["strategy"]
            if rc.get("reason"):
                draft["avoidance"]["recheck_reason"] = rc["reason"]
            _event("draft_recheck", f"Re-check: {len(rc.get('queries') or [])} queries, {len(rc.get('new_docs') or [])} new docs, "
                                    f"{rc.get('evaluated', 0)} evaluated; strategy {draft['strategy']}")
        except Exception as e:
            draft["recheck"] = {"skipped": True, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            _event("warn", f"Re-check skipped: {type(e).__name__}: {str(e)[:160]}")

    _event("done", f"Draft: {len(claims)} claims ({primary_form} 1-{mirror_no - 1}, {mirror_form} {mirror_no}-{len(claims)}), strategy {draft['strategy']}, "
                   f"{draft['llm_calls']} LLM calls")
    return {"draft_claims": draft, "adjudication": adj, "events": events,
            "phase_results": {"phase4b": {"status": "completed", "data": {"claims": len(claims), "strategy": draft["strategy"],
                                                                            "open_flags": len(open_flags), "llm_calls": draft["llm_calls"]}}}}
