"""ReAct-style search loop: the model sits at the search box.

Each step the model sees what the previous query returned (total, the top
titles with years) and the elements still without candidates, and decides
the next query as (specific items) AND (neighbourhood terms) [+ CPC group].
The query string itself is composed by code so the syntax stays valid; the
model only picks the words — including patent vocabulary it learns from
the returned titles.

Sources (2026-09-18):
- ReAct, Yao et al. 2022 (arXiv 2210.03629): "generate both reasoning traces
  and task-specific actions in an interleaved manner ... reasoning traces
  help the model induce, track, and update action plans as well as handle
  exceptions, while actions allow it to interface with external sources".
- patent-search-pilot agent.py (per-element searches, results promoted by
  how many element searches reach them).
- btrettel (examiner, HN 2022-11-07): "Many examiners, myself included, keep
  lists of search queries with a lot of synonyms to use later ... my saved
  search queries keep growing" — the loop keeps a growing synonym list.
- r/patentexaminer: "Use natural language to describe the problem being
  solved, concisely"; Google help 7049475: unquoted words are stemmed.
"""

from __future__ import annotations

import json
import os

from .query_gen import _group, cpc_clause

MAX_STEPS = int(os.environ.get("REACT_MAX_STEPS", "10"))
REACT_THINKING = int(os.environ.get("REACT_THINKING", "8192"))
# J6 guardrail: the flash models narrow a query until its field is too small to hold the
# answer (h1n totals 9.6k-38k against 2.5-pro's 96k-126k, pool reach 1/5 vs 4/5). When a step
# comes back under REACT_MIN_TOTAL, drop the last specific item and search once more.
REACT_MIN_TOTAL = int(os.environ.get("REACT_MIN_TOTAL", "0"))
TITLES_SHOWN = 25

STEP_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "observation": {"type": "STRING"},
        "decision": {"type": "STRING"},
        "covered_elements": {"type": "ARRAY", "items": {"type": "STRING"}},
        "learned_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
        "next": {"type": "OBJECT", "properties": {
            "target_elements": {"type": "ARRAY", "items": {"type": "STRING"}},
            "specific": {"type": "ARRAY", "items": {"type": "STRING"}},
            "broad": {"type": "ARRAY", "items": {"type": "STRING"}},
            "cpc_group": {"type": "STRING"}}},
        "stop": {"type": "BOOLEAN"},
    },
    "required": ["observation", "decision", "next", "stop"],
}


def compose(specific: list[str], broad: list[str], cpc_group: str | None) -> str:
    """(specific OR …) (broad OR …) [CPC=<group>/low] — clause last."""
    sp = _group([t for t in specific if t], cap=4)
    br = _group([t for t in broad if t], cap=15)
    q = " ".join(x for x in (sp, br) if x)
    c = cpc_clause(cpc_group or "")
    return f"{q} {c}".strip() if c else q


def _history_block(steps: list[dict]) -> str:
    if not steps:
        return "(no query yet)"
    out = []
    for s in steps:
        titles = "; ".join(f"{t['title'][:70]} ({t.get('year') or '?'})" for t in s.get("top", [])[:TITLES_SHOWN])
        out.append(f"step {s['n']}: target {s.get('target_elements')} | specific {s.get('specific')} | broad {len(s.get('broad') or [])} terms"
                   f" | cpc {s.get('cpc_group') or '-'}\n  total={s.get('total')} returned={s.get('returned')} new={s.get('new')}\n  titles: {titles or '(none)'}")
    return "\n".join(out)


def step_prompt(elements: list[dict], broad_terms: list[str], cpc_groups: list[str], steps: list[dict],
                uncovered: list[str], learned: list[str], budget_left: int) -> str:
    from app import prompts
    els = "\n".join(f"  {e['id']}: {e['text'][:160]} | items: {', '.join((e.get('facets') or {}).get('thing', [])[:4])}"
                    + (f" | names: {', '.join((e.get('facets') or {}).get('named', [])[:2])}" if (e.get('facets') or {}).get('named') else "")
                    for e in elements)
    return prompts.render("search.react_step", elements=els, broad=", ".join(broad_terms[:15]), cpc_groups=", ".join(cpc_groups[:8]) or "none",
                          history=_history_block(steps), uncovered=", ".join(uncovered) or "none", learned=", ".join(learned) or "none",
                          budget_left=budget_left)


async def run_react(elements: list[dict], broad_terms: list[str], cpc_groups: list[str], search, budget_left,
                    event=None, call=None, max_steps: int = MAX_STEPS) -> list[dict]:
    """search(query) -> (hits: list[Candidate], total, channel). Returns the
    step log (query rows in the funnel format + observation/decision)."""
    if call is None:
        from app.llm import call_llm as call
    from app.llm import stage_model
    system = "You are a patent examiner running a prior-art search, one query at a time. Output JSON only."
    steps: list[dict] = []
    uncovered = [e["id"] for e in elements[1:]] or [e["id"] for e in elements]
    learned: list[str] = []
    seen_pubs: set[str] = set()
    for n in range(1, max_steps + 1):
        if budget_left() <= 0:
            break
        prompt = step_prompt(elements, broad_terms + learned, cpc_groups, steps, uncovered, learned, budget_left())
        try:
            # ten calls a job: thinking is cheap here and it is where the query is decided
            resp = await call(system, prompt, response_schema=STEP_SCHEMA, thinking_budget=REACT_THINKING, model=stage_model("search"))
            d = json.loads(resp)
        except Exception as exc:
            steps.append({"n": n, "error": f"{type(exc).__name__}: {exc}"[:160], "query": "", "kind": "react", "total": None,
                          "returned": 0, "hits": 0, "new": 0, "pubs": [], "new_pubs": [], "channel": "none", "top": []})
            break
        for t in d.get("learned_terms") or []:
            t = " ".join(str(t).lower().split())
            if t and t not in learned and t not in broad_terms:
                learned.append(t)
        for eid in d.get("covered_elements") or []:
            if eid in uncovered:
                uncovered.remove(eid)
        if d.get("stop") and steps and (budget_left() <= 1 or not (d.get("next") or {}).get("specific")):
            steps.append({"n": n, "observation": d.get("observation"), "decision": d.get("decision"), "query": "", "kind": "react:stop",
                          "total": None, "returned": 0, "hits": 0, "new": 0, "pubs": [], "new_pubs": [], "channel": "none", "top": []})
            break
        nx = d.get("next") or {}
        specific = [" ".join(str(t).lower().split()) for t in (nx.get("specific") or []) if str(t).strip()][:4]
        broad = [" ".join(str(t).lower().split()) for t in (nx.get("broad") or []) if str(t).strip()][:15] or broad_terms[:12]
        cpc = str(nx.get("cpc_group") or "").split("/")[0] or None
        # MPEP 904.01(c): "all analogous arts must be searched regardless of where the claimed
        # invention is classified" — every predicted group is tried once before any is reused.
        # h1i H1-02: the facet call predicted A61B6 / G01S5 / G05D1 (three of the six gold
        # families' groups) but the model kept to A61N5 / G06T7 / G16H40 for all nine steps.
        used = {s.get("cpc_group") for s in steps if s.get("cpc_group")}
        unused = [g for g in cpc_groups if g not in used]
        forced = None
        if unused and (cpc in used or not cpc) and budget_left() <= len(unused):
            forced, cpc = unused[0], unused[0]
        query = compose(specific, broad, cpc)
        if not query or any(s.get("query") == query for s in steps):
            # a repeated or empty query: spend the step on the broad group alone with a different CPC
            query = compose(specific, broad, None) if cpc else compose(specific[:2], broad, cpc_groups[len(steps) % max(1, len(cpc_groups))] if cpc_groups else None)
            if not query or any(s.get("query") == query for s in steps):
                break
        hits, total, chan = await search(query)
        widened = None
        if (REACT_MIN_TOTAL and total is not None and total < REACT_MIN_TOTAL
                and len(specific) > 1 and budget_left() >= 1):
            wider = compose(specific[:-1], broad, cpc)
            if wider and not any(st.get("query") == wider for st in steps):
                w_hits, w_total, w_chan = await search(wider)
                if (w_total or 0) > (total or 0):
                    widened = {"from": query, "from_total": total, "dropped": specific[-1]}
                    query, hits, total, chan, specific = wider, w_hits, w_total, w_chan, specific[:-1]
        pubs = [(c.pub_num or c.title[:80]) for c in hits]
        new = [p for p in pubs if p not in seen_pubs]
        seen_pubs.update(pubs)
        row = {"n": n, "kind": "react", "query": query, "target_elements": nx.get("target_elements") or [], "specific": specific,
               "broad": broad, "cpc_group": cpc, "cpc_forced": forced, "widened": widened,
               "observation": d.get("observation"), "decision": d.get("decision"),
               "channel": chan, "total": total, "hits": len(hits), "returned": len(hits), "new": len(new), "pubs": pubs, "new_pubs": new,
               "papers": sum(1 for c in hits if c.match_type != "Patent"),
               "facets_used": {"specific": specific, "broad": broad, **({"cpc": [cpc]} if cpc else {})},
               "elements": nx.get("target_elements") or [],
               "top": [{"title": c.title, "year": c.year, "pub": c.pub_num} for c in hits[:TITLES_SHOWN]],
               "_hits": hits}
        steps.append(row)
        if event:
            event("react_step", f"step {n}: {d.get('decision', '')[:120]} → total {total}, new {len(new)}",
                  {k: v for k, v in row.items() if k not in ("_hits", "pubs", "new_pubs", "top")})
        # every element covered is not the end: the budget is spent on element pairs in the
        # learned vocabulary (h1i H1-01 stopped at 6/10 with 2 of 5 gold found)
    return steps
