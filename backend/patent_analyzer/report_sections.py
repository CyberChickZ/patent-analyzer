"""New report sections: candidate inventions with verbatim evidence, the
search loop's per-round statistics, and the element × document quote
matrix. Rendered as HTML (matching report_generator's .sec/.sec-t/.sec-b
classes) and Markdown, injected after "Invention Summary".

Wording rule: the report says whether prior art poses a blocking risk;
it never says an application would be granted (applications stay
unpublished for 18 months, so the record is always incomplete).
"""

from __future__ import annotations

import html


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


# ── 1. candidate inventions / elements / evidence ──

def extraction_html(extraction: dict | None) -> str:
    cands = (extraction or {}).get("candidate_inventions") or []
    if not cands:
        if (extraction or {}).get("no_invention_reason"):
            return (f'<div class="sec"><div class="sec-t">Candidate Inventions</div>'
                    f'<div class="sec-b">No patentable invention identified: {_e(extraction["no_invention_reason"])}</div></div>')
        return ""
    out = ['<div class="sec"><div class="sec-t">Candidate Inventions</div>',
           '<div class="sec-note">Each element is stated in claim language; the quote is copied verbatim from your document '
           'and located by the system — elements whose quote could not be found are marked unsupported.</div>']
    for c in cands:
        n_el = len(c.get("elements") or [])
        n_ok = sum(1 for e in c.get("elements") or [] if not e.get("unsupported"))
        out.append(f'<details {"open" if c is cands[0] else ""} style="margin:0.6rem 0">'
                   f'<summary><b>{_e(c.get("id", ""))}</b> · <span class="badge">{_e(c.get("level", ""))}</span> '
                   f'{_e(c.get("concept", ""))} <span style="color:var(--text2)">({n_ok}/{n_el} elements grounded'
                   f'{", CPC " + _e(", ".join(c.get("cpc_pred") or [])) if c.get("cpc_pred") else ""})</span></summary>')
        draft = c.get("independent_claim_draft") or {}
        for kind in ("method", "system"):
            if draft.get(kind):
                out.append(f'<div class="sec-b" style="margin:0.4rem 0"><b>Draft {kind} claim:</b> {_e(draft[kind])}</div>')
        out.append('<table class="tbl"><thead><tr><th>#</th><th>Element (claim language)</th><th>Evidence (verbatim)</th><th>Where</th></tr></thead><tbody>')
        for e in c.get("elements") or []:
            loc = e.get("evidence_loc") or {}
            where = ", ".join(f"{k} {v}" for k, v in loc.items() if v not in (None, "", [])) if isinstance(loc, dict) else _e(loc)
            style = ' style="opacity:.55"' if e.get("unsupported") else ""
            flag = ' <span class="badge" style="background:#fee2e2;color:#991b1b">unsupported</span>' if e.get("unsupported") else ""
            if e.get("edited_by_user"):
                flag += ' <span class="badge" style="background:#fef3c7;color:#92400e">edited by reviewer</span>'
            out.append(f'<tr{style}><td>{_e(e.get("id", ""))}</td><td>{_e(e.get("text", ""))}{flag}</td>'
                       f'<td><q>{_e(e.get("evidence_quote", ""))}</q></td><td>{_e(where)}</td></tr>')
        out.append("</tbody></table></details>")
    out.append("</div>")
    return "\n".join(out)


def extraction_md(extraction: dict | None) -> list[str]:
    cands = (extraction or {}).get("candidate_inventions") or []
    if not cands:
        return []
    lines = ["## Candidate Inventions", ""]
    for c in cands:
        lines.append(f"### {c.get('id', '')} · {c.get('level', '')} — {c.get('concept', '')}")
        draft = c.get("independent_claim_draft") or {}
        for kind in ("method", "system"):
            if draft.get(kind):
                lines.append(f"- **Draft {kind} claim:** {draft[kind]}")
        lines += ["", "| # | Element | Evidence (verbatim) | Where |", "|---|---|---|---|"]
        for e in c.get("elements") or []:
            loc = e.get("evidence_loc") or {}
            where = ", ".join(f"{k} {v}" for k, v in loc.items() if v not in (None, "", [])) if isinstance(loc, dict) else str(loc)
            tag = (" _(unsupported)_" if e.get("unsupported") else "") + (" _(edited by reviewer)_" if e.get("edited_by_user") else "")
            lines.append(f"| {e.get('id', '')} | {e.get('text', '')}{tag} | {e.get('evidence_quote', '')} | {where} |")
        lines.append("")
    return lines


# ── 2. search loop rounds ──

def loop_html(search_stats: dict | None) -> str:
    rounds = (search_stats or {}).get("loop_rounds") or []
    els = (search_stats or {}).get("loop_elements") or []
    if not rounds:
        return ""
    n = len(els)
    out = ['<div class="sec"><div class="sec-t">Search Rounds</div>',
           f'<div class="sec-note">{n} elements searched; each round adds candidates and re-checks which elements have a '
           'plausible match. "Covered" here is a cheap lexical/semantic proxy used to decide what to search next — '
           'the evidence-grade judgement is in the evaluation below.</div>',
           '<table class="tbl"><thead><tr><th>Round</th><th>Queries</th><th>Google</th><th>SerpAPI</th><th>Seeds</th>'
           '<th>Citation expansion</th><th>Pool</th><th>Covered</th><th>CPC hint</th></tr></thead><tbody>']
    for r in rounds:
        out.append(f'<tr><td>{r.get("round")}</td><td>{r.get("n_queries")}</td>'
                   f'<td>{r.get("gp_calls")}{" (blocked)" if r.get("gp_blocked") else ""}</td><td>{r.get("serpapi_calls")}</td>'
                   f'<td>{r.get("seeds")}</td><td>+{r.get("expanded")}</td><td>{r.get("pool_size")}</td>'
                   f'<td>{len(r.get("covered") or [])}/{n}</td><td>{_e(r.get("cpc_hint") or "")}</td></tr>')
    out.append("</tbody></table>")
    out.append(queries_html(search_stats))
    last = rounds[-1]
    unc = last.get("uncovered") or []
    if unc:
        names = {e["id"]: e["text"] for e in els}
        out.append('<div class="sec-b" style="margin-top:.5rem"><b>Not yet matched by any candidate:</b><ul>'
                   + "".join(f"<li>{_e(names.get(u, u))}</li>" for u in unc[:12]) + "</ul></div>")
    q = (search_stats or {}).get("serpapi_quota") or []
    if q:
        out.append('<div class="sec-note">SerpAPI keys this month: ' +
                   ", ".join(f'{_e(k["key"])} {k["used"]}/{k["cap"]}' for k in q) + "</div>")
    out.append("</div>")
    return "\n".join(out)


_KIND_NOTE = {
    "named": "the element's distinctive names only (chemical / organism / product names, acronym OR expansion)",
    "named+thing": "distinctive names AND the 'what it is' forms",
    "thing+place": "'what it is' forms AND 'where it operates' forms",
    "thing": "'what it is' forms only",
    "strict": "names ∧ thing ∧ place ∧ apparatus", "loose": "thing ∧ place", "core": "thing only",
}


def _query_explainer(q: dict) -> str:
    fu = q.get("facets_used") or {}
    parts = []
    for k in ("named", "thing", "place", "apparatus"):
        if fu.get(k):
            parts.append(f"{k}: " + ", ".join(fu[k][:8]))
    src = "; ".join(parts)
    els = ", ".join(q.get("elements") or [])
    note = _KIND_NOTE.get(q.get("kind") or q.get("mode") or "", "")
    return " · ".join(x for x in (note, f"from elements {els}" if els else "", src) if x)


def queries_html(search_stats: dict | None) -> str:
    """Every query the loop issued: how it was built, what came back, what
    survived to the final list. Google syntax: space = AND, OR, parentheses,
    \"phrase\" = exact, unquoted words are stemmed; a form in its own
    parentheses only has to co-occur in the document."""
    rounds = (search_stats or {}).get("loop_rounds") or []
    qs = [q for r in rounds for q in (r.get("queries") or [])]
    if not qs:
        return ""
    fd = {d.get("pub_num"): d for d in (search_stats or {}).get("funnel_docs") or [] if d.get("pub_num")}
    pruned = {p.get("pub_num") for p in (search_stats or {}).get("pruned") or []}
    out = ['<div class="sec-b" style="margin-top:.6rem"><b>Every query</b> (space = AND, OR, "phrase" = exact; unquoted words are stemmed by Google; '
           'a multi-word form in parentheses only has to co-occur in the document):</div>',
           '<div style="overflow-x:auto"><table class="tbl"><thead><tr><th>#</th><th>Kind</th><th>Query</th><th>Built from</th>'
           '<th>Total on Google</th><th>Taken</th><th>New</th><th>Kept after screen</th><th>In final top-30</th><th>Elements its documents touch</th></tr></thead><tbody>']
    for i, q in enumerate(qs, 1):
        pubs = q.get("pubs") or []
        kept = sum(1 for p in pubs if p in pruned)
        top = sum(1 for p in pubs if (fd.get(p) or {}).get("rank"))
        touched = sorted({e for p in pubs for e in ((fd.get(p) or {}).get("elements") or [])})
        out.append(f'<tr><td>{q.get("n") or i}</td><td>{_e(q.get("kind") or q.get("mode") or "")}</td>'
                   f'<td><code>{_e((q.get("query") or "")[:220])}</code></td><td>{_e(_query_explainer(q))}</td>'
                   f'<td>{q.get("total") if q.get("total") is not None else "?"}</td><td>{q.get("hits")}</td>'
                   f'<td>{q.get("new") if q.get("new") is not None else "–"}</td><td>{kept if pubs else "–"}</td><td>{top if pubs else "–"}</td>'
                   f'<td>{_e(", ".join(touched)) if touched else "–"}</td></tr>')
    out.append("</tbody></table></div>")
    r0 = rounds[0]
    if r0.get("cited_total") is not None:
        out.append(f'<div class="sec-note">Citation expansion: {r0.get("seeds")} seed patents → {r0.get("cited_total")} distinct cited documents, '
                   f'{r0.get("expanded")} taken (most-cited first; priority date before the cutoff).</div>')
    pr = (search_stats or {}).get("prune") or {}
    if pr:
        out.append(f'<div class="sec-note">Screen: pool {pr.get("pool")} → embedding shortlist {pr.get("stage1_out")} '
                   f'(top-100 per element) → LLM read {pr.get("stage2_in")} abstracts in {pr.get("stage2_calls")} calls, '
                   f'{pr.get("stage2_worth")} judged worth reading, {pr.get("stage2_out")} kept for full evaluation.</div>')
    return "\n".join(out)


def loop_md(search_stats: dict | None) -> list[str]:
    rounds = (search_stats or {}).get("loop_rounds") or []
    n = len((search_stats or {}).get("loop_elements") or [])
    if not rounds:
        return []
    lines = ["## Search Rounds", "", "| Round | Queries | Google | SerpAPI | Seeds | Citation expansion | Pool | Covered |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rounds:
        lines.append(f"| {r.get('round')} | {r.get('n_queries')} | {r.get('gp_calls')} | {r.get('serpapi_calls')} | "
                     f"{r.get('seeds')} | +{r.get('expanded')} | {r.get('pool_size')} | {len(r.get('covered') or [])}/{n} |")
    lines.append("")
    qs = [q for r in rounds for q in (r.get("queries") or [])]
    if qs:
        pruned = {p.get("pub_num") for p in (search_stats or {}).get("pruned") or []}
        lines += ["| # | Kind | Query | Built from | Total | Taken | New | Kept after screen |", "|---|---|---|---|---|---|---|---|"]
        for i, q in enumerate(qs, 1):
            pubs = q.get("pubs") or []
            lines.append(f"| {q.get('n') or i} | {q.get('kind') or q.get('mode') or ''} | `{(q.get('query') or '')[:160]}` | "
                         f"{_query_explainer(q)} | {q.get('total') if q.get('total') is not None else '?'} | {q.get('hits')} | "
                         f"{q.get('new') if q.get('new') is not None else '–'} | {sum(1 for p in pubs if p in pruned) if pubs else '–'} |")
        lines.append("")
    return lines


# ── 3. element × document quote matrix ──

def quote_matrix_html(scoring_report: list[dict] | None, checklist: list[dict] | None, top_n: int = 8) -> str:
    docs = [d for d in (scoring_report or []) if d.get("checklist_results")][:top_n]
    crits = [c.get("criterion", "") for c in (checklist or []) if c.get("criterion")]
    if not docs or not crits:
        return ""
    out = ['<div class="sec"><div class="sec-t">Evidence Matrix</div>',
           '<div class="sec-note">Rows: your elements. Columns: top candidate documents. A cell shows the number of verbatim quotes '
           'the evaluator gave; green = every quote was found in the document, amber = some were, red = none (score set to 0). '
           'This report states blocking risk only; it does not predict grant.</div>',
           '<div style="overflow-x:auto"><table class="tbl"><thead><tr><th>Element</th>']
    for d in docs:
        out.append(f'<th title="{_e(d.get("title", ""))}">{_e((d.get("pub_num") or d.get("title", ""))[:18])}</th>')
    out.append("</tr></thead><tbody>")
    for crit in crits:
        out.append(f"<tr><td>{_e(crit[:90])}</td>")
        for d in docs:
            item = (d.get("checklist_results") or {}).get(crit) or {}
            qs = item.get("quote_checks") or []
            n_ok = sum(1 for q in qs if q.get("verified"))
            score = item.get("score")
            if score is None:
                score = 2 if item.get("match") else 0
            if not qs and score <= 0:
                cell, bg = "–", "transparent"
            elif not qs:
                cell, bg = f"s{score}", "#f1f5f9"
            elif n_ok == len(qs):
                cell, bg = f"{n_ok}✓", "#dcfce7"
            elif n_ok:
                cell, bg = f"{n_ok}/{len(qs)}", "#fef3c7"
            else:
                cell, bg = f"0/{len(qs)}", "#fee2e2"
            tip = " | ".join(q.get("quote", "")[:120] for q in qs[:3])
            out.append(f'<td style="background:{bg};text-align:center" title="{_e(tip)}">{cell}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div></div>")
    return "\n".join(out)


def quote_matrix_md(scoring_report: list[dict] | None, checklist: list[dict] | None, top_n: int = 6) -> list[str]:
    docs = [d for d in (scoring_report or []) if d.get("checklist_results")][:top_n]
    crits = [c.get("criterion", "") for c in (checklist or []) if c.get("criterion")]
    if not docs or not crits:
        return []
    head = "| Element | " + " | ".join((d.get("pub_num") or d.get("title", ""))[:18] for d in docs) + " |"
    lines = ["## Evidence Matrix", "", head, "|" + "---|" * (len(docs) + 1)]
    for crit in crits:
        cells = []
        for d in docs:
            item = (d.get("checklist_results") or {}).get(crit) or {}
            qs = item.get("quote_checks") or []
            n_ok = sum(1 for q in qs if q.get("verified"))
            cells.append(f"{n_ok}/{len(qs)}" if qs else "–")
        lines.append(f"| {crit[:70]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


# ── 4. prior-art determination (rule output of patent_analyzer.adjudicate) ──

_RISK_STYLE = {"blocking": ("#fee2e2", "#991b1b"), "relevant": ("#fef3c7", "#92400e"), "related": ("#e0f2fe", "#075985")}
_LABEL_TEXT = {"102": "single-reference blocking risk (§102 pattern)",
               "103": "combination blocking risk (§103 pattern)",
               "ALLOW": "no blocking reference among the evaluated documents"}


def adjudication_html(adj: dict | None) -> str:
    if not adj or not adj.get("n_elements"):
        return ""
    bg, fg = _RISK_STYLE.get(adj.get("risk", "related"), _RISK_STYLE["related"])
    n = adj["n_elements"]
    out = ['<div class="sec"><div class="sec-t">Prior-Art Determination</div>',
           '<div class="sec-note">Deterministic rule over the verified evidence above: an element counts as disclosed by a '
           'document only when the evaluator scored it and at least one verbatim quote was located in that document. '
           'One document disclosing every element is the §102 pattern; a union of up to three is the §103 pattern. '
           'This states blocking risk from the documents evaluated here — it is not a prediction of grant, and an '
           'unpublished application or a document outside the search can always change the picture.</div>',
           f'<div class="sec-b"><span class="badge" style="background:{bg};color:{fg}">{_e(adj.get("risk", ""))}</span> '
           f'<b>{_e(_LABEL_TEXT.get(adj.get("label"), adj.get("label", "")))}</b> — {_e(adj.get("reason", ""))}</div>']
    per = adj.get("per_doc_coverage") or []
    if per:
        out.append('<table class="tbl"><thead><tr><th>Document</th><th>Elements disclosed</th><th>Coverage</th>'
                   '<th>Missing</th></tr></thead><tbody>')
        for d in per[:10]:
            miss = d.get("missing") or []
            out.append(f'<tr><td title="{_e(d.get("title", ""))}">{_e(d.get("pub_num") or d.get("title", ""))[:24]}</td>'
                       f'<td>{d.get("n_covered", 0)}/{n}</td><td>{d.get("coverage", 0):.0%}</td>'
                       f'<td>{_e("; ".join(m[:60] for m in miss[:4]))}{" …" if len(miss) > 4 else ""}</td></tr>')
        out.append("</tbody></table>")
    combo = adj.get("combo")
    if combo and len(combo.get("docs") or []) >= 2:
        out.append(f'<div class="sec-b" style="margin-top:.5rem"><b>Best combination:</b> {_e(" + ".join(combo["docs"]))} '
                   f'covers {combo.get("n_covered", 0)}/{n} elements'
                   + (f'; still undisclosed: {_e("; ".join(m[:60] for m in combo["missing"][:4]))}' if combo.get("missing") else "")
                   + '</div>')
    out.append("</div>")
    return "\n".join(out)


def adjudication_md(adj: dict | None) -> list[str]:
    if not adj or not adj.get("n_elements"):
        return []
    n = adj["n_elements"]
    lines = ["## Prior-Art Determination", "",
             f"**{adj.get('risk', '')}** — {_LABEL_TEXT.get(adj.get('label'), adj.get('label', ''))}. {adj.get('reason', '')}",
             "", "_Blocking risk from the documents evaluated here only; not a prediction of grant._", ""]
    per = adj.get("per_doc_coverage") or []
    if per:
        lines += ["| Document | Disclosed | Coverage | Missing |", "|---|---|---|---|"]
        for d in per[:10]:
            miss = d.get("missing") or []
            lines.append(f"| {d.get('pub_num') or d.get('title', '')} | {d.get('n_covered', 0)}/{n} | "
                         f"{d.get('coverage', 0):.0%} | {'; '.join(m[:60] for m in miss[:4])}{' …' if len(miss) > 4 else ''} |")
        lines.append("")
    combo = adj.get("combo")
    if combo and len(combo.get("docs") or []) >= 2:
        lines += [f"**Best combination:** {' + '.join(combo['docs'])} covers {combo.get('n_covered', 0)}/{n} elements"
                  + (f"; still undisclosed: {'; '.join(m[:60] for m in combo['missing'][:4])}" if combo.get("missing") else ""), ""]
    return lines


def inject_html(report_html: str, extraction: dict | None, search_stats: dict | None,
                scoring_report: list[dict] | None, checklist: list[dict] | None,
                adjudication: dict | None = None) -> str:
    anchor = '<div class="sec-t">Invention Summary</div>'
    block = "\n".join(x for x in (extraction_html(extraction), loop_html(search_stats),
                                  quote_matrix_html(scoring_report, checklist),
                                  adjudication_html(adjudication)) if x)
    if not block:
        return report_html
    i = report_html.find(anchor)
    if i < 0:
        return report_html + block
    j = report_html.find("</div>\n</div>", i)
    j = report_html.find("</div>", j + 6) + 6 if j >= 0 else report_html.find("</div>", i) + 6
    return report_html[:j] + "\n" + block + report_html[j:]


def inject_md(report_md: str, extraction: dict | None, search_stats: dict | None,
              scoring_report: list[dict] | None, checklist: list[dict] | None,
              adjudication: dict | None = None) -> str:
    lines = (extraction_md(extraction) + loop_md(search_stats) + quote_matrix_md(scoring_report, checklist)
             + adjudication_md(adjudication))
    if not lines:
        return report_md
    marker = "## Novelty Assessment"
    block = "\n".join(lines)
    if marker in report_md:
        return report_md.replace(marker, block + "\n" + marker, 1)
    return report_md + "\n" + block
