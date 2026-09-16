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
import re


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _clip(text: str, n: int) -> str:
    """Cut on a word boundary, and say that it was cut.

    A hard slice produced "...via communicati; ...communication pul; ...a
    contrastiv" in the missing-elements sentence — a list of claim limitations
    chopped mid-word, which reads as a bug (Harry, 2026-09-20, job 99a35c00).
    """
    t = " ".join(str(text or "").split())
    if len(t) <= n:
        return t
    cut = t[:n]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > n * 0.6 else cut).rstrip(" ,;:") + "\u2026"


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
    out.append(lens_attribution_html(search_stats))
    out.append("</div>")
    return "\n".join(out)


# ── 2b. recall channel health ──
#
# A channel that timed out, crashed or hit a rate limit changes what the
# determination below is based on: the search was narrower than it looks.
# nodes/search.py records that per channel in search_stats["channel_health"];
# this renders it, and stays silent when every channel was fine.

_HEALTH_STYLE = {"timeout": ("#fee2e2", "#991b1b", "timed out"),
                 "crashed": ("#fee2e2", "#991b1b", "crashed"),
                 "errored": ("#fee2e2", "#991b1b", "errored"),
                 "limited": ("#fef3c7", "#92400e", "rate-limited / quota")}
_HEALTH_NOTE = ("These channels did not return everything they could have, so the prior art below is a "
                "narrower sample than a clean run would produce. A determination of \"no blocking reference\" "
                "is correspondingly weaker — re-run the phase once the limit clears before relying on it.")


def _degraded(search_stats: dict | None) -> list[dict]:
    return [h for h in ((search_stats or {}).get("channel_health") or [])
            if h.get("status") in _HEALTH_STYLE]


def channel_health_html(search_stats: dict | None) -> str:
    bad = _degraded(search_stats)
    if not bad:
        return ""
    ok = [h for h in ((search_stats or {}).get("channel_health") or []) if h.get("status") in ("ok", "empty")]
    out = ['<div class="sec"><div class="sec-t">Search Coverage Warnings</div>',
           f'<div class="sec-note">{_e(_HEALTH_NOTE)}</div>',
           '<table class="tbl"><thead><tr><th>Channel</th><th>What happened</th><th>Results</th>'
           '<th>Time</th><th>Detail</th></tr></thead><tbody>']
    for h in bad:
        bg, fg, label = _HEALTH_STYLE[h["status"]]
        out.append(f'<tr><td><code>{_e(h.get("channel"))}</code></td>'
                   f'<td><span style="background:{bg};color:{fg};padding:.1rem .4rem;border-radius:3px">{_e(label)}</span></td>'
                   f'<td>{h.get("n", 0)}</td><td>{h.get("seconds", 0)}s</td>'
                   f'<td>{_e(h.get("detail") or "")}</td></tr>')
    out.append("</tbody></table>")
    if ok:
        out.append('<div class="sec-note">Channels that ran normally: '
                   + ", ".join(f'<code>{_e(h["channel"])}</code> ({h.get("n", 0)})' for h in ok) + "</div>")
    out.append("</div>")
    return "\n".join(out)


def channel_health_md(search_stats: dict | None) -> list[str]:
    bad = _degraded(search_stats)
    if not bad:
        return []
    lines = ["## Search Coverage Warnings", "", _HEALTH_NOTE, "",
             "| Channel | What happened | Results | Time | Detail |", "|---|---|---|---|---|"]
    for h in bad:
        lines.append(f'| `{h.get("channel")}` | {_HEALTH_STYLE[h["status"]][2]} | {h.get("n", 0)} | '
                     f'{h.get("seconds", 0)}s | {(h.get("detail") or "").replace("|", "/")} |')
    lines.append("")
    return lines


# ── 2c. evidence coverage ──
#
# A document evaluated with source "no_content" was read from nothing: no PDF
# (the download failed or none was offered) and no abstract worth the name. It
# still occupies a row in the evaluation, and it still counts as "checked" when
# the determination says no reference discloses the invention. Measured on the
# M2 e2e runs: 17/25 and 24/25 no_content. That has to be on the page.

_EVIDENCE_NOTE = ("A reference read from nothing cannot disclose anything, so every no-content row below is "
                  "an unchecked reference, not a cleared one. Treat the determination as provisional while "
                  "this fraction is high \u2014 the usual causes are prior-art PDF downloads failing and "
                  "patent candidates arriving with a title but no abstract.")


def _evidence_mix(scoring_report: list[dict] | None) -> dict:
    mix: dict[str, int] = {}
    for r in scoring_report or []:
        k = r.get("source") or "unknown"
        mix[k] = mix.get(k, 0) + 1
    return mix


# What a document's `source` means, spelled out. "claims_only" is a real answer,
# not a gap: amie_patents.descriptions holds the specification for US
# publications and there is none for any other country in the public source, so
# a non-US patent is read from its claims (when it has any) and its abstract.
_EV_LABEL = {"pdf": "full PDF", "full_text": "abstract + claims + specification",
             "claims_only": "abstract + claims only (no specification: non-US publication)",
             "abstract": "abstract only", "no_content": "nothing to read"}


_READ_GAP_NOTE = ("Every reference in that difference is an <b>unchecked</b> reference, not a cleared one. "
                  "A determination of \u201cno blocking reference\u201d rests on what was read, not on what "
                  "was delivered \u2014 read the row below that accounts for the difference before relying on it.")


def read_gap_html(read_gap: dict | None) -> str:
    """Delivered vs deep-read, itemised. Printed whenever they differ, in red,
    above everything else the report injects."""
    g = read_gap or {}
    if not g.get("delivered"):
        return ""
    if not g.get("shortfall"):
        return ('<div class="sec"><div class="sec-t">Reading Coverage</div>'
                f'<div class="sec-note">All {int(g["delivered"])} delivered references were deep-read.</div></div>')
    rows = "".join(f'<tr><td>{_e(k)}</td><td>{v}</td></tr>' for k, v in
                   sorted((g.get("reasons") or {}).items(), key=lambda kv: -kv[1]))
    return ('<div class="sec" style="border:2px solid #dc2626">'
            '<div class="sec-t" style="color:#dc2626">Reading Coverage</div>'
            f'<div class="sec-note" style="color:#dc2626"><b>Delivered {int(g["delivered"])} references, '
            f'deep-read {int(g["read"])}. {int(g["shortfall"])} were never read.</b> {_READ_GAP_NOTE}</div>'
            '<table class="tbl"><thead><tr><th>Why a reference was not read</th><th>References</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>')


def read_gap_md(read_gap: dict | None) -> list[str]:
    g = read_gap or {}
    if not g.get("delivered"):
        return []
    if not g.get("shortfall"):
        return ["## Reading Coverage", "",
                f"All {int(g['delivered'])} delivered references were deep-read.", ""]
    lines = ["## Reading Coverage", "",
             f"**Delivered {int(g['delivered'])} references, deep-read {int(g['read'])}. "
             f"{int(g['shortfall'])} were never read.** "
             + re.sub(r"</?b>", "**", _READ_GAP_NOTE), "",
             "| Why a reference was not read | References |", "|---|---|"]
    for k, v in sorted((g.get("reasons") or {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return lines


def evidence_coverage_html(scoring_report: list[dict] | None) -> str:
    mix = _evidence_mix(scoring_report)
    total, empty = sum(mix.values()), mix.get("no_content", 0)
    if not total or not empty:
        return ""
    out = ['<div class="sec"><div class="sec-t">Evidence Coverage</div>',
           f'<div class="sec-note"><b>{empty} of {total} references were evaluated with no text at all.</b> '
           f'{_e(_EVIDENCE_NOTE)}</div>',
           '<table class="tbl"><thead><tr><th>What was read</th><th>References</th><th>Share</th></tr></thead><tbody>']
    for k, n in sorted(mix.items(), key=lambda kv: -kv[1]):
        hl = ' style="background:#fee2e2"' if k == "no_content" else ""
        out.append(f'<tr{hl}><td>{_e(_EV_LABEL.get(k, k))}</td><td>{n}</td><td>{100 * n / total:.0f}%</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


def evidence_coverage_md(scoring_report: list[dict] | None) -> list[str]:
    mix = _evidence_mix(scoring_report)
    total, empty = sum(mix.values()), mix.get("no_content", 0)
    if not total or not empty:
        return []
    lines = ["## Evidence Coverage", "",
             f"**{empty} of {total} references were evaluated with no text at all.** {_EVIDENCE_NOTE}", "",
             "| What was read | References | Share |", "|---|---|---|"]
    for k, n in sorted(mix.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {_EV_LABEL.get(k, k)} | {n} | {100 * n / total:.0f}% |")
    lines.append("")
    return lines


# ── 2d. paper full text ──
#
# Where each paper's text came from, and — the part that matters — which papers
# have no reachable copy at all. Those are not failures to hide: a paper behind
# a publisher paywall is deliberately not fetched, because OSU Libraries'
# Responsible Use policy forbids programmatic downloading of licensed content
# and a violation can suspend access for the whole university. The section ends
# with the list to open by hand instead. See patent_analyzer/fulltext.py.

_FT_LABEL = {"arxiv": "arXiv", "oa": "open access", "abstract_only": "no open-access copy"}

_FT_NOTE = ("<b>Resolved</b> is how many papers an open-access copy was found for; <b>read</b> is how many "
            "of those actually returned a PDF. They differ, and the second number is the one that counts: "
            "publisher hosts answer an automated client with a 403 and PubMed Central puts a proof-of-work "
            "challenge in front of the file. Everything not read was evaluated from its abstract. No copy is "
            "fetched through the university subscription, on purpose: OSU Libraries' Responsible Use of "
            "Licensed Electronic Resources forbids “using scripts, spiders, crawlers, or other computer "
            "programs to automatically download content”, and states that a violation “may suspend "
            "access for the entire OSU community”. Open the pages below yourself if you need them read.")

_FT_POLICY_URL = "https://library.oregonstate.edu/responsible-use-licensed-electronic-resources"


def _ft_stats(search_stats: dict | None) -> dict:
    ft = (search_stats or {}).get("fulltext_oa") or {}
    return ft if ft.get("papers") else {}


def fulltext_tier_html(search_stats: dict | None) -> str:
    ft = _ft_stats(search_stats)
    if not ft:
        return ""
    total = int(ft.get("papers", 0))
    out = ['<div class="sec"><div class="sec-t">Paper Full Text</div>',
           f'<div class="sec-note">{_e(_FT_NOTE)} '
           f'<a href="{_FT_POLICY_URL}">Policy</a>.</div>',
           '<table class="tbl"><thead><tr><th>Source</th><th>Resolved</th><th>Share</th>'
           '<th>Read as PDF</th></tr></thead><tbody>']
    read = ft.get("read") or {}
    for k in ("arxiv", "oa", "abstract_only"):
        n = int(ft.get(k, 0))
        r = int(read.get(k, 0))
        hl = ' style="background:#fef3c7"' if k == "abstract_only" and n else ""
        out.append(f'<tr{hl}><td>{_e(_FT_LABEL[k])}</td><td>{n}</td>'
                   f'<td>{100 * n / total:.0f}%</td><td>{r if k != "abstract_only" else "&mdash;"}</td></tr>')
    n_read = sum(int(read.get(k, 0)) for k in ("arxiv", "oa"))
    out.append(f'<tr><td><b>read</b></td><td colspan="2"></td><td><b>{n_read} of {total} '
               f'({100 * n_read / total:.0f}%)</b></td></tr>')
    out.append("</tbody></table>")
    rows = ft.get("manifest") or []
    if rows:
        out.append(f'<div class="sec-note"><b>{len(rows)} paper(s) to open by hand.</b></div>')
        out.append('<table class="tbl"><thead><tr><th>#</th><th>DOI</th><th>Title</th>'
                   '<th>Landing page</th><th>Why not automatic</th></tr></thead><tbody>')
        for i, r in enumerate(rows, 1):
            lp = r.get("landing_page") or ""
            link = f'<a href="{_e(lp)}">{_e(lp[:60])}</a>' if lp else "&mdash;"
            out.append(f'<tr><td>{i}</td><td>{_e(r.get("doi") or "—")}</td>'
                       f'<td>{_e((r.get("title") or "")[:90])}</td><td>{link}</td>'
                       f'<td>{_e(r.get("reason") or "")}</td></tr>')
        out.append("</tbody></table>")
    out.append("</div>")
    return "\n".join(out)


def fulltext_tier_md(search_stats: dict | None) -> list[str]:
    ft = _ft_stats(search_stats)
    if not ft:
        return []
    total = int(ft.get("papers", 0))
    note = _FT_NOTE.replace("<b>", "**").replace("</b>", "**")
    lines = ["## Paper Full Text", "", f"{note} <{_FT_POLICY_URL}>", "",
             "| Source | Resolved | Share | Read as PDF |", "|---|---|---|---|"]
    read = ft.get("read") or {}
    for k in ("arxiv", "oa", "abstract_only"):
        n = int(ft.get(k, 0))
        r = int(read.get(k, 0)) if k != "abstract_only" else "-"
        lines.append(f"| {_FT_LABEL[k]} | {n} | {100 * n / total:.0f}% | {r} |")
    n_read = sum(int(read.get(k, 0)) for k in ("arxiv", "oa"))
    lines += [f"| **read** | | | **{n_read} of {total} ({100 * n_read / total:.0f}%)** |", ""]
    rows = ft.get("manifest") or []
    if rows:
        lines += [f"**{len(rows)} paper(s) to open by hand.**", "",
                  "| # | DOI | Title | Landing page | Why not automatic |", "|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            lines.append(f"| {i} | {r.get('doi') or '-'} | {(r.get('title') or '-')[:90]} | "
                         f"{r.get('landing_page') or '-'} | {r.get('reason') or ''} |")
        lines.append("")
    return lines


def lens_attribution_html(search_stats: dict | None) -> str:
    """Lens trial terms: results sourced from Lens carry "Data Sourced from The Lens"
    with a link and the logo (Lens.org attribution requirement, trial to 2026-10-02)."""
    used = any((r.get("lens") or {}).get("bridge_patents") or (r.get("lens") or {}).get("search_calls")
               for r in (search_stats or {}).get("loop_rounds") or [])
    if not used:
        return ""
    return ('<div class="sec-note" style="margin-top:.5rem"><a href="https://www.lens.org" target="_blank" rel="noopener">'
            '<img src="https://about.lens.org/wp-content/uploads/2021/04/Lens-logo-tagline.png" alt="The Lens" style="height:18px;vertical-align:middle;margin-right:.35rem">'
            'Data Sourced from The Lens</a> — paper→patent bridge and CPC-scoped patent searches (lens_bridge / lens_search sources).</div>')


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
        if q.get("observation") or q.get("decision"):
            # the ReAct loop's own reasoning for this query (Harry H6: every step must be readable)
            out.append(f'<tr><td></td><td colspan="9" class="sec-note"><b>saw:</b> {_e(q.get("observation") or "")} '
                       f'<b>· chose:</b> {_e(q.get("decision") or "")}'
                       + (f' <b>· CPC</b> {_e(q.get("cpc_group"))}' + (" (forced: an unused predicted group)" if q.get("cpc_forced") else "")
                          if q.get("cpc_group") else "") + '</td></tr>')
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

_MATRIX_NOTE = ("Rows: your elements. Columns: top candidate documents. Green = this reference discloses the element "
                "by the same rule the determination above uses (scored at or above the threshold AND at least one quote "
                "located in the reference); amber = scored but the quote could not be located, so it does not count; "
                "grey dash = not disclosed or not evaluated. The count is verbatim quotes located. This report states "
                "blocking risk only; it does not predict grant.")


def _cell_of(adjudication: dict | None, doc: dict, crit: str) -> dict | None:
    """The (document, element) cell adjudicate already decided, or None.

    Never re-derive coverage from `score` here: the determination uses
    element_covered (min_score AND a verified quote) and a second, looser rule in
    this file is exactly how one page came to show "best single covers 4/5"
    beside a badge reading "1/5 covered"."""
    key = (doc.get("pub_num") or "").strip()
    title = (doc.get("title") or "").strip()
    for row in (adjudication or {}).get("per_doc_coverage") or []:
        if (key and row.get("pub_num") == key) or (not key and title and row.get("title") == title):
            return (row.get("cells") or {}).get(crit)
    return None


def quote_matrix_html(scoring_report: list[dict] | None, checklist: list[dict] | None, top_n: int = 8,
                      adjudication: dict | None = None) -> str:
    docs = [d for d in (scoring_report or []) if d.get("checklist_results")][:top_n]
    crits = [c.get("criterion", "") for c in (checklist or []) if c.get("criterion")]
    if not docs or not crits:
        return ""
    out = ['<div class="sec"><div class="sec-t">Evidence Matrix</div>',
           f'<div class="sec-note">{_e(_MATRIX_NOTE)}</div>',
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
            cell_info = _cell_of(adjudication, d, crit)
            if cell_info is not None:
                covered = bool(cell_info.get("covered"))
                reason = cell_info.get("reason") or ""
            else:                      # no adjudication (e.g. no elements): fall back, and say so
                covered = bool(qs) and n_ok == len(qs)
                reason = "" if covered else "quote_not_verified" if qs else "not_disclosed"
            if covered:
                cell, bg = f"{n_ok}✓" if qs else "✓", "#dcfce7"
            elif reason == "quote_not_verified":
                cell, bg = (f"{n_ok}/{len(qs)}" if qs else "s?"), "#fef3c7"
            elif reason == "below_min_score":
                cell, bg = f"{n_ok}/{len(qs)}" if qs else "low", "#fef3c7"
            else:
                cell, bg = "–", "transparent"
            tip = " | ".join(q.get("quote", "")[:120] for q in qs[:3]) or reason.replace("_", " ")
            out.append(f'<td style="background:{bg};text-align:center" title="{_e(tip)}">{cell}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div></div>")
    return "\n".join(out)


def quote_matrix_md(scoring_report: list[dict] | None, checklist: list[dict] | None, top_n: int = 6,
                    adjudication: dict | None = None) -> list[str]:
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
            cell_info = _cell_of(adjudication, d, crit)
            covered = bool(cell_info.get("covered")) if cell_info is not None else (bool(qs) and n_ok == len(qs))
            mark = "**✓**" if covered else ""
            cells.append(f"{mark}{n_ok}/{len(qs)}" if qs else (mark or "–"))
        lines.append(f"| {crit[:70]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


# ── 3b. draft claims (nodes/draft.py output; for attorney review) ──

_DRAFT_NOTE = ("Drafted from the grounded elements and the evidence matrix above. Each limitation cites the passage of your "
               "document it comes from and which evaluated references disclose it. This is a starting point for a patent "
               "attorney; it does not predict grant.")
_ORIGIN_ICON = {"element": "●", "dependent_hint": "◆", "component_element": "◇", "refinement": "▹", "distinguishing": "★"}
_STRATEGY_TEXT = {"as_is": "Independent claims as extracted",
                  "narrowed": "Independent claims narrowed by a limitation none of the charted references discloses",
                  "unresolved": "Unresolved — every candidate limitation is disclosed by the charted references",
                  "no_prior_art": "No evaluated prior art — claims drafted, coverage unknown",
                  "no_elements": "No grounded elements to draft from"}


def _cov_badge(l: dict) -> str:
    cov = l.get("coverage") or {}
    rc = cov.get("recheck") or {}
    if cov.get("status") == "unknown" or not cov.get("checked_against"):
        badge = '<span class="badge" style="background:#f1f5f9;color:#475569">unknown</span>'
    elif cov.get("covered_by"):
        badge = f'<span class="badge" style="background:#e2e8f0;color:#334155">disclosed by {_e(", ".join(cov["covered_by"]))}</span>'
    else:
        badge = '<span class="badge" style="background:#dcfce7;color:#166534">not disclosed by charted refs</span>'
    if rc.get("queried"):
        if rc.get("covered_by_new"):
            badge += f' <span class="badge" style="background:#fee2e2;color:#991b1b">re-check: disclosed by {_e(", ".join(rc["covered_by_new"]))}</span>'
        else:
            badge += f' <span class="badge" style="background:#e0f2fe;color:#075985">re-check: {rc.get("new_docs", 0)} new docs, none disclose</span>'
    return badge


def _footnotes(draft: dict) -> tuple[dict[str, int], list[tuple[int, str, dict]]]:
    """lid -> footnote number; [(n, lid, basis)] in claim order."""
    idx, rows = {}, []
    for c in draft.get("claims") or []:
        for l in c.get("limitations") or []:
            for b in l.get("basis") or []:
                n = len(rows) + 1
                idx.setdefault(l.get("lid"), n)
                rows.append((n, l.get("lid", ""), b))
    return idx, rows


def _where(loc) -> str:
    if not isinstance(loc, dict):
        return ""
    return ", ".join(f"{k} {loc[k]}" for k in ("section", "heading", "para", "source", "method") if loc.get(k) not in (None, "", []))


def draft_html(draft: dict | None, extraction: dict | None = None) -> str:
    if not draft or not draft.get("claims"):
        if (draft or {}).get("strategy") == "no_elements":
            return ('<div class="sec"><div class="sec-t">Draft Claims (for attorney review)</div>'
                    '<div class="sec-b">No grounded elements were available to draft from.</div></div>')
        return ""
    strategy = draft.get("strategy", "")
    av = draft.get("avoidance") or {}
    fn, rows = _footnotes(draft)
    out = ['<div class="sec draft-sec"><div class="sec-t">Draft Claims (for attorney review)</div>',
           f'<div class="sec-note">{_e(_DRAFT_NOTE)}</div>']
    # 1. strategy line
    color = {"narrowed": ("#dcfce7", "#166534"), "unresolved": ("#fee2e2", "#991b1b"), "as_is": ("#e0f2fe", "#075985")}.get(strategy, ("#f1f5f9", "#475569"))
    out.append(f'<div class="sec-b"><span class="badge" style="background:{color[0]};color:{color[1]}">{_e(strategy)}</span> '
               f'<b>{_e(_STRATEGY_TEXT.get(strategy, strategy))}</b><br><span style="font-size:.85em">{_e(av.get("reason", ""))}'
               + (f' <i>{_e(av.get("recheck_reason", ""))}</i>' if av.get("recheck_reason") else "") + '</span></div>')
    # 2. claims
    out.append('<div class="sec-b" style="font-family:Georgia,serif;line-height:1.55">')
    for c in draft["claims"]:
        edited = ' <span class="badge" style="background:#fef3c7;color:#92400e">edited by reviewer</span>' if c.get("edited_by_user") else ""
        out.append(f'<div style="margin:.7rem 0"><b>{c.get("no")}.</b> {_e(c.get("preamble", ""))}{edited}')
        lims = c.get("limitations") or []
        for i, l in enumerate(lims):
            icon = _ORIGIN_ICON["distinguishing"] if l.get("distinguishing") else _ORIGIN_ICON.get(l.get("origin", ""), "")
            tail = ";" if i < len(lims) - 2 else ("; and" if i == len(lims) - 2 else ".")
            n = fn.get(l.get("lid"))
            flags = [f for f in l.get("flags") or [] if not f.get("fixed")]
            red = ' style="background:#fee2e2"' if flags else ""
            ed = ' <span class="badge" style="background:#fef3c7;color:#92400e">edited by reviewer</span>' if l.get("edited_by_user") else ""
            out.append(f'<div style="margin-left:1.6rem"{red}><span title="{_e(l.get("origin", ""))}" style="color:var(--text2)">{icon}</span> '
                       f'{_e(l.get("text", ""))}{tail}<sup>[{n}]</sup> {_cov_badge(l)}{ed}</div>')
        if c.get("depends_on") is None and not lims:
            out.append("</div>")
        else:
            out.append("</div>")
    out.append('<div class="sec-note">● element · ◆ dependent hint · ◇ component element · ▹ refinement · ★ distinguishing (not disclosed by the '
               'charted references). Superscripts refer to the basis table; "disclosed by" reflects the charted references and the re-check only.</div></div>')
    # 3. basis footnotes
    out.append('<details open style="margin:.5rem 0"><summary><b>Basis</b> — where each limitation comes from (verbatim)</summary>'
               '<table class="tbl"><thead><tr><th>#</th><th>limitation</th><th>element</th><th>verbatim quote</th><th>where</th></tr></thead><tbody>')
    for n, lid, b in rows:
        out.append(f'<tr><td>{n}</td><td>{_e(lid)}</td><td>{_e(b.get("element_id", ""))}</td><td><q>{_e(b.get("evidence_quote", ""))}</q></td>'
                   f'<td>{_e(_where(b.get("evidence_loc")))}</td></tr>')
    out.append("</tbody></table></details>")
    # 4. 112(b)
    d = draft.get("definiteness") or {}
    flags = d.get("flags") or []
    out.append(f'<details {"open" if d.get("open_flags") else ""} style="margin:.5rem 0"><summary><b>112(b) self-check</b> — '
               f'{len(flags)} rule flag(s), {len(d.get("open_flags") or [])} open after {d.get("passes", 0)} pass(es)</summary>')
    if flags:
        out.append('<table class="tbl"><thead><tr><th>claim.limitation</th><th>category</th><th>span</th><th>rule</th><th>fixed?</th><th>note</th></tr></thead><tbody>')
        for f in flags:
            st = "" if f.get("fixed") else ' style="background:#fee2e2"'
            out.append(f'<tr{st}><td>{_e(f.get("lid", ""))}</td><td>{_e(f.get("category", ""))}</td><td>{_e(f.get("span", ""))}</td>'
                       f'<td>{_e(f.get("rule", ""))}</td><td>{"yes" if f.get("fixed") else "open"}</td><td>{_e(f.get("note", ""))}</td></tr>')
        out.append("</tbody></table>")
    else:
        out.append('<div class="sec-b">No rule flags (antecedent basis, relative terms, exemplary phrasing, 112(f) placeholders).</div>')
    adv = d.get("llm_advisory") or {}
    if adv:
        out.append('<details style="margin:.3rem 0"><summary>LLM advisory (undefined terms, contradictions, omissions — advisory only, text unchanged)</summary><ul>')
        for no in sorted(adv, key=lambda x: int(x) if str(x).isdigit() else 0):
            a = adv[no] or {}
            reasons = "; ".join(f'{r.get("category")}: {", ".join(r.get("claim_recitations") or [])[:120]}' for r in a.get("reasons") or []) or "no issue raised"
            out.append(f'<li>claim {_e(no)} — {_e(a.get("likelihood", ""))} ({a.get("p_indefinite", 0):.2f}): {_e(reasons)}</li>')
        out.append("</ul></details>")
    out.append("</details>")
    # 5. re-check
    rc = draft.get("recheck") or {}
    if rc and not rc.get("skipped"):
        out.append(f'<details style="margin:.5rem 0"><summary><b>Re-check</b> — {len(rc.get("queries") or [])} queries, '
                   f'{len(rc.get("new_docs") or [])} new documents, {rc.get("evaluated", 0)} evaluated on the new limitations</summary>')
        if rc.get("queries"):
            out.append('<table class="tbl"><thead><tr><th>query</th><th>channel</th><th>total</th><th>new</th></tr></thead><tbody>')
            for q in rc["queries"]:
                out.append(f'<tr><td><code>{_e(q.get("query", ""))}</code></td><td>{_e(q.get("channel", ""))}</td><td>{_e(q.get("total"))}</td><td>{q.get("new", 0)}</td></tr>')
            out.append("</tbody></table>")
        if rc.get("new_docs"):
            out.append('<table class="tbl"><thead><tr><th>document</th><th>text</th><th>discloses</th></tr></thead><tbody>')
            for nd in rc["new_docs"]:
                name = _e(nd.get("pub_num") or nd.get("title", ""))
                name = f'<a href="{_e(nd["url"])}" target="_blank">{name}</a>' if nd.get("url") else name
                out.append(f'<tr><td>{name} <span style="color:var(--text2)">{_e((nd.get("title") or "")[:80])}</span></td><td>{_e(nd.get("text_mode", ""))}</td>'
                           f'<td>{_e(", ".join(nd.get("covered") or [])) or "—"}</td></tr>')
            out.append("</tbody></table>")
        out.append("</details>")
    elif rc.get("skipped"):
        out.append(f'<div class="sec-note">Re-check skipped{": " + _e(rc.get("reason") or rc.get("error") or "") if (rc.get("reason") or rc.get("error")) else ""}.</div>')
    # 6. A2 draft for comparison
    core = next((c for c in (extraction or {}).get("candidate_inventions") or [] if c.get("id") == draft.get("candidate_id")), None)
    a2 = (core or {}).get("independent_claim_draft") or {}
    if a2.get("method") or a2.get("system"):
        out.append('<details style="margin:.5rem 0"><summary>Pre-search draft (A2) for comparison</summary>')
        for k in ("method", "system"):
            if a2.get(k):
                out.append(f'<div class="sec-b" style="margin:.3rem 0"><b>{k}:</b> {_e(a2[k])}</div>')
        out.append("</details>")
    out.append("</div>")
    return "\n".join(out)


def draft_md(draft: dict | None, extraction: dict | None = None) -> list[str]:
    if not draft or not draft.get("claims"):
        return ["## Draft Claims (for attorney review)", "", "No grounded elements were available to draft from.", ""] \
            if (draft or {}).get("strategy") == "no_elements" else []
    av = draft.get("avoidance") or {}
    fn, rows = _footnotes(draft)
    lines = ["## Draft Claims (for attorney review)", "", f"_{_DRAFT_NOTE}_", "",
             f"**Strategy: {draft.get('strategy')}** — {_STRATEGY_TEXT.get(draft.get('strategy'), '')}. {av.get('reason', '')}"
             + (f" {av.get('recheck_reason')}" if av.get("recheck_reason") else ""), ""]
    for c in draft["claims"]:
        lines.append(f"**{c.get('no')}.** {c.get('preamble', '')}")
        lims = c.get("limitations") or []
        for i, l in enumerate(lims):
            cov = l.get("coverage") or {}
            if cov.get("status") == "unknown" or not cov.get("checked_against"):
                tag = "unknown"
            elif cov.get("covered_by"):
                tag = "disclosed by " + ", ".join(cov["covered_by"])
            else:
                tag = "not disclosed by charted refs"
            rc = cov.get("recheck") or {}
            if rc.get("queried"):
                tag += "; re-check: " + (("disclosed by " + ", ".join(rc["covered_by_new"])) if rc.get("covered_by_new") else f"{rc.get('new_docs', 0)} new docs, none disclose")
            tail = ";" if i < len(lims) - 2 else ("; and" if i == len(lims) - 2 else ".")
            star = " ★" if l.get("distinguishing") else ""
            open_f = [f for f in l.get("flags") or [] if not f.get("fixed")]
            lines.append(f"    - {l.get('text', '')}{tail} [^{fn.get(l.get('lid'))}] _({l.get('origin', '')}{star}; {tag}"
                         + (f"; 112(b) open: {', '.join(f['category'] for f in open_f)}" if open_f else "") + ")_")
        lines.append("")
    lines += ["**Basis**", "", "| # | limitation | element | verbatim quote | where |", "|---|---|---|---|---|"]
    for n, lid, b in rows:
        q = (b.get("evidence_quote") or "").replace("|", "\\|")
        lines.append(f"| {n} | {lid} | {b.get('element_id', '')} | {q} | {_where(b.get('evidence_loc'))} |")
    d = draft.get("definiteness") or {}
    lines += ["", f"**112(b) self-check** — {len(d.get('flags') or [])} rule flag(s), {len(d.get('open_flags') or [])} open after {d.get('passes', 0)} pass(es)", ""]
    if d.get("flags"):
        lines += ["| claim.limitation | category | span | rule | fixed? |", "|---|---|---|---|---|"]
        lines += [f"| {f.get('lid', '')} | {f.get('category', '')} | {f.get('span', '')} | {f.get('rule', '')} | {'yes' if f.get('fixed') else 'OPEN'} |" for f in d["flags"]]
    adv = d.get("llm_advisory") or {}
    if adv:
        lines += ["", "LLM advisory (advisory only):"]
        for no in sorted(adv, key=lambda x: int(x) if str(x).isdigit() else 0):
            a = adv[no] or {}
            reasons = "; ".join(f"{r.get('category')}: {', '.join(r.get('claim_recitations') or [])[:120]}" for r in a.get("reasons") or []) or "no issue raised"
            lines.append(f"- claim {no} — {a.get('likelihood', '')} ({a.get('p_indefinite', 0):.2f}): {reasons}")
    rc = draft.get("recheck") or {}
    if rc and not rc.get("skipped"):
        lines += ["", f"**Re-check** — {len(rc.get('queries') or [])} queries, {len(rc.get('new_docs') or [])} new documents, {rc.get('evaluated', 0)} evaluated", ""]
        if rc.get("queries"):
            lines += ["| query | channel | total | new |", "|---|---|---|---|"]
            lines += [f"| `{q.get('query', '')}` | {q.get('channel', '')} | {q.get('total')} | {q.get('new', 0)} |" for q in rc["queries"]]
        if rc.get("new_docs"):
            lines += ["", "| document | text | discloses |", "|---|---|---|"]
            lines += [f"| {nd.get('pub_num') or nd.get('title', '')} | {nd.get('text_mode', '')} | {', '.join(nd.get('covered') or []) or '—'} |" for nd in rc["new_docs"]]
    core = next((c for c in (extraction or {}).get("candidate_inventions") or [] if c.get("id") == draft.get("candidate_id")), None)
    a2 = (core or {}).get("independent_claim_draft") or {}
    if a2.get("method") or a2.get("system"):
        lines += ["", "Pre-search draft (A2) for comparison:"]
        lines += [f"- {k}: {a2[k]}" for k in ("method", "system") if a2.get(k)]
    lines.append("")
    return lines


# ── 4. prior-art determination (rule output of patent_analyzer.adjudicate + claim chart) ──

_RISK_STYLE = {"blocking": ("#fee2e2", "#991b1b"), "relevant": ("#fef3c7", "#92400e"), "related": ("#e0f2fe", "#075985")}
# Plain sentence first, statute in brackets. "§102" and "§103 flags" on their own told a
# reader nothing about what this report was claiming (Harry, 2026-09-20, on job 99a35c00):
# the label has to say what was found before it says which section it lives under.
_LABEL_TEXT = {"102": "One document already shows everything (§102, anticipation)",
               "103": "Two or three documents together show everything "
                      "(§103 screen — a flag for review, not a legal obviousness finding)",
               "ALLOW": "No document or combination we read shows all the elements"}
_SCORE_TEXT = {2: "Present", 1: "Partial", 0: "–"}
_DETERMINATION_NOTE = ("Deterministic rule over the verified evidence: an element counts as disclosed by a reference only when "
                       "the evaluator scored it and at least one verbatim quote was located in that reference's text. One "
                       "reference disclosing every element is the §102 pattern (MPEP 2131). The §103 flags — a union of up to "
                       "three references, or a primary reference disclosing at least 70% of the elements — are screening "
                       "heuristics, not the statutory obviousness test: the 70% threshold comes from the PANORAMA "
                       "benchmark's scoring rule (App. C.5.3 (a)) and has no MPEP basis, and neither flag establishes the Graham "
                       "inquiries (MPEP 2141 II), an articulated rationale (MPEP 2143 I.(A)-(G)), a motivation to combine "
                       "(2143.01) or a reasonable expectation of success (2143.02). They say the elements are known across the "
                       "art, which is where an examiner would start, not where one would finish. This states blocking risk from "
                       "the documents evaluated here only — an unpublished application or a document outside the search can "
                       "always change the picture.")


_ABSTRACT_ONLY_NOTE = ("Columns marked “abstract only” are references whose full text could not be obtained — "
                       "most journal articles are behind bot protection or are not open access at all. For those, "
                       "an empty cell means the abstract does not mention the element, NOT that the reference fails "
                       "to disclose it, and no such reference is treated as disclosing the whole invention on its "
                       "own. Upload the full text and re-run the evidence step to settle them.")

_BRI_NOTE = ("Claims were construed under the broadest reasonable interpretation (MPEP 2111): the criteria "
             "were read the way a US examiner reads a pending claim, so a reference satisfies one when it "
             "discloses the same thing under a different name.")


def asserted_rationale_note(adj: dict | None) -> str:
    """Shown on the verdict card and next to the chart when the §103 rests on a
    reason nobody could quote. MPEP 2143.01 allows a motivation to be found in
    the skilled person's background knowledge, which has no quote — so this is
    permitted, and it is labelled every single place the verdict appears."""
    from .adjudicate import asserted_rationale
    note = asserted_rationale(adj)
    return (f"This §103 rests on a {note}. Every other finding is quoted from a reference and located "
            f"in its text; this one is not, so it is not part of a prima facie case.") if note else ""


def abstract_only_note(chart: dict | None) -> str:
    """Shown only when a chart actually has such a column."""
    return _ABSTRACT_ONLY_NOTE if any(d.get("abstract_only") for d in (chart or {}).get("docs") or []) else ""


def construction_note(adj: dict | None) -> str:
    """How the claims were read, when the evaluation recorded it. Absent on
    jobs run before the construction was tracked — say nothing rather than
    assert a construction the run did not record."""
    c = ((adj or {}).get("construction") or "").lower()
    return _BRI_NOTE if c == "bri" else ""


def determination_label(adj: dict | None) -> str:
    if not adj:
        return ""
    from .adjudicate import asserted_rationale
    suffix = f" — {asserted_rationale(adj)}" if asserted_rationale(adj) else ""
    label = adj.get("label")
    if label == "103" and adj.get("basis") == "primary_partial":
        thr = ((adj.get("params") or {}).get("single_partial_103") or 0.7)
        return (f"Best single document shows most of it (≥{thr:.0%} — screening flag)" + suffix)
    return _LABEL_TEXT.get(label, str(label or "")) + suffix


def coverage_lines(adj: dict | None) -> list[str]:
    """The two numbers the verdict rests on, as sentences.

    They replace `Rule: best single coverage 25%; union of 1 documents covers
    2/8`, which is the rule's own log line: correct, and unreadable by the
    person the report is for. The rule text is still in the report, under "How
    this was decided" — moved, not dropped.
    """
    if not adj or not adj.get("n_elements"):
        return []
    n = adj["n_elements"]
    per = adj.get("per_doc_coverage") or []
    best_n = per[0]["n_covered"] if per else 0
    combo_n = ((adj.get("combo") or {}).get("n_covered")) or 0
    k = (adj.get("params") or {}).get("max_combo", 3)
    return [f"Best single document: {best_n} of {n} elements.",
            f"Best combination (up to {k} documents): {combo_n} of {n}."]


def how_it_was_decided(adj: dict | None, chart: dict | None = None) -> list[str]:
    """Everything an attorney needs and nobody else reads, in one place.

    It used to be the first thing on the verdict card: a single paragraph
    carrying BRI, 2131, 2141, 2143, 2143.01, 2143.02 and PANORAMA C.5.3 before
    the reader got to the answer. Same words, folded.
    """
    parts = [asserted_rationale_note(adj), construction_note(adj), _DETERMINATION_NOTE,
             abstract_only_note(chart)]
    reason = (adj or {}).get("reason") or ""
    if reason:
        parts.append("Rule output, verbatim: " + reason)
    return [p for p in parts if p]


def _combo_lines(chart: dict | None) -> list[tuple[str, list[str], str]]:
    """(reference name, elements it discloses, url) per chart column."""
    out = []
    for i, d in enumerate((chart or {}).get("docs") or []):
        els = [r["element"] for r in chart.get("rows") or [] if r["cells"][i].get("covered")]
        out.append((d.get("pub_num") or d.get("title") or d.get("key") or f"ref {i + 1}", els, d.get("url") or ""))
    return out


def claim_chart_html(chart: dict | None) -> str:
    if not chart or not chart.get("rows") or not chart.get("docs"):
        return ""
    n = chart.get("n_elements", len(chart["rows"]))
    # The legend goes above the table. Under it, a reader hits "Partial · 1✓" with
    # nothing to read it against and has to scroll past the whole chart to find out.
    out = ['<div class="sec-note" style="margin-bottom:.4rem">How to read this: <b>Present</b> / <b>Partial</b> is the '
           'evaluator\'s score for that element, and <b>n\u2713</b> is how many verbatim quotes were located in that '
           'document. <b>"quote not located"</b> means it was scored but no quote could be found, so it does not count. '
           'Hover any cell for the quote.</div>',
           '<div style="overflow-x:auto"><table class="tbl claim-chart"><thead><tr><th>Element</th>']
    for d in chart["docs"]:
        name = _e((d.get("pub_num") or d.get("title") or d.get("key") or "")[:28])
        name = f'<a href="{_e(d["url"])}" target="_blank">{name}</a>' if d.get("url") else name
        tag = ' · abstract only' if d.get("abstract_only") else ''
        out.append(f'<th title="{_e(d.get("title", ""))}">{name}<br><span style="font-weight:400;color:var(--text2)">'
                   f'{d.get("n_covered", 0)}/{n} elements{tag}</span></th>')
    out.append("</tr></thead><tbody>")
    for r in chart["rows"]:
        out.append(f'<tr><td>{_e(_clip(r["element"], 140))}</td>')
        for c in r["cells"]:
            score = int(c.get("score") or 0)
            if c.get("covered"):
                bg, txt = "#dcfce7", f'{_SCORE_TEXT.get(score, score)} · {c.get("n_verified", 0)}✓'
            elif score > 0:
                bg, txt = "#fef3c7", f'{_SCORE_TEXT.get(score, score)} · quote not located'
            else:
                bg, txt = "transparent", "–"
            tip = c.get("quote") or c.get("analysis") or ""
            out.append(f'<td style="background:{bg};text-align:center;white-space:nowrap" title="{_e(tip)}">{txt}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")

    return "\n".join(out)


def _paragraphs_html(text: str) -> str:
    return "".join(f"<p>{_e(p.strip())}</p>" for p in (text or "").split("\n\n") if p.strip())


def determination_html(adj: dict | None, chart: dict | None = None, explanation: str = "") -> str:
    if not adj or not adj.get("n_elements"):
        return ""
    bg, fg = _RISK_STYLE.get(adj.get("risk", "related"), _RISK_STYLE["related"])
    n = adj["n_elements"]
    label = adj.get("label")
    # Three lines and nothing else above the fold: the answer, and the two numbers it
    # rests on. Everything with an MPEP section number in it is one click away.
    cov = "".join(f'<div style="font-size:.92em">{_e(l)}</div>' for l in coverage_lines(adj))
    out = ['<div class="sec det-sec" style="border-left:3px solid ' + fg + '"><div class="sec-t sec-t-lg">Prior-Art Determination</div>',
           f'<div class="sec-b det-verdict"><span class="badge" style="background:{bg};color:{fg}">{_e(adj.get("risk", ""))}</span> '
           f'<b>{_e(determination_label(adj))}</b>{cov}</div>']
    how = how_it_was_decided(adj, chart)
    if how:
        out.append('<details class="det-how" style="margin:.6rem 0"><summary style="cursor:pointer;color:var(--text2)">'
                   'How this was decided (for attorneys)</summary>'
                   + "".join(f'<div class="sec-note" style="margin-top:.4rem">{_e(x)}</div>' for x in how)
                   + '</details>')
    out.append(claim_chart_html(chart))
    combos = _combo_lines(chart)
    if label == "103":
        rel = [c for c in combos if c[1]]
        if adj.get("basis") == "combination":
            rel = [c for c in rel if c[0] in adj["combo"]["docs"]] or rel
        else:
            rel = rel[:1]
        out.append('<div class="sec-b det-103" style="margin-top:.6rem"><b>§103 combination relied on:</b><ul>')
        for name, els, url in rel:
            nm = f'<a href="{_e(url)}" target="_blank">{_e(name)}</a>' if url else _e(name)
            out.append(f'<li>{nm} — {len(els)}/{n} elements: {_e("; ".join(_clip(e, 70) for e in els[:6]))}{" …" if len(els) > 6 else ""}</li>')
        out.append("</ul>")
        unc = (chart or {}).get("uncovered") or (adj.get("combo") or {}).get("missing") or []
        if unc:
            out.append(f'<div><b>Not disclosed by any reference (would need a secondary reference or a routine modification):</b> '
                       f'{_e("; ".join(_clip(u, 70) for u in unc[:6]))}{" …" if len(unc) > 6 else ""}</div>')
        if explanation:
            out.append('<div style="margin-top:.5rem"><b>Why this combination reads as obvious — or not</b> '
                       '<span style="color:var(--text2)">(written from the rule output above; the determination itself is not the model\'s):</span>'
                       f'{_paragraphs_html(explanation)}</div>')
        out.append("</div>")
    elif label == "102" and combos:
        name, els, url = combos[0]
        nm = f'<a href="{_e(url)}" target="_blank">{_e(name)}</a>' if url else _e(name)
        out.append(f'<div class="sec-b" style="margin-top:.6rem"><b>Anticipating reference:</b> {nm} discloses all {n} elements with located quotes '
                   f'(see chart). No combination is needed; a §103 reading of the same reference is implied.</div>')
    else:
        unc = (chart or {}).get("uncovered") or []
        per = adj.get("per_doc_coverage") or []
        out.append(f'<div class="sec-b" style="margin-top:.6rem"><b>What is missing:</b> '
                   + (f'{len(unc)} of the {n} elements have no verified disclosure in any document we read — '
                      f'{_e("; ".join(_clip(u, 70) for u in unc[:8]))}{" …" if len(unc) > 8 else ""}. ' if unc else "")
                   + 'This is a statement about the documents evaluated here, not a prediction about examination.</div>')
    out.append("</div>")
    return "\n".join(out)


def determination_md(adj: dict | None, chart: dict | None = None, explanation: str = "") -> list[str]:
    if not adj or not adj.get("n_elements"):
        return []
    n = adj["n_elements"]
    lines = ["## Prior-Art Determination", "",
             f"**{adj.get('risk', '')}** — {determination_label(adj)}.", ""]
    cov = coverage_lines(adj)
    if cov:
        lines += [f"- {l}" for l in cov] + [""]
    how = how_it_was_decided(adj, chart)
    if how:
        lines += ["### How this was decided (for attorneys)", ""]
        lines += [x + "\n" for x in how]
    if chart and chart.get("rows") and chart.get("docs"):
        lines += ["How to read this table: **Present** / **Partial** is the evaluator's score for that element, "
                  "and **n✓** is how many verbatim quotes were located in that document. "
                  '**"quote not located"** means it was scored but no quote could be found, so it does not count.', ""]
        head = "| Element | " + " | ".join(
            f"{(d.get('pub_num') or d.get('title') or d.get('key') or '')[:24]} ({d.get('n_covered', 0)}/{n}"
            + (", abstract only)" if d.get("abstract_only") else ")") for d in chart["docs"]) + " |"
        lines += [head, "|" + "---|" * (len(chart["docs"]) + 1)]
        for r in chart["rows"]:
            cells = []
            for c in r["cells"]:
                score = int(c.get("score") or 0)
                if c.get("covered"):
                    cells.append(f"{_SCORE_TEXT.get(score, score)} {c.get('n_verified', 0)}✓")
                elif score > 0:
                    cells.append(f"{_SCORE_TEXT.get(score, score)} (quote not located)")
                else:
                    cells.append("–")
            lines.append(f"| {_clip(r['element'], 90)} | " + " | ".join(cells) + " |")
        lines.append("")
    combos = _combo_lines(chart)
    if adj.get("label") == "103":
        rel = [c for c in combos if c[1]]
        if adj.get("basis") == "combination":
            rel = [c for c in rel if c[0] in adj["combo"]["docs"]] or rel
        else:
            rel = rel[:1]
        lines.append("**§103 combination relied on:**")
        for name, els, _ in rel:
            lines.append(f"- {name} — {len(els)}/{n} elements: {'; '.join(_clip(e, 70) for e in els[:6])}{' …' if len(els) > 6 else ''}")
        unc = (chart or {}).get("uncovered") or (adj.get("combo") or {}).get("missing") or []
        if unc:
            lines.append(f"- Not disclosed by any reference: {'; '.join(_clip(u, 70) for u in unc[:6])}{' …' if len(unc) > 6 else ''}")
        if explanation:
            lines += ["", "**Why this combination reads as obvious — or not** (written from the rule output; the determination is not the model's):", "",
                      explanation.strip()]
        lines.append("")
    elif adj.get("label") == "102" and combos:
        lines += [f"**Anticipating reference:** {combos[0][0]} discloses all {n} elements with located quotes.", ""]
    else:
        unc = (chart or {}).get("uncovered") or []
        lines += ["**What is missing:** "
                  + (f"{len(unc)} of the {n} elements have no verified disclosure in any document we read — "
                     f"{'; '.join(_clip(u, 70) for u in unc[:8])}{' …' if len(unc) > 8 else ''}. " if unc else "")
                  + "This is a statement about the documents evaluated here, not a prediction about examination.", ""]
    return lines


# kept for callers that still pass adjudication to inject_*; the report itself now renders the determination at the top
adjudication_html = determination_html
adjudication_md = determination_md


# ── 6. run cost / resource accounting ──
#
# What the run actually spent, per phase. The dollar figure is an estimate from
# list prices (patent_analyzer.metering.PRICES), not a bill, and the section
# says so rather than letting a reader treat it as one.

_PHASE_TITLE = {"idca": "1 · Read & classify", "extract": "2 · Extract elements",
                "search": "3 · Prior-art recall", "evaluate": "4 · Evaluate",
                "draft": "4b · Draft claims", "report": "5 · Report"}


def _fmt_secs(s) -> str:
    s = float(s or 0)
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.1f}m"


def _tok(n) -> str:
    n = int(n or 0)
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def cost_html(cost: dict | None) -> str:
    ph = (cost or {}).get("phases") or {}
    if not ph:
        return ""
    t = cost.get("totals") or {}
    out = ['<div class="sec"><div class="sec-t">Run Cost</div>',
           f'<div class="sec-note">{_e(cost.get("note") or "")}</div>',
           '<table class="tbl"><thead><tr><th>Phase</th><th>Time</th><th>LLM calls</th><th>In</th>'
           '<th>Out+thought</th><th>External calls</th><th>Est. cost</th></tr></thead><tbody>']
    for name, m in ph.items():
        llm = m.get("llm") or {}
        pin = sum(d.get("prompt_tokens", 0) for d in llm.values())
        pout = sum(d.get("output_tokens", 0) + d.get("thought_tokens", 0) for d in llm.values())
        out.append(f'<tr><td>{_e(_PHASE_TITLE.get(name, name))}</td><td>{_fmt_secs(m.get("seconds"))}</td>'
                   f'<td>{m.get("llm_calls", 0)}</td><td>{_tok(pin)}</td><td>{_tok(pout)}</td>'
                   f'<td>{m.get("external_calls", 0)}</td><td>${m.get("cost_usd", 0):.4f}</td></tr>')
    tl = t.get("llm") or {}
    tin = sum(d.get("prompt_tokens", 0) for d in tl.values())
    tout = sum(d.get("output_tokens", 0) + d.get("thought_tokens", 0) for d in tl.values())
    out.append(f'<tr style="font-weight:600;border-top:2px solid #cbd5e1"><td>Total</td>'
               f'<td>{_fmt_secs(t.get("seconds"))}</td><td>{t.get("llm_calls", 0)}</td>'
               f'<td>{_tok(tin)}</td><td>{_tok(tout)}</td><td>{t.get("external_calls", 0)}</td>'
               f'<td>${t.get("cost_usd", 0):.4f}</td></tr>')
    out.append("</tbody></table>")
    if tl:
        out.append('<div class="sec-note">By model: ' + ", ".join(
            f'<code>{_e(m)}</code> {d.get("calls", 0)} calls, ${d.get("cost_usd", 0):.4f}'
            + (f' ({d["errors_429"]}&times; 429)' if d.get("errors_429") else "")
            for m, d in tl.items()) + "</div>")
    ext = t.get("external") or {}
    if ext:
        out.append('<div class="sec-note">External: ' +
                   ", ".join(f'<code>{_e(k)}</code> {v}' for k, v in sorted(ext.items())) + "</div>")
    bqt = t.get("bigquery") or {}
    if bqt.get("queries"):
        out.append(f'<div class="sec-note">BigQuery: {bqt["queries"]} queries, '
                   f'{bqt.get("gib_billed", 0)} GiB billed, ${bqt.get("cost_usd", 0):.4f} '
                   f'(at ${cost.get("bigquery_usd_per_tib", 0)}/TiB)</div>')
    out.append("</div>")
    return "\n".join(out)


def cost_md(cost: dict | None) -> list[str]:
    ph = (cost or {}).get("phases") or {}
    if not ph:
        return []
    t = cost.get("totals") or {}
    lines = ["## Run Cost", "", cost.get("note") or "", "",
             "| Phase | Time | LLM calls | In | Out+thought | External calls | Est. cost |",
             "|---|---|---|---|---|---|---|"]
    for name, m in ph.items():
        llm = m.get("llm") or {}
        pin = sum(d.get("prompt_tokens", 0) for d in llm.values())
        pout = sum(d.get("output_tokens", 0) + d.get("thought_tokens", 0) for d in llm.values())
        lines.append(f'| {_PHASE_TITLE.get(name, name)} | {_fmt_secs(m.get("seconds"))} | {m.get("llm_calls", 0)} | '
                     f'{_tok(pin)} | {_tok(pout)} | {m.get("external_calls", 0)} | ${m.get("cost_usd", 0):.4f} |')
    lines.append(f'| **Total** | {_fmt_secs(t.get("seconds"))} | {t.get("llm_calls", 0)} | | | '
                 f'{t.get("external_calls", 0)} | **${t.get("cost_usd", 0):.4f}** |')
    lines.append("")
    return lines


def inject_html(report_html: str, extraction: dict | None, search_stats: dict | None,
                scoring_report: list[dict] | None, checklist: list[dict] | None,
                adjudication: dict | None = None, draft: dict | None = None,
                cost: dict | None = None, read_gap: dict | None = None) -> str:
    anchor = '<div class="sec-t">Invention Summary</div>'
    adj_block = determination_html(adjudication) if adjudication and "Prior-Art Determination" not in report_html else ""
    # read_gap first: how much of the delivered prior art was actually read
    # qualifies every number under it.
    from .fulltext_gap import uploaded_fulltext_html
    block = "\n".join(x for x in (read_gap_html(read_gap), extraction_html(extraction),
                                  channel_health_html(search_stats),
                                  evidence_coverage_html(scoring_report),
                                  fulltext_tier_html(search_stats),
                                  uploaded_fulltext_html(scoring_report), loop_html(search_stats),
                                  quote_matrix_html(scoring_report, checklist, adjudication=adjudication), adj_block,
                                  draft_html(draft, extraction), cost_html(cost)) if x)
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
              adjudication: dict | None = None, draft: dict | None = None,
              cost: dict | None = None, read_gap: dict | None = None) -> str:
    adj_lines = determination_md(adjudication) if adjudication and "## Prior-Art Determination" not in report_md else []
    from .fulltext_gap import uploaded_fulltext_md
    lines = (read_gap_md(read_gap) + extraction_md(extraction) + channel_health_md(search_stats)
             + evidence_coverage_md(scoring_report) + fulltext_tier_md(search_stats)
             + uploaded_fulltext_md(scoring_report)
             + loop_md(search_stats)
             + quote_matrix_md(scoring_report, checklist, adjudication=adjudication) + adj_lines
             + draft_md(draft, extraction) + cost_md(cost))
    if not lines:
        return report_md
    block = "\n".join(lines)
    for marker in ("## Evaluation Criteria", "## Novelty Assessment"):
        if marker in report_md:
            return report_md.replace(marker, block + "\n" + marker, 1)
    return report_md + "\n" + block
