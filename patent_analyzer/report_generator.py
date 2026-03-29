#!/usr/bin/env python3
"""
Generate a self-contained static HTML report from patent analysis results.

Design principles:
- Default view: ONLY docs with overlap (possible matches)
- Cards grouped by hit count, most hits first
- Hover → show matched points as short tags
- Click → expand full evaluation detail
- Clean, readable, lots of whitespace
"""

import argparse
import json
import html as html_mod
import re
from datetime import datetime, timezone


def esc(text: str) -> str:
    return html_mod.escape(str(text)) if text else ""


def score_color(score: float) -> str:
    if score > 0.7:
        return "#dc2626"
    if score > 0.4:
        return "#ea580c"
    if score > 0.2:
        return "#d97706"
    if score > 0:
        return "#2563eb"
    return "#9ca3af"


def format_summary(text: str) -> str:
    """Format summary with numbered list + keyword highlighting."""
    # Technical keywords to highlight
    KW = [
        "dense descriptor", "dense object", "contrastive loss", "self-supervised",
        "pixel correspondence", "3D reconstruction", "TSDF", "RGBD", "RGB-D",
        "object mask", "change detection", "domain randomization", "hard negative",
        "cross-object", "multi-object", "descriptor space", "robotic grasping",
        "visual descriptor", "nearest neighbor", "ResNet", "FCN",
        "prior art", "novelty", "anticipat", "obvious", "102", "103",
        "manipulation", "deformed", "class generalization", "instance specific",
    ]

    def highlight(t: str) -> str:
        """Add <mark> tags around keywords."""
        result = esc(t)
        for kw in KW:
            pattern = re.compile(re.escape(esc(kw)), re.IGNORECASE)
            result = pattern.sub(lambda m: f'<mark>{m.group()}</mark>', result)
        return result

    # Check for numbered items like (1)...(2)...
    parts = re.split(r'\s*\((\d+)\)\s*', text)
    if len(parts) >= 3:
        preamble = parts[0].strip().rstrip(':;,')
        items_html = '<ol class="sum-list">'
        for i in range(1, len(parts) - 1, 2):
            item_text = parts[i + 1].strip().rstrip(';,')
            if item_text:
                items_html += f'<li>{highlight(item_text)}</li>'
        items_html += '</ol>'
        if preamble:
            return f'<p class="sum-preamble">{highlight(preamble)}:</p>{items_html}'
        return items_html

    # Check for markdown bold **text**
    text_clean = text.replace('**', '')

    # Split into paragraphs on double newline or sentence boundaries
    paragraphs = re.split(r'\n\n+', text_clean)
    if len(paragraphs) <= 1:
        paragraphs = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text_clean)

    html = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        html += f'<p class="sum-para">{highlight(p)}</p>'
    return html if html else f'<p class="sum-para">{highlight(text)}</p>'


def format_eval_summary(text: str) -> str:
    """Format evaluation summary with keyword highlighting and paragraph breaks."""
    KW = [
        "prior art", "novelty", "novel", "anticipated", "obvious", "102", "103",
        "overlap", "coverage", "unique", "distinguishing", "closest",
        "dense descriptor", "contrastive", "self-supervised", "robotic",
        "3D reconstruction", "object mask", "domain randomization",
    ]

    def hl(t: str) -> str:
        result = esc(t)
        for kw in KW:
            pattern = re.compile(re.escape(esc(kw)), re.IGNORECASE)
            result = pattern.sub(lambda m: f'<mark>{m.group()}</mark>', result)
        return result

    text_clean = text.replace('**', '')
    paragraphs = re.split(r'\n\n+', text_clean)
    if len(paragraphs) <= 1:
        paragraphs = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text_clean)

    html = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        html += f'<p class="eval-para">{hl(p)}</p>'
    return html if html else f'<p class="eval-para">{hl(text)}</p>'


def shorten_checklist(item: str, max_len: int = 60) -> str:
    """Shorten a checklist item to a readable tag."""
    # Remove common prefixes
    item = re.sub(r'^(The (system|method|network|training|process|pipeline|entire|invention)\s+(uses?|includes?|performs?|enables?|is|can|from)\s+)', '', item, flags=re.IGNORECASE)
    item = re.sub(r'^(A |An )', '', item)
    # Truncate
    if len(item) > max_len:
        item = item[:max_len].rsplit(' ', 1)[0] + '...'
    return item


def generate_html(data: dict) -> str:
    phase1 = data.get("phase1", {})
    phase2 = data.get("phase2", {})
    search = data.get("search", {})
    evaluation = data.get("evaluation", {})

    doc_mode = phase1.get("doc_mode", "unknown")
    summary_text = phase1.get("summary", "")
    invention_type = phase1.get("invention_type", "Unknown")

    checklist = phase2.get("checklist", [])
    search_groups = search.get("groups", [])
    total_patents = search.get("summary", {}).get("total_patents", 0)
    total_papers = search.get("summary", {}).get("total_papers", 0)

    scoring_report = evaluation.get("scoring_report", [])
    overall_summary = evaluation.get("summary", "")

    generated_at = data.get("generated_at", datetime.now(timezone.utc).isoformat())

    # Split into groups
    hits_docs = [m for m in scoring_report if (m.get("similarity_score", 0) or 0) > 0]
    no_hits_docs = [m for m in scoring_report if (m.get("similarity_score", 0) or 0) == 0]

    def build_card(m: dict, idx: int) -> str:
        title = esc(m.get("title", "Unknown"))
        mid = esc(m.get("id", ""))
        mtype = m.get("manuscript_type", "Document")
        url = m.get("url", "")
        # Prefer real abstract over search snippet
        abstract = m.get("abstract", "")
        raw_snippet = m.get("snippet", "")
        display_text = esc(abstract if abstract else raw_snippet)
        snippet_short = display_text[:200] + "..." if len(display_text) > 200 else display_text
        snippet = display_text

        evals = m.get("similarity_categories", m.get("evaluations", {}))
        matched = []
        unmatched = []
        md_lines = []

        for req, ev in (evals or {}).items():
            is_match = ev.get("match", False) if isinstance(ev, dict) else False
            analysis = esc(ev.get("analysis", "")) if isinstance(ev, dict) else ""
            short = esc(shorten_checklist(req))
            if is_match:
                matched.append((short, req, analysis))
            else:
                unmatched.append((short, req, analysis))
            md_lines.append(f"- [{'MATCH' if is_match else 'NO MATCH'}] **{esc(req)}**: {analysis}")

        hit_count = len(matched)
        total = len(evals) if evals else len(checklist)
        score_num = hit_count / total if total > 0 else 0

        # Hover zone: full checklist lines, one per row
        hover_html = ""
        if matched:
            hover_html = '<div class="hover-hits">'
            for short, full, analysis in matched:
                hover_html += f'<div class="hv-item"><span class="hv-icon">&#x2705;</span> {esc(full)}</div>'
            hover_html += '</div>'

        # Expanded detail
        detail_html = '<div class="detail-panel">'
        if matched:
            detail_html += '<div class="detail-group"><div class="detail-label">Matched Points</div>'
            for short, full, analysis in matched:
                detail_html += f'<div class="d-item d-hit"><span class="d-icon">&#x2705;</span><div><div class="d-req">{esc(full)}</div>'
                if analysis:
                    detail_html += f'<div class="d-analysis">{analysis}</div>'
                detail_html += '</div></div>'
            detail_html += '</div>'
        if unmatched:
            detail_html += '<div class="detail-group"><div class="detail-label">Unmatched Points</div>'
            for short, full, analysis in unmatched:
                detail_html += f'<div class="d-item d-miss"><span class="d-icon">&#x2501;</span><div><div class="d-req">{esc(full)}</div>'
                if analysis:
                    detail_html += f'<div class="d-analysis">{analysis}</div>'
                detail_html += '</div></div>'
            detail_html += '</div>'
        detail_html += '</div>'

        # Markdown
        md = f"# {title}\n- **ID**: {mid}\n- **Type**: {mtype}\n- **Matched**: {hit_count}/{total}\n"
        if url:
            md += f"- **URL**: {url}\n"
        md += f"\n## Evaluation\n" + "\n".join(md_lines)
        md_esc = esc(md).replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

        badge_type = "Patent" if mtype == "Patent" else "Paper"
        card_cls = "card card-hit" if hit_count > 0 else "card card-none"

        # Patent-specific metadata
        filing = esc(m.get("filing_date", ""))
        grant = esc(m.get("grant_date", ""))
        inventor = esc(m.get("inventor", ""))
        assignee = esc(m.get("assignee", ""))
        authors_str = esc(m.get("authors", ""))
        year = m.get("year", "")
        patent_link = m.get("patent_link", "")

        # Build subtitle line
        if badge_type == "Patent":
            sub_parts = [mid] if mid and len(mid) < 25 else []
            if assignee: sub_parts.append(assignee)
            if filing: sub_parts.append(f"Filed {filing}")
            if grant: sub_parts.append(f"Granted {grant}")
            subtitle = " &middot; ".join(sub_parts) if sub_parts else "Patent"
        else:
            sub_parts = []
            if authors_str: sub_parts.append(authors_str)
            if year: sub_parts.append(str(year))
            subtitle = " &middot; ".join(sub_parts) if sub_parts else "Paper"

        return f'''
<div class="{card_cls}" data-hits="{hit_count}" data-md="{md_esc}">
  <div class="card-top" onclick="toggle(this.parentElement)">
    <div class="card-left">
      <span class="type-dot {'dot-patent' if badge_type == 'Patent' else 'dot-paper'}"></span>
      <div>
        <div class="card-title">{title}</div>
        <div class="card-id">{subtitle}</div>
      </div>
    </div>
    <div class="card-right">
      {f'<span class="hit-pct" style="color:{score_color(score_num)}">{score_num:.0%}</span>' if hit_count > 0 else '<span class="no-badge">&mdash;</span>'}
      {f'<a class="pdf-link" href="{esc(patent_link)}" target="_blank" onclick="event.stopPropagation()">Patent</a>' if patent_link and badge_type == 'Patent' else ''}
      {f'<a class="pdf-link" href="{esc(url)}" target="_blank" onclick="event.stopPropagation()">PDF</a>' if url else ''}
      <button class="md-btn" onclick="event.stopPropagation();showMd(this.closest(\'.card\'))">MD</button>
    </div>
  </div>
  {hover_html}
  {f'<div class="card-snippet"><span class="snip-short">{snippet_short}</span><span class="snip-full">{snippet}</span></div>' if snippet else ''}
  {detail_html}
</div>'''

    # Build cards
    hits_html = "".join(build_card(m, i) for i, m in enumerate(hits_docs))
    nohits_html = "".join(build_card(m, i) for i, m in enumerate(no_hits_docs))

    # Search groups
    sg_html = ""
    for ar in search_groups:
        gid = esc(ar.get("group_id", ""))
        label = esc(ar.get("label", ""))
        pq = esc(ar.get("patent_query", ""))
        pm = ar.get("patent_matches_found", 0)
        sq = esc(ar.get("paper_query", ""))
        sm = ar.get("paper_matches_found", 0)
        sg_html += f'''<div class="sg"><span class="sg-id">{gid}</span> {label}
          {f'<div class="sg-q">Patent: {pq} <span>({pm})</span></div>' if pq else ''}
          {f'<div class="sg-q">Scholar: {sq} <span>({sm})</span></div>' if sq else ''}
        </div>'''

    return f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prior Art Search Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#f7f8fa;--card:#fff;--border:#e2e6ec;--text:#111318;--text2:#4b5563;
  --accent:#2563eb;--hit:#16a34a;--miss:#9ca3af;--patent:#8b5cf6;--paper:#0ea5e9;
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,sans-serif;
}}
body{{background:var(--bg);color:var(--text);padding:2rem 1rem;line-height:1.55}}
.wrap{{max-width:960px;margin:0 auto}}

/* Header */
.hdr{{text-align:center;padding:1.5rem 0 2rem;border-bottom:1px solid var(--border);margin-bottom:2rem}}
.hdr h1{{font-size:1.4rem;font-weight:700;color:var(--text)}}
.hdr .sub{{font-size:0.82rem;color:var(--text2);margin:0.3rem 0 1rem}}
.pills{{display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap}}
.pill{{font-size:0.75rem;padding:0.25rem 0.7rem;border-radius:99px;background:#eef1f5;color:var(--text2)}}
.pill b{{color:var(--text)}}
.hdr-actions{{margin-top:0.75rem}}

/* Sections */
.sec{{margin:1.5rem 0;padding:1rem 1.25rem;border:1px solid var(--border);border-radius:10px;background:var(--card)}}
.sec-t{{font-size:0.95rem;font-weight:600;margin-bottom:0.5rem;color:var(--text)}}
.sec-t-lg{{font-size:1.15rem;font-weight:700;letter-spacing:-0.01em}}
.sec-count{{font-size:0.78rem;font-weight:400;color:var(--text2);margin-left:0.4rem}}
.sec-note{{font-size:0.8rem;color:var(--text2);margin-bottom:0.75rem;line-height:1.5;font-style:italic}}
.sec-b{{font-size:0.88rem;color:var(--text2);line-height:1.7}}
.sec-b p{{margin-bottom:0.5rem}}

/* Invention summary */
.sum-preamble{{font-size:0.92rem;font-weight:600;color:var(--text);margin-bottom:0.4rem}}
.sum-list{{padding-left:1.3rem;margin:0.4rem 0}}
.sum-list li{{margin-bottom:0.5rem;line-height:1.65;color:var(--text);font-size:0.9rem}}
.sum-para{{margin-bottom:0.6rem;line-height:1.7;color:var(--text);font-size:0.9rem}}

/* Evaluation summary */
.eval-sec{{border-left:3px solid var(--accent)}}
.eval-para{{margin-bottom:0.6rem;line-height:1.75;color:var(--text);font-size:0.9rem}}

/* Keyword highlight */
mark{{background:#fef3c7;color:#92400e;padding:0.05rem 0.2rem;border-radius:3px;font-weight:500}}

/* Checklist section */
.checklist-sec{{background:#fafbfc}}
.cl-list{{padding-left:1.5rem;margin:0;counter-reset:cl}}
.cl-item{{margin-bottom:0.45rem;line-height:1.6;font-size:0.85rem;color:var(--text);padding:0.3rem 0.5rem;border-radius:6px}}
.cl-item:nth-child(odd){{background:#f3f4f6}}
.cl-item::marker{{color:var(--accent);font-weight:700}}
.tog{{cursor:pointer;user-select:none}}
.tog::before{{content:'\\25B8';margin-right:0.4rem;font-size:0.7rem;display:inline-block;transition:transform .15s}}
.tog.open::before{{transform:rotate(90deg)}}
.tog-body{{display:none;margin-top:0.75rem}}
.tog-body.open{{display:block}}

/* Search groups */
.sg{{padding:0.5rem 0;border-bottom:1px solid var(--border);font-size:0.82rem}}
.sg:last-child{{border-bottom:none}}
.sg-id{{font-weight:600;color:var(--accent)}}
.sg-q{{font-family:"SF Mono",Monaco,monospace;font-size:0.75rem;color:var(--text2);margin:0.2rem 0 0 1rem;word-break:break-all}}
.sg-q span{{color:var(--accent)}}

/* Filter bar */
.bar{{display:flex;gap:0.5rem;margin:1.5rem 0 1rem;align-items:center}}
.bar-title{{font-size:1rem;font-weight:600;flex:1}}
.bar-actions{{display:flex;gap:0.5rem;align-items:center}}
.fbtn{{font-size:0.78rem;padding:0.3rem 0.75rem;border-radius:7px;border:1px solid var(--border);background:var(--card);color:var(--text2);cursor:pointer}}
.fbtn.on{{background:var(--accent);border-color:var(--accent);color:#fff}}
.fbtn-accent{{border-color:var(--accent);color:var(--accent)}}
.fbtn-accent:hover{{background:var(--accent);color:#fff}}

/* Dropdown */
.dropdown{{position:relative}}
.dropdown-menu{{display:none;position:absolute;right:0;top:calc(100% + 4px);background:var(--card);border:1px solid var(--border);border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.1);z-index:50;min-width:220px;padding:0.3rem 0;overflow:hidden}}
.dropdown.open .dropdown-menu{{display:block}}
.dropdown-menu button{{display:block;width:100%;text-align:left;padding:0.45rem 0.9rem;font-size:0.8rem;border:none;background:none;color:var(--text);cursor:pointer}}
.dropdown-menu button:hover{{background:#f0f3f7}}
.dropdown-menu hr{{border:none;border-top:1px solid var(--border);margin:0.25rem 0}}
.bar-note{{font-size:0.78rem;color:var(--text2);margin-bottom:1rem}}

/* Cards */
.card{{border:1px solid var(--border);border-radius:10px;background:var(--card);margin-bottom:0.6rem;overflow:hidden;transition:box-shadow .15s}}
.card:hover{{box-shadow:0 2px 8px rgba(0,0,0,0.06)}}
.card-hit{{border-left:3px solid var(--accent)}}
.card-none{{opacity:0.7}}

.card-top{{display:flex;align-items:center;justify-content:space-between;padding:0.75rem 1rem;cursor:pointer;gap:0.75rem}}
.card-left{{display:flex;align-items:center;gap:0.6rem;flex:1;min-width:0}}
.type-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.dot-patent{{background:var(--patent)}}
.dot-paper{{background:var(--paper)}}
.card-title{{font-size:0.88rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card-id{{font-size:0.72rem;color:var(--text2);font-family:"SF Mono",Monaco,monospace}}
.card-right{{display:flex;align-items:center;gap:0.5rem;flex-shrink:0}}
.hit-pct{{font-size:0.88rem;font-weight:700}}
.no-badge{{color:var(--miss);font-size:0.78rem}}
.pdf-link{{font-size:0.72rem;color:var(--accent);text-decoration:none}}
.md-btn{{font-size:0.68rem;color:var(--accent);background:none;border:1px solid var(--accent);border-radius:5px;padding:0.1rem 0.4rem;cursor:pointer;opacity:0;transition:opacity .15s}}
.card:hover .md-btn{{opacity:1}}
.md-btn:hover{{background:var(--accent);color:#fff}}

/* Hover checklist — full lines, 300ms delay */
.hover-hits{{padding:0 1rem 0.5rem;opacity:0;max-height:0;overflow:hidden;transition:opacity .2s ease .3s,max-height .25s ease .3s}}
.card:hover .hover-hits{{opacity:1;max-height:500px}}
.card.open .hover-hits{{opacity:0;max-height:0;transition:none}}
.hv-item{{font-size:0.82rem;color:#166534;padding:0.2rem 0;display:flex;align-items:flex-start;gap:0.3rem;line-height:1.5}}
.hv-icon{{flex-shrink:0}}

.card-snippet{{font-size:0.82rem;color:var(--text2);padding:0.3rem 1rem 0.6rem;line-height:1.55;border-top:1px solid var(--border)}}
.snip-full{{display:none}}
.card:hover .snip-short{{display:none}}
.card:hover .snip-full{{display:inline}}
.card.open .snip-short{{display:none}}
.card.open .snip-full{{display:inline}}

/* Detail panel (click to expand) */
.detail-panel{{display:none;padding:0.5rem 1rem 1rem;border-top:1px solid var(--border)}}
.card.open .detail-panel{{display:block}}
.detail-group{{margin-bottom:0.75rem}}
.detail-label{{font-size:0.75rem;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:0.03em;margin-bottom:0.35rem}}
.d-item{{display:flex;gap:0.4rem;padding:0.3rem 0;align-items:flex-start}}
.d-icon{{flex-shrink:0;font-size:0.8rem}}
.d-hit .d-icon{{color:var(--hit)}}
.d-miss .d-icon{{color:var(--miss)}}
.d-req{{font-size:0.82rem;color:var(--text)}}
.d-analysis{{font-size:0.75rem;color:var(--text2);margin-top:0.1rem}}

/* Modal */
.overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:999;justify-content:center;align-items:center;padding:2rem}}
.overlay.on{{display:flex}}
.modal{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.5rem;max-width:760px;width:100%;max-height:80vh;overflow-y:auto;position:relative}}
.modal-x{{position:absolute;top:0.8rem;right:1rem;background:none;border:none;font-size:1.3rem;color:var(--text2);cursor:pointer}}
.modal-acts{{display:flex;gap:0.5rem;margin-bottom:0.75rem}}
.modal-btn{{padding:0.35rem 0.8rem;border-radius:7px;border:1px solid var(--border);background:#f3f4f6;color:var(--text);cursor:pointer;font-size:0.82rem}}
.modal-btn:hover{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.modal-pre{{white-space:pre-wrap;font-family:"SF Mono",Monaco,monospace;font-size:0.8rem;line-height:1.6;color:var(--text2);padding:1rem;background:#f9fafb;border-radius:8px;border:1px solid var(--border)}}
</style>
</head><body>
<div class="wrap">

<div class="hdr">
  <h1>Prior Art Search Report</h1>
  <div class="sub">Possible matches for professional review &mdash; not a legal determination</div>
  <div class="pills">
    <span class="pill"><b>{esc(doc_mode.title())}</b> &middot; {esc(invention_type)}</span>
    <span class="pill">Searched <b>{total_patents}</b> patents + <b>{total_papers}</b> papers</span>
    <span class="pill"><b>{len(hits_docs)}</b> possible matches</span>
  </div>
</div>

<div class="sec">
  <div class="sec-t">Invention Summary</div>
  <div class="sec-b">{format_summary(summary_text)}</div>
</div>

{f'<div class="sec eval-sec"><div class="sec-t sec-t-lg">Novelty Assessment</div><div class="sec-b">{format_eval_summary(overall_summary)}</div></div>' if overall_summary else ''}

<div class="sec checklist-sec">
  <div class="sec-t sec-t-lg">Evaluation Checklist <span class="sec-count">{len(checklist)} items</span></div>
  <div class="sec-note">Each item is an atomic, testable requirement derived from the invention disclosure. Prior art is evaluated against every item.</div>
  <ol class="cl-list">
    {"".join(f'<li class="cl-item">{esc(item)}</li>' for item in checklist)}
  </ol>
</div>

{f'''<div class="sec">
  <div class="sec-t tog" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">Search Groups ({len(search_groups)})</div>
  <div class="tog-body">{sg_html}</div>
</div>''' if sg_html else ''}

<div class="bar">
  <div class="bar-title">Possible Matches</div>
  <div class="bar-actions">
    <button class="fbtn" onclick="showAll(this)">All ({len(scoring_report)})</button>
    <button class="fbtn on" onclick="showHits(this)">With Overlap ({len(hits_docs)})</button>
    <button class="fbtn" onclick="showNone(this)">No Overlap ({len(no_hits_docs)})</button>
  </div>
</div>
<div class="bar-note">Hover for matched points &middot; Click to expand full evaluation</div>

<div id="hits">{hits_html}</div>
<div id="nohits" style="display:none">{nohits_html}</div>

</div>

<div class="overlay" id="ov" onclick="if(event.target===this)closeMd()">
<div class="modal">
  <button class="modal-x" onclick="closeMd()">&times;</button>
  <div class="modal-acts">
    <button class="modal-btn" onclick="copyMd()">Copy</button>
    <button class="modal-btn" onclick="dlMd()">Download .md</button>
  </div>
  <pre class="modal-pre" id="mdC"></pre>
</div>
</div>

<script>
let md='';
function toggle(c){{c.classList.toggle('open')}}
function showMd(c){{
  const r=c.getAttribute('data-md');
  md=r.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#x27;/g,"'");
  document.getElementById('mdC').textContent=md;
  document.getElementById('ov').classList.add('on');
}}
function closeMd(){{document.getElementById('ov').classList.remove('on')}}
function copyMd(){{navigator.clipboard.writeText(md).then(()=>{{event.target.textContent='Copied!';setTimeout(()=>event.target.textContent='Copy',1200)}})}}
function dlMd(){{const b=new Blob([md],{{type:'text/markdown'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='match.md';a.click()}}

function setFilter(btn){{document.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('on'));btn.classList.add('on')}}
function showAll(b){{setFilter(b);document.getElementById('hits').style.display='';document.getElementById('nohits').style.display=''}}
function showHits(b){{setFilter(b);document.getElementById('hits').style.display='';document.getElementById('nohits').style.display='none'}}
function showNone(b){{setFilter(b);document.getElementById('hits').style.display='none';document.getElementById('nohits').style.display=''}}

// Close dropdown on outside click
document.addEventListener('click',e=>{{if(!e.target.closest('.dropdown'))document.querySelectorAll('.dropdown.open').forEach(d=>d.classList.remove('open'))}});

function buildReportMd(mode){{
  const cards=document.querySelectorAll(mode==='hits'?'#hits .card':mode==='none'?'#nohits .card':'#hits .card, #nohits .card');
  let out='# Prior Art Search Report\\n\\n';
  out+='| # | Title | Type | Match |\\n|---|-------|------|-------|\\n';
  let idx=1;
  cards.forEach(c=>{{
    const t=c.querySelector('.card-title')?.textContent||'';
    const tp=c.querySelector('.card-id')?.textContent||'';
    const pct=c.querySelector('.hit-pct')?.textContent||c.querySelector('.no-badge')?.textContent||'0';
    out+=`| ${{idx++}} | ${{t}} | ${{tp}} | ${{pct}} |\\n`;
  }});
  out+='\\n---\\n\\n';
  idx=1;
  cards.forEach(c=>{{
    const raw=c.getAttribute('data-md')||'';
    const decoded=raw.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#x27;/g,"'");
    if(decoded)out+=decoded+'\\n\\n---\\n\\n';
  }});
  return out;
}}
function copyReportMd(mode){{
  const text=buildReportMd(mode);
  navigator.clipboard.writeText(text).then(()=>{{
    const btn=event.target;btn.textContent='Copied!';setTimeout(()=>btn.textContent=btn.dataset.label||btn.textContent,1200);
  }});
  document.querySelectorAll('.dropdown.open').forEach(d=>d.classList.remove('open'));
}}
function dlReportMd(mode){{
  const text=buildReportMd(mode);
  const b=new Blob([text],{{type:'text/markdown'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download=`prior-art-${{mode}}.md`;a.click();
  document.querySelectorAll('.dropdown.open').forEach(d=>d.classList.remove('open'));
}}

// Full report MD — copies EVERYTHING (summary + checklist + all matches)
function buildFullMd(mode){{
  let out='# Prior Art Search Report\\n\\n';
  // Invention summary
  const sumSec=document.querySelector('.sec .sec-b');
  if(sumSec)out+='## Invention Summary\\n\\n'+sumSec.textContent.trim()+'\\n\\n';
  // Eval summary
  const evalSec=document.querySelector('.eval-sec .sec-b');
  if(evalSec)out+='## Novelty Assessment\\n\\n'+evalSec.textContent.trim()+'\\n\\n';
  // Checklist
  const clItems=document.querySelectorAll('.cl-item');
  if(clItems.length){{
    out+='## Evaluation Checklist\\n\\n';
    clItems.forEach((li,i)=>out+=`${{i+1}}. ${{li.textContent.trim()}}\\n`);
    out+='\\n';
  }}
  // Matches table
  const cards=mode==='hits'?document.querySelectorAll('#hits .card'):document.querySelectorAll('#hits .card, #nohits .card');
  out+='## Matches\\n\\n| # | Title | Score |\\n|---|-------|-------|\\n';
  let idx=1;
  cards.forEach(c=>{{
    const t=c.querySelector('.card-title')?.textContent||'';
    const pct=c.querySelector('.hit-pct')?.textContent||'—';
    out+=`| ${{idx++}} | ${{t}} | ${{pct}} |\\n`;
  }});
  out+='\\n---\\n\\n';
  // Per-doc details
  cards.forEach(c=>{{
    const raw=c.getAttribute('data-md')||'';
    const decoded=raw.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#x27;/g,"'");
    if(decoded)out+=decoded+'\\n\\n---\\n\\n';
  }});
  return out;
}}
function copyFullMd(mode){{
  navigator.clipboard.writeText(buildFullMd(mode||'all')).then(()=>{{
    event.target.textContent='Copied!';setTimeout(()=>event.target.textContent=event.target.dataset.orig||event.target.textContent,1500);
  }});
  document.querySelectorAll('.dropdown.open').forEach(d=>d.classList.remove('open'));
}}
function dlFullMd(mode){{
  const b=new Blob([buildFullMd(mode||'all')],{{type:'text/markdown'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='patent-analysis-report.md';a.click();
  document.querySelectorAll('.dropdown.open').forEach(d=>d.classList.remove('open'));
}}

document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeMd()}});
</script>
</body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(generate_html(data))
    print(f"Report generated: {args.output}")


if __name__ == "__main__":
    main()
