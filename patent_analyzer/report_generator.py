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
        snippet = esc(m.get("snippet", ""))

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

        # Tags for hover (short labels, green)
        tags_html = ""
        if matched:
            tags_html = '<div class="hover-tags">' + "".join(
                f'<span class="tag tag-hit">{s}</span>' for s, _, _ in matched
            ) + '</div>'

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

        return f'''
<div class="{card_cls}" data-hits="{hit_count}" data-md="{md_esc}">
  <div class="card-top" onclick="toggle(this.parentElement)">
    <div class="card-left">
      <span class="type-dot {'dot-patent' if badge_type == 'Patent' else 'dot-paper'}"></span>
      <div>
        <div class="card-title">{title}</div>
        <div class="card-id">{esc(badge_type)} &middot; {mid}</div>
      </div>
    </div>
    <div class="card-right">
      {f'<span class="hit-badge">{hit_count}/{total}</span>' if hit_count > 0 else '<span class="no-badge">0</span>'}
      {f'<a class="pdf-link" href="{esc(url)}" target="_blank" onclick="event.stopPropagation()">PDF</a>' if url else ''}
      <button class="md-btn" onclick="event.stopPropagation();showMd(this.closest(\'.card\'))">MD</button>
    </div>
  </div>
  {tags_html}
  {f'<div class="card-snippet">{snippet}</div>' if snippet and hit_count > 0 else ''}
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
  --bg:#f7f8fa;--card:#fff;--border:#e2e6ec;--text:#1a1d23;--text2:#6b7280;
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

/* Sections */
.sec{{margin:1.5rem 0;padding:1rem 1.25rem;border:1px solid var(--border);border-radius:10px;background:var(--card)}}
.sec-t{{font-size:0.95rem;font-weight:600;margin-bottom:0.5rem}}
.sec-b{{font-size:0.88rem;color:var(--text2);line-height:1.7}}
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
.fbtn{{font-size:0.78rem;padding:0.3rem 0.75rem;border-radius:7px;border:1px solid var(--border);background:var(--card);color:var(--text2);cursor:pointer}}
.fbtn.on{{background:var(--accent);border-color:var(--accent);color:#fff}}
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
.hit-badge{{background:var(--accent);color:#fff;font-size:0.72rem;font-weight:700;padding:0.15rem 0.55rem;border-radius:6px}}
.no-badge{{color:var(--miss);font-size:0.72rem}}
.pdf-link{{font-size:0.72rem;color:var(--accent);text-decoration:none}}
.md-btn{{font-size:0.68rem;color:var(--accent);background:none;border:1px solid var(--accent);border-radius:5px;padding:0.1rem 0.4rem;cursor:pointer}}
.md-btn:hover{{background:var(--accent);color:#fff}}

/* Hover tags */
.hover-tags{{display:none;padding:0.4rem 1rem 0.6rem;gap:0.35rem;flex-wrap:wrap}}
.card:hover .hover-tags{{display:flex}}
.card.open .hover-tags{{display:none}}
.tag{{font-size:0.72rem;padding:0.15rem 0.5rem;border-radius:5px}}
.tag-hit{{background:#dcfce7;color:#166534}}

.card-snippet{{font-size:0.8rem;color:var(--text2);padding:0 1rem 0.6rem;font-style:italic;line-height:1.5}}

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
  <div class="sec-b">{esc(summary_text)}</div>
</div>

{f'<div class="sec"><div class="sec-t">Search Overview</div><div class="sec-b">{esc(overall_summary)}</div></div>' if overall_summary else ''}

{f'''<div class="sec">
  <div class="sec-t tog" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">Search Groups ({len(search_groups)})</div>
  <div class="tog-body">{sg_html}</div>
</div>''' if sg_html else ''}

<div class="bar">
  <div class="bar-title">Possible Matches</div>
  <button class="fbtn" onclick="showAll(this)">All ({len(scoring_report)})</button>
  <button class="fbtn on" onclick="showHits(this)">With Overlap ({len(hits_docs)})</button>
  <button class="fbtn" onclick="showNone(this)">No Overlap ({len(no_hits_docs)})</button>
</div>
<div class="bar-note">Hover for matched points &middot; Click to expand full evaluation &middot; MD to export Markdown</div>

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
