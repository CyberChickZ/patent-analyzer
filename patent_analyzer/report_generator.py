#!/usr/bin/env python3
"""
Generate a self-contained static HTML report from patent analysis results.

UI design:
- NO standalone checklist section — checklist lives inside match cards
- Match cards: scrollable list, sorted by match count descending
- Hover a card → show only MATCHED checklist items (green hits)
- Click a card → expand to show ALL checklist items (hits + misses)
- Click "Export" → Markdown modal with copy/download
- Framing: "Possible Matches" for lawyer/staff review, NOT judgment

Usage:
    python3 report_generator.py --input results.json --output report.html
"""

import argparse
import json
import html as html_mod
from datetime import datetime, timezone


def escape(text: str) -> str:
    return html_mod.escape(str(text)) if text else ""


def score_color(score: float) -> str:
    if score > 0.7:
        return "#ff3b30"
    if score > 0.4:
        return "#ff9500"
    return "#0a84ff"


def match_count_label(hits: int, total: int) -> str:
    if hits == 0:
        return "No overlap detected"
    return f"{hits}/{total} points matched"


def generate_html(data: dict) -> str:
    phase1 = data.get("phase1", {})
    phase2 = data.get("phase2", {})
    search = data.get("search", {})
    evaluation = data.get("evaluation", {})

    status = phase1.get("status", "unknown")
    doc_mode = phase1.get("doc_mode", "unknown")
    summary_text = phase1.get("summary", "No summary available.")
    invention_type = phase1.get("invention_type", "Unknown")

    checklist = phase2.get("checklist", [])
    search_groups = search.get("groups", [])
    total_patents = search.get("summary", {}).get("total_patents", 0)
    total_papers = search.get("summary", {}).get("total_papers", 0)

    scoring_report = evaluation.get("scoring_report", [])
    overall_summary = evaluation.get("summary", "")

    generated_at = data.get("generated_at", datetime.now(timezone.utc).isoformat())

    # Count matches with >0 hits
    docs_with_hits = [m for m in scoring_report if (m.get("similarity_score", 0) or 0) > 0]

    # Build match cards — each card has hover (hits only) + click (expand all)
    match_cards_html = ""
    for i, m in enumerate(scoring_report):
        title = escape(m.get("title", "Unknown"))
        mid = escape(m.get("id", ""))
        mtype = escape(m.get("manuscript_type", "Document"))
        score_num = m.get("similarity_score", 0) or 0
        snippet = escape(m.get("snippet", ""))
        url = m.get("url", "")

        evals = m.get("similarity_categories", m.get("evaluations", {}))

        # Separate matched vs unmatched
        matched_items = []
        unmatched_items = []
        md_lines = []
        total_items = len(evals) if evals else len(checklist)
        hit_count = 0

        if evals:
            for req, ev in evals.items():
                is_match = ev.get("match", False) if isinstance(ev, dict) else False
                analysis = escape(ev.get("analysis", "")) if isinstance(ev, dict) else ""
                if is_match:
                    hit_count += 1
                    matched_items.append((req, analysis))
                else:
                    unmatched_items.append((req, analysis))
                md_status = "MATCH" if is_match else "NO MATCH"
                md_lines.append(f"- [{md_status}] **{escape(req)}**: {analysis}")

        s_color = score_color(score_num)

        # Hover zone: only matched items
        hover_html = ""
        if matched_items:
            hover_html = '<div class="hover-hits">'
            for req, analysis in matched_items:
                hover_html += f'<div class="hit-item"><span class="hit-icon">&#x2705;</span> {escape(req)}'
                if analysis:
                    hover_html += f'<div class="hit-analysis">{analysis}</div>'
                hover_html += '</div>'
            hover_html += '</div>'

        # Expanded zone: ALL items (shown on click)
        expanded_html = '<div class="expanded-evals">'
        for req, analysis in matched_items:
            expanded_html += f'<div class="eval-item"><span class="eval-true">&#x2705;</span> {escape(req)}'
            if analysis:
                expanded_html += f'<div class="eval-analysis">{analysis}</div>'
            expanded_html += '</div>'
        for req, analysis in unmatched_items:
            expanded_html += f'<div class="eval-item"><span class="eval-false">&#x274C;</span> {escape(req)}'
            if analysis:
                expanded_html += f'<div class="eval-analysis">{analysis}</div>'
            expanded_html += '</div>'
        expanded_html += '</div>'

        # Markdown for export
        md_content = f"""# {title}
- **ID**: {mid}
- **Type**: {mtype}
- **Matched Points**: {hit_count}/{total_items}
{f'- **URL**: {url}' if url else ''}
{f'- **Snippet**: {snippet}' if snippet else ''}

## Checklist Evaluation
""" + "\n".join(md_lines)

        md_escaped = escape(md_content).replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

        # Card visibility class
        card_class = "match-card has-hits" if hit_count > 0 else "match-card no-hits"

        match_cards_html += f'''
        <div class="{card_class}" data-hits="{hit_count}" data-md="{md_escaped}">
          <div class="mc-header" onclick="toggleCard(this.parentElement)">
            <div class="mc-title-row">
              <div class="mc-title" title="{title}">{title}</div>
              <div class="mc-hits" style="color:{s_color}">{match_count_label(hit_count, total_items)}</div>
            </div>
            <div class="mc-meta">
              <span class="mc-badge">{mtype}</span>
              <span class="mc-id">{mid}</span>
              {f'<a class="mc-link" href="{escape(url)}" target="_blank" onclick="event.stopPropagation()">View PDF</a>' if url else ''}
              <button class="mc-export" onclick="event.stopPropagation(); showMarkdown(this.closest(\'.match-card\'))">Export MD</button>
            </div>
          </div>
          {f'<div class="mc-snippet">&ldquo;{snippet}&rdquo;</div>' if snippet else ''}
          {hover_html}
          {expanded_html}
        </div>'''

    # Search summary cards
    agent_cards_html = ""
    for ar in search_groups:
        gid = escape(ar.get("group_id", ""))
        label = escape(ar.get("label", ""))
        pq = escape(ar.get("patent_query", ""))
        pm = ar.get("patent_matches_found", 0)
        sq = escape(ar.get("paper_query", ""))
        sm = ar.get("paper_matches_found", 0)

        agent_cards_html += f'''
        <div class="ar-card">
          <div class="ar-card-header"><span class="ar-group-id">{gid}</span></div>
          {f'<div class="ar-label">{label}</div>' if label else ''}
          {f'<div class="ar-query-section"><div class="ar-query-title">Patent <span>{pm}</span></div><div class="ar-query-text">{pq}</div></div>' if pq else ''}
          {f'<div class="ar-query-section"><div class="ar-query-title">Scholar <span>{sm}</span></div><div class="ar-query-text">{sq}</div></div>' if sq else ''}
        </div>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prior Art Search Report</title>
<style>
:root {{
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
  line-height: 1.5;
  --bg: #0b0e13; --card: #0f141c; --border: #1b2533;
  --text: #e7eef7; --text2: #8899aa; --accent: #0a84ff;
  --hit: #34c759; --miss: #48505a;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); padding: 2rem; }}
.container {{ max-width: 1100px; margin: 0 auto; }}

.report-header {{ text-align: center; padding: 2rem 0; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }}
.report-header h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }}
.report-header .subtitle {{ color: var(--text2); font-size: 0.9rem; margin-bottom: 1rem; }}
.pill-row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center; margin: 0.75rem 0; }}
.pill {{ display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.3rem 0.8rem; border-radius: 99px; background: #1a2233; font-size: 0.8rem; color: var(--text2); }}
.pill strong {{ color: var(--text); }}

.section {{ margin: 1.5rem 0; padding: 1.25rem; border: 1px solid var(--border); border-radius: 12px; background: var(--card); }}
.section-title {{ font-size: 1.05rem; font-weight: 600; margin-bottom: 0.75rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }}
.section-body {{ font-size: 0.9rem; color: var(--text2); line-height: 1.65; }}
.collapsible-header {{ cursor: pointer; display: flex; align-items: center; gap: 0.5rem; }}
.collapsible-header::before {{ content: "\\25B6"; font-size: 0.65rem; transition: transform 0.2s; }}
.collapsible-header.open::before {{ transform: rotate(90deg); }}
.collapsible-body {{ display: none; margin-top: 0.75rem; }}
.collapsible-body.open {{ display: block; }}

.ar-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 0.75rem; }}
.ar-card {{ border: 1px solid var(--border); border-radius: 10px; padding: 0.75rem; background: #111822; }}
.ar-group-id {{ font-weight: 600; color: var(--accent); font-size: 0.85rem; }}
.ar-label {{ font-size: 0.82rem; color: var(--text); margin: 0.3rem 0; }}
.ar-query-section {{ margin-top: 0.4rem; }}
.ar-query-title {{ font-size: 0.72rem; font-weight: 600; color: var(--text2); text-transform: uppercase; display: flex; justify-content: space-between; }}
.ar-query-title span {{ color: var(--accent); }}
.ar-query-text {{ font-size: 0.78rem; padding: 0.3rem 0.5rem; background: #0b0e13; border-radius: 5px; margin-top: 0.2rem; font-family: "SF Mono", Monaco, monospace; word-break: break-all; }}

/* ========== Match Cards ========== */
.matches-header {{ font-size: 1.05rem; font-weight: 600; margin: 1.5rem 0 0.5rem; }}
.matches-note {{ font-size: 0.82rem; color: var(--text2); margin-bottom: 1rem; }}

.match-card {{
  border: 1px solid var(--border); border-radius: 12px; padding: 1rem;
  background: var(--card); margin-bottom: 0.75rem; transition: border-color 0.2s;
}}
.match-card.has-hits {{ border-left: 3px solid var(--accent); }}
.match-card:hover {{ border-color: var(--accent); }}

.mc-header {{ cursor: pointer; }}
.mc-title-row {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }}
.mc-title {{ font-weight: 600; font-size: 0.92rem; flex: 1; }}
.mc-hits {{ font-weight: 600; font-size: 0.85rem; white-space: nowrap; }}
.mc-meta {{ display: flex; gap: 0.5rem; align-items: center; margin-top: 0.3rem; flex-wrap: wrap; }}
.mc-badge {{ padding: 0.12rem 0.45rem; border-radius: 5px; background: #1a2233; font-size: 0.72rem; color: var(--text2); }}
.mc-id {{ font-size: 0.75rem; color: var(--text2); font-family: "SF Mono", Monaco, monospace; }}
.mc-link {{ font-size: 0.75rem; color: var(--accent); text-decoration: none; }}
.mc-export {{ font-size: 0.72rem; color: var(--accent); background: none; border: 1px solid var(--accent); border-radius: 5px; padding: 0.1rem 0.5rem; cursor: pointer; }}
.mc-export:hover {{ background: var(--accent); color: #fff; }}
.mc-snippet {{ font-size: 0.82rem; color: var(--text2); font-style: italic; margin: 0.4rem 0; padding: 0.4rem 0.6rem; background: #111822; border-radius: 6px; }}

/* Hover zone: only matched items, visible on hover */
.hover-hits {{ display: none; margin-top: 0.6rem; padding: 0.5rem; background: #111822; border-radius: 8px; border-left: 2px solid var(--hit); }}
.match-card:hover .hover-hits {{ display: block; }}
.match-card.expanded .hover-hits {{ display: none; }}
.hit-item {{ padding: 0.25rem 0; font-size: 0.82rem; display: flex; align-items: flex-start; gap: 0.3rem; }}
.hit-icon {{ color: var(--hit); flex-shrink: 0; }}
.hit-analysis {{ font-size: 0.78rem; color: var(--text2); margin-left: 1.4rem; }}

/* Expanded zone: ALL items, visible on click */
.expanded-evals {{ display: none; margin-top: 0.6rem; }}
.match-card.expanded .expanded-evals {{ display: block; }}
.eval-item {{ padding: 0.3rem 0; font-size: 0.82rem; display: flex; align-items: flex-start; gap: 0.3rem; border-top: 1px solid #1a2233; }}
.eval-item:first-child {{ border-top: none; }}
.eval-true {{ color: var(--hit); flex-shrink: 0; }}
.eval-false {{ color: var(--miss); flex-shrink: 0; }}
.eval-analysis {{ font-size: 0.78rem; color: var(--text2); margin-left: 1.4rem; margin-top: 0.1rem; }}

/* Filter controls */
.filter-bar {{ display: flex; gap: 0.75rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }}
.filter-btn {{ padding: 0.3rem 0.8rem; border-radius: 8px; border: 1px solid var(--border); background: var(--card); color: var(--text2); cursor: pointer; font-size: 0.82rem; }}
.filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}

/* Modal */
.modal-overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: center; padding: 2rem; }}
.modal-overlay.active {{ display: flex; }}
.modal-content {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; max-width: 800px; width: 100%; max-height: 80vh; overflow-y: auto; position: relative; }}
.modal-close {{ position: absolute; top: 1rem; right: 1rem; background: none; border: none; color: var(--text2); cursor: pointer; font-size: 1.2rem; }}
.modal-actions {{ display: flex; gap: 0.75rem; margin-bottom: 1rem; }}
.modal-btn {{ padding: 0.4rem 1rem; border-radius: 8px; border: 1px solid var(--border); background: #1a2233; color: var(--text); cursor: pointer; font-size: 0.85rem; }}
.modal-btn:hover {{ background: var(--accent); border-color: var(--accent); }}
.modal-md {{ white-space: pre-wrap; font-family: "SF Mono", Monaco, monospace; font-size: 0.82rem; line-height: 1.6; color: var(--text2); padding: 1rem; background: #0b0e13; border-radius: 8px; }}
</style>
</head>
<body>
<div class="container">

  <div class="report-header">
    <h1>Prior Art Search Report</h1>
    <div class="subtitle">Possible matches for attorney/staff review &mdash; not a legal determination</div>
    <div class="pill-row">
      <span class="pill">Type: <strong>{escape(doc_mode)}</strong></span>
      <span class="pill">Category: <strong>{escape(invention_type)}</strong></span>
      <span class="pill">Searched: <strong>{total_patents} patents, {total_papers} papers</strong></span>
      <span class="pill">Candidates: <strong>{len(docs_with_hits)} with overlap</strong></span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Invention Summary</div>
    <div class="section-body"><p>{escape(summary_text)}</p></div>
  </div>

  {f"""<div class="section">
    <div class="section-title">Search Overview</div>
    <div class="section-body"><p>{escape(overall_summary)}</p></div>
  </div>""" if overall_summary else ""}

  {f"""<div class="section">
    <div class="section-title collapsible-header" onclick="toggleCollapse(this)">Search Groups ({len(search_groups)})</div>
    <div class="collapsible-body"><div class="ar-grid">{agent_cards_html}</div></div>
  </div>""" if agent_cards_html else ""}

  <div class="matches-header">Possible Matches ({len(scoring_report)} evaluated, {len(docs_with_hits)} with overlap)</div>
  <div class="matches-note">Hover to see matched points. Click to expand full evaluation. These are candidates for professional review.</div>

  <div class="filter-bar">
    <button class="filter-btn active" onclick="filterCards('all', this)">All ({len(scoring_report)})</button>
    <button class="filter-btn" onclick="filterCards('hits', this)">With Overlap ({len(docs_with_hits)})</button>
    <button class="filter-btn" onclick="filterCards('none', this)">No Overlap ({len(scoring_report) - len(docs_with_hits)})</button>
  </div>

  <div id="cardContainer">
    {match_cards_html}
  </div>

</div>

<div class="modal-overlay" id="mdModal" onclick="if(event.target===this)closeModal()">
  <div class="modal-content">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <div class="modal-actions">
      <button class="modal-btn" onclick="copyMarkdown()">Copy Markdown</button>
      <button class="modal-btn" onclick="downloadMarkdown()">Download .md</button>
    </div>
    <pre class="modal-md" id="mdContent"></pre>
  </div>
</div>

<script>
let currentMd = '';

function toggleCard(card) {{
  card.classList.toggle('expanded');
}}

function showMarkdown(card) {{
  const raw = card.getAttribute('data-md');
  currentMd = raw.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#x27;/g,"'");
  document.getElementById('mdContent').textContent = currentMd;
  document.getElementById('mdModal').classList.add('active');
}}

function closeModal() {{ document.getElementById('mdModal').classList.remove('active'); }}

function copyMarkdown() {{
  navigator.clipboard.writeText(currentMd).then(() => {{
    event.target.textContent = 'Copied!';
    setTimeout(() => event.target.textContent = 'Copy Markdown', 1500);
  }});
}}

function downloadMarkdown() {{
  const blob = new Blob([currentMd], {{type: 'text/markdown'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'match-evaluation.md';
  a.click();
}}

function toggleCollapse(header) {{
  header.classList.toggle('open');
  header.nextElementSibling.classList.toggle('open');
}}

function filterCards(mode, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.match-card').forEach(card => {{
    const hits = parseInt(card.getAttribute('data-hits') || '0');
    if (mode === 'all') card.style.display = '';
    else if (mode === 'hits') card.style.display = hits > 0 ? '' : 'none';
    else card.style.display = hits === 0 ? '' : 'none';
  }});
}}

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description="Generate patent analysis HTML report")
    parser.add_argument("--input", required=True, help="Path to results.json")
    parser.add_argument("--output", required=True, help="Output HTML path")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    html_content = generate_html(data)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report generated: {args.output}")


if __name__ == "__main__":
    main()
