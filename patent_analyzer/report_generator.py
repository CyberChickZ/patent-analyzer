#!/usr/bin/env python3
"""
Generate a self-contained static HTML report from patent analysis results.

Usage:
    python3 generate_report.py --input results.json --output report.html
"""

import argparse
import json
import html
import sys
from datetime import datetime, timezone


def escape(text: str) -> str:
    return html.escape(str(text)) if text else ""


def score_color(score: float) -> str:
    if score > 0.7:
        return "#ff3b30"
    if score > 0.4:
        return "#ff9500"
    return "#34c759"


def risk_level(score: float) -> tuple[str, str]:
    if score > 0.7:
        return "HIGH", "#ff3b30"
    if score > 0.4:
        return "MEDIUM", "#ff9500"
    if score > 0.2:
        return "LOW", "#34c759"
    return "CLEAR", "#30d158"


def generate_html(data: dict) -> str:
    phase1 = data.get("phase1", {})
    phase2 = data.get("phase2", {})
    search = data.get("search", {})
    evaluation = data.get("evaluation", {})

    # IDCA info
    status = phase1.get("status", "unknown")
    doc_mode = phase1.get("doc_mode", "unknown")
    summary_text = phase1.get("summary", "No summary available.")
    invention_type = phase1.get("invention_type", "Unknown")
    category_reasoning = phase1.get("reasoning", "")

    # Checklist
    checklist = phase2.get("checklist", [])
    atoms = phase2.get("delegation", {}).get("atoms", [])
    groups = phase2.get("delegation", {}).get("groups", [])

    # Search results
    search_groups = search.get("groups", [])
    total_patents = search.get("summary", {}).get("total_patents", 0)
    total_papers = search.get("summary", {}).get("total_papers", 0)

    # Evaluation
    scoring_report = evaluation.get("scoring_report", [])
    overall_summary = evaluation.get("summary", "")
    overall_reasoning = evaluation.get("reasoning", "")
    agent_report = search_groups

    # Risk
    top_score = max((m.get("similarity_score", 0) or 0 for m in scoring_report), default=0)
    risk, risk_color = risk_level(top_score)

    generated_at = data.get("generated_at", datetime.now(timezone.utc).isoformat())

    # Build match cards HTML
    match_cards_html = ""
    for i, m in enumerate(scoring_report):
        title = escape(m.get("title", "Unknown"))
        mid = escape(m.get("id", ""))
        mtype = escape(m.get("manuscript_type", "Document"))
        score_num = m.get("similarity_score", 0) or 0
        s_color = score_color(score_num)
        snippet = escape(m.get("snippet", ""))
        url = m.get("url", "")

        evals = m.get("similarity_categories", m.get("evaluations", {}))
        evals_html = ""
        md_lines = []
        if evals:
            evals_html = '<div class="eval-list">'
            for req, ev in evals.items():
                is_match = ev.get("match", False) if isinstance(ev, dict) else False
                icon = "&#x2705;" if is_match else "&#x274C;"
                icon_class = "eval-true" if is_match else "eval-false"
                analysis = escape(ev.get("analysis", "")) if isinstance(ev, dict) else ""
                evals_html += f'''
                <div class="eval-item">
                  <div class="eval-req"><span class="{icon_class}">{icon}</span> {escape(req)}</div>
                  {f'<div class="eval-analysis">{analysis}</div>' if analysis else ''}
                </div>'''
                md_status = "MATCH" if is_match else "NO MATCH"
                md_lines.append(f"- [{md_status}] **{escape(req)}**: {analysis}")
            evals_html += "</div>"

        # Build markdown for this match
        md_content = f"""# {title}
- **ID**: {mid}
- **Type**: {mtype}
- **Score**: {score_num:.0%}
{f'- **URL**: {url}' if url else ''}
{f'- **Snippet**: {snippet}' if snippet else ''}

## Checklist Evaluation
""" + "\n".join(md_lines)

        md_escaped = escape(md_content).replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

        match_cards_html += f'''
        <div class="match-card" onclick="showMarkdown(this)" data-md="{md_escaped}">
          <div class="mc-header">
            <div class="mc-title-row">
              <div class="mc-title" title="{title}">{title}</div>
              <div class="mc-score" style="color:{s_color}">{score_num:.0%} Match</div>
            </div>
            <div class="mc-meta">
              <span class="mc-badge">{mtype}</span>
              <span class="mc-id">{mid}</span>
              {f'<a class="mc-link" href="{escape(url)}" target="_blank" onclick="event.stopPropagation()">View PDF</a>' if url else ''}
            </div>
          </div>
          {f'<div class="mc-snippet">&ldquo;{snippet}&rdquo;</div>' if snippet else ''}
          {evals_html}
        </div>'''

    # Build agent report cards
    agent_cards_html = ""
    for ar in agent_report:
        gid = escape(ar.get("group_id", ""))
        label = escape(ar.get("label", ""))
        excerpt_text = escape(ar.get("excerpt", ""))
        pq = escape(ar.get("patent_query", ""))
        pm = ar.get("patent_matches_found", 0)
        sq = escape(ar.get("paper_query", ""))
        sm = ar.get("paper_matches_found", 0)

        agent_cards_html += f'''
        <div class="ar-card">
          <div class="ar-card-header"><span class="ar-group-id">{gid}</span></div>
          {f'<div class="ar-label">{label}</div>' if label else ''}
          {f'<div class="ar-excerpt">&ldquo;{excerpt_text}&rdquo;</div>' if excerpt_text else ''}
          {f'''<div class="ar-query-section">
            <div class="ar-query-title">Patent Query <span>{pm} matches</span></div>
            <div class="ar-query-text">{pq}</div>
          </div>''' if pq else ''}
          {f'''<div class="ar-query-section">
            <div class="ar-query-title">Paper Query <span>{sm} matches</span></div>
            <div class="ar-query-text">{sq}</div>
          </div>''' if sq else ''}
        </div>'''

    # Atoms table
    atoms_html = ""
    if atoms:
        atoms_html = '<table class="atoms-table"><thead><tr><th>ID</th><th>Name</th><th>Core</th><th>Distinct</th><th>Keywords</th></tr></thead><tbody>'
        for atom in atoms:
            kws = ", ".join(atom.get("keywords", [])[:6])
            atoms_html += f'''<tr>
              <td>{escape(atom.get("id",""))}</td>
              <td>{escape(atom.get("name",""))}</td>
              <td>{atom.get("core_score",0):.2f}</td>
              <td>{atom.get("distinctiveness_score",0):.2f}</td>
              <td class="kw-cell">{escape(kws)}</td>
            </tr>'''
        atoms_html += "</tbody></table>"

    # Checklist HTML
    checklist_html = ""
    if checklist:
        checklist_html = '<ol class="checklist-list">'
        for item in checklist:
            checklist_html += f"<li>{escape(item)}</li>"
        checklist_html += "</ol>"

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Patent Analysis Report</title>
<style>
:root {{
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  --bg: #0b0e13;
  --card: #0f141c;
  --border: #1b2533;
  --text: #e7eef7;
  --text2: #8899aa;
  --accent: #0a84ff;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg); color: var(--text); padding: 2rem; }}
.container {{ max-width: 1100px; margin: 0 auto; }}

/* Header */
.report-header {{
  text-align: center; padding: 2rem 0; border-bottom: 1px solid var(--border); margin-bottom: 2rem;
}}
.report-header h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }}
.report-header .meta {{ color: var(--text2); font-size: 0.85rem; }}

/* Pills */
.pill-row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0; }}
.pill {{
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.35rem 0.9rem; border-radius: 99px;
  background: #1a2233; font-size: 0.82rem; color: var(--text2);
}}
.pill strong {{ color: var(--text); }}

/* Sections */
.section {{
  margin: 1.5rem 0; padding: 1.25rem; border: 1px solid var(--border);
  border-radius: 12px; background: var(--card);
}}
.section-title {{
  font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;
  padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}}
.section-body {{ font-size: 0.92rem; color: var(--text2); line-height: 1.65; }}
.section-body p {{ margin-bottom: 0.75rem; }}

/* Summary */
.exec-text {{ font-size: 0.95rem; color: var(--text); }}
.exec-reasoning {{ margin-top: 0.75rem; padding: 0.75rem; background: #111822; border-radius: 8px; font-size: 0.88rem; }}

/* Agent report grid */
.ar-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }}
.ar-card {{
  border: 1px solid var(--border); border-radius: 10px; padding: 1rem;
  background: #111822; transition: border-color 0.2s;
}}
.ar-card:hover {{ border-color: var(--accent); }}
.ar-card-header {{ margin-bottom: 0.5rem; }}
.ar-group-id {{ font-weight: 600; color: var(--accent); }}
.ar-label {{ font-size: 0.88rem; color: var(--text); margin-bottom: 0.4rem; }}
.ar-excerpt {{ font-size: 0.82rem; color: var(--text2); font-style: italic; margin-bottom: 0.75rem; }}
.ar-query-section {{ margin-top: 0.5rem; }}
.ar-query-title {{
  font-size: 0.78rem; font-weight: 600; color: var(--text2); text-transform: uppercase;
  display: flex; justify-content: space-between;
}}
.ar-query-title span {{ color: var(--accent); }}
.ar-query-text {{
  font-size: 0.82rem; padding: 0.4rem 0.6rem; background: #0b0e13;
  border-radius: 6px; margin-top: 0.25rem; font-family: "SF Mono", Monaco, monospace;
  word-break: break-all;
}}

/* Match cards */
.matches-header {{ font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 1rem; }}
.match-card {{
  border: 1px solid var(--border); border-radius: 12px; padding: 1rem;
  background: var(--card); margin-bottom: 1rem; cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
}}
.match-card:hover {{ border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }}
.mc-header {{ margin-bottom: 0.5rem; }}
.mc-title-row {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }}
.mc-title {{ font-weight: 600; font-size: 0.95rem; flex: 1; }}
.mc-score {{ font-weight: 700; font-size: 0.95rem; white-space: nowrap; }}
.mc-meta {{ display: flex; gap: 0.6rem; align-items: center; margin-top: 0.35rem; flex-wrap: wrap; }}
.mc-badge {{
  display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px;
  background: #1a2233; font-size: 0.75rem; color: var(--text2);
}}
.mc-id {{ font-size: 0.78rem; color: var(--text2); font-family: "SF Mono", Monaco, monospace; }}
.mc-link {{ font-size: 0.78rem; color: var(--accent); text-decoration: none; }}
.mc-link:hover {{ text-decoration: underline; }}
.mc-snippet {{
  font-size: 0.85rem; color: var(--text2); font-style: italic;
  margin: 0.5rem 0; padding: 0.5rem 0.75rem; background: #111822; border-radius: 8px;
}}

/* Evaluation items */
.eval-list {{ margin-top: 0.75rem; }}
.eval-item {{ padding: 0.4rem 0; border-top: 1px solid #1a2233; }}
.eval-item:first-child {{ border-top: none; }}
.eval-req {{ font-size: 0.85rem; display: flex; align-items: flex-start; gap: 0.4rem; }}
.eval-true {{ color: #34c759; }}
.eval-false {{ color: #ff3b30; }}
.eval-analysis {{ font-size: 0.8rem; color: var(--text2); margin-left: 1.6rem; margin-top: 0.15rem; }}

/* Atoms table */
.atoms-table {{
  width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem;
}}
.atoms-table th {{
  text-align: left; padding: 0.5rem; border-bottom: 2px solid var(--border);
  color: var(--text2); font-size: 0.78rem; text-transform: uppercase;
}}
.atoms-table td {{ padding: 0.5rem; border-bottom: 1px solid var(--border); }}
.kw-cell {{ font-size: 0.8rem; color: var(--text2); }}

/* Checklist */
.checklist-list {{ padding-left: 1.5rem; }}
.checklist-list li {{ margin-bottom: 0.4rem; font-size: 0.88rem; }}

/* Modal */
.modal-overlay {{
  display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7);
  z-index: 1000; justify-content: center; align-items: center; padding: 2rem;
}}
.modal-overlay.active {{ display: flex; }}
.modal-content {{
  background: var(--card); border: 1px solid var(--border); border-radius: 16px;
  padding: 1.5rem; max-width: 800px; width: 100%; max-height: 80vh; overflow-y: auto;
  position: relative;
}}
.modal-close {{
  position: absolute; top: 1rem; right: 1rem; background: none; border: none;
  color: var(--text2); cursor: pointer; font-size: 1.2rem;
}}
.modal-close:hover {{ color: var(--text); }}
.modal-actions {{ display: flex; gap: 0.75rem; margin-bottom: 1rem; }}
.modal-btn {{
  padding: 0.4rem 1rem; border-radius: 8px; border: 1px solid var(--border);
  background: #1a2233; color: var(--text); cursor: pointer; font-size: 0.85rem;
}}
.modal-btn:hover {{ background: var(--accent); border-color: var(--accent); }}
.modal-md {{
  white-space: pre-wrap; font-family: "SF Mono", Monaco, Consolas, monospace;
  font-size: 0.85rem; line-height: 1.6; color: var(--text2);
  padding: 1rem; background: #0b0e13; border-radius: 8px;
}}

/* Collapsible */
.collapsible-header {{
  cursor: pointer; display: flex; align-items: center; gap: 0.5rem;
}}
.collapsible-header::before {{ content: "\\25B6"; font-size: 0.7rem; transition: transform 0.2s; }}
.collapsible-header.open::before {{ transform: rotate(90deg); }}
.collapsible-body {{ display: none; margin-top: 0.75rem; }}
.collapsible-body.open {{ display: block; }}

/* Risk badge */
.risk-badge {{
  display: inline-block; padding: 0.3rem 1rem; border-radius: 99px;
  font-weight: 700; font-size: 0.9rem; letter-spacing: 0.05em;
}}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="report-header">
    <h1>Patent Novelty Analysis Report</h1>
    <div class="meta">Generated: {escape(generated_at)}</div>
    <div class="pill-row">
      <span class="pill">Status: <strong>{escape(status.upper())}</strong></span>
      <span class="pill">Type: <strong>{escape(doc_mode)}</strong></span>
      <span class="pill">Category: <strong>{escape(invention_type)}</strong></span>
      <span class="pill">Prior Art: <strong>{total_patents} patents, {total_papers} papers</strong></span>
      <span class="risk-badge" style="background:{risk_color}22;color:{risk_color}">Risk: {risk}</span>
    </div>
  </div>

  <!-- Executive Summary -->
  <div class="section">
    <div class="section-title">Executive Summary</div>
    <div class="section-body">
      <p class="exec-text">{escape(overall_summary) if overall_summary else escape(summary_text)}</p>
      {f'<div class="exec-reasoning"><strong>Reasoning:</strong> {escape(overall_reasoning)}</div>' if overall_reasoning else ''}
      {f'<div class="exec-reasoning" style="margin-top:0.5rem"><strong>Category Reasoning:</strong> {escape(category_reasoning)}</div>' if category_reasoning else ''}
    </div>
  </div>

  <!-- Invention Summary -->
  <div class="section">
    <div class="section-title collapsible-header" onclick="toggleCollapse(this)">Invention Summary</div>
    <div class="collapsible-body">
      <div class="section-body"><p>{escape(summary_text)}</p></div>
    </div>
  </div>

  <!-- Checklist -->
  <div class="section">
    <div class="section-title collapsible-header" onclick="toggleCollapse(this)">Evaluation Checklist ({len(checklist)} items)</div>
    <div class="collapsible-body">
      {checklist_html}
    </div>
  </div>

  <!-- Atoms -->
  {f"""<div class="section">
    <div class="section-title collapsible-header" onclick="toggleCollapse(this)">Atomic Components ({len(atoms)} atoms)</div>
    <div class="collapsible-body">{atoms_html}</div>
  </div>""" if atoms else ""}

  <!-- Subagent Search Summary -->
  {f"""<div class="section">
    <div class="section-title">Search Summary</div>
    <div class="ar-grid">{agent_cards_html}</div>
  </div>""" if agent_cards_html else ""}

  <!-- Prior Art Matches -->
  {f"""<div class="matches-header">Prior Art Matches ({len(scoring_report)} evaluated)</div>
  {match_cards_html}""" if scoring_report else '<div class="section"><div class="section-body">No prior art matches evaluated.</div></div>'}

</div>

<!-- Markdown Modal -->
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

function showMarkdown(card) {{
  const raw = card.getAttribute('data-md');
  currentMd = raw.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#x27;/g,"'");
  document.getElementById('mdContent').textContent = currentMd;
  document.getElementById('mdModal').classList.add('active');
}}

function closeModal() {{
  document.getElementById('mdModal').classList.remove('active');
}}

function copyMarkdown() {{
  navigator.clipboard.writeText(currentMd).then(() => {{
    const btn = event.target;
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy Markdown', 1500);
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
  const body = header.nextElementSibling;
  body.classList.toggle('open');
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
