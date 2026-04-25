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
import html as html_mod
import json
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


def md_inline(t: str) -> str:
    """Convert inline markdown (bold/italic/code) to HTML. Input must already be HTML-escaped."""
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<em>\1</em>', t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    return t


def render_markdown(text: str) -> str:
    """Convert markdown text to HTML. Handles headings, lists, bold, italic, code, paragraphs."""
    if not text:
        return ""
    lines = text.split("\n")
    html_parts = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Empty line — skip (paragraph breaks handled by accumulation)
        if not stripped:
            i += 1
            continue

        # Headings
        hm = re.match(r'^(#{1,4})\s+(.+)$', stripped)
        if hm:
            level = len(hm.group(1))
            # Map ## → h3, ### → h4, # → h2, #### → h5
            tag = f"h{min(level + 1, 6)}"
            html_parts.append(f'<{tag} class="md-heading">{md_inline(esc(hm.group(2)))}</{tag}>')
            i += 1
            continue

        # Unordered list block (- item or * item)
        if re.match(r'^[-*]\s+', stripped):
            items = []
            while i < len(lines) and re.match(r'^\s*[-*]\s+', lines[i]):
                item_text = re.sub(r'^\s*[-*]\s+', '', lines[i]).strip()
                items.append(f'<li>{md_inline(esc(item_text))}</li>')
                i += 1
            html_parts.append(f'<ul class="md-list">{"".join(items)}</ul>')
            continue

        # Ordered list block (1. item, 2. item, etc.)
        if re.match(r'^\d+\.\s+', stripped):
            items = []
            while i < len(lines) and re.match(r'^\s*\d+\.\s+', lines[i]):
                item_text = re.sub(r'^\s*\d+\.\s+', '', lines[i]).strip()
                items.append(f'<li>{md_inline(esc(item_text))}</li>')
                i += 1
            html_parts.append(f'<ol class="md-list">{"".join(items)}</ol>')
            continue

        # Regular paragraph — accumulate contiguous non-blank, non-special lines
        para_lines = []
        while i < len(lines):
            cl = lines[i].strip()
            if not cl:
                i += 1
                break
            if re.match(r'^#{1,4}\s+', cl) or re.match(r'^[-*]\s+', cl) or re.match(r'^\d+\.\s+', cl):
                break
            para_lines.append(cl)
            i += 1
        para_text = " ".join(para_lines)
        html_parts.append(f'<p class="md-para">{md_inline(esc(para_text))}</p>')

    return "\n".join(html_parts)


def _extract_key_terms(text: str) -> list[str]:
    """Dynamically extract key technical terms from text."""
    terms = set()
    # Quoted terms
    for m in re.finditer(r'"([^"]{3,40})"', text):
        terms.add(m.group(1))
    # Backtick terms
    for m in re.finditer(r'`([^`]{2,40})`', text):
        terms.add(m.group(1))
    # Capitalized multi-word phrases (e.g. "Contrastive Loss", "Self-Supervised Learning")
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)+)\b', text):
        terms.add(m.group(1))
    # ALL-CAPS acronyms 2-6 chars (e.g. TSDF, RGBD, CNN, LLM)
    for m in re.finditer(r'\b([A-Z]{2,6})\b', text):
        terms.add(m.group(1))
    # Hyphenated technical terms (e.g. self-supervised, cross-object)
    for m in re.finditer(r'\b([a-z]+-[a-z]+(?:-[a-z]+)?)\b', text, re.IGNORECASE):
        if len(m.group(1)) > 5:
            terms.add(m.group(1))
    return sorted(terms, key=len, reverse=True)


def format_summary(text: str) -> str:
    """Format invention summary with markdown rendering and dynamic keyword highlighting."""
    terms = _extract_key_terms(text)

    def highlight(t: str) -> str:
        result = md_inline(esc(t))
        for kw in terms:
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

    # Use render_markdown for structured text, then apply highlighting
    rendered = render_markdown(text)
    if rendered:
        for kw in terms:
            pattern = re.compile(re.escape(esc(kw)), re.IGNORECASE)
            rendered = pattern.sub(lambda m: f'<mark>{m.group()}</mark>', rendered)
        return rendered

    return f'<p class="sum-para">{highlight(text)}</p>'


def _format_cl_item(item) -> tuple[str, str]:
    """Return (criterion_text, weight_html) for a checklist item."""
    if isinstance(item, dict):
        w = item.get("weight", 0)
        criterion = item.get("criterion", str(item))
        weight_pct = f'{w * 100:.0f}%' if w > 0 else ''
        weight_html = f' <span class="cl-weight">{esc(weight_pct)}</span>' if weight_pct else ''
        return criterion, weight_html
    return str(item), ''


def _format_cl_item_plain(item) -> str:
    """Return just the criterion text (no weight prefix) for display in cards."""
    if isinstance(item, dict):
        return item.get("criterion", str(item))
    return str(item)


def shorten_checklist(item: str, max_len: int = 60) -> str:
    """Shorten a checklist item to a readable tag."""
    # Strip [w=X.XX] prefix if present
    item = re.sub(r'^\[w=[\d.]+\]\s*', '', item)
    # Remove common prefixes
    item = re.sub(r'^(The (system|method|network|training|process|pipeline|entire|invention)\s+(uses?|includes?|performs?|enables?|is|can|from)\s+)', '', item, flags=re.IGNORECASE)
    item = re.sub(r'^(A |An )', '', item)
    if len(item) > max_len:
        item = item[:max_len].rsplit(' ', 1)[0] + '...'
    return item


def _clean_req_text(text: str) -> str:
    """Strip [w=X.XX] prefix from requirement/criterion text for display."""
    return re.sub(r'^\[w=[\d.]+\]\s*', '', text)


def _generate_implied_html(data: dict) -> str:
    """Abbreviated report for Implied inventions — no prior art search was run."""
    phase1 = data.get("phase1", {})
    phase2 = data.get("phase2", {})
    title = esc(data.get("source_title", data.get("source_filename", "Document")))
    summary = esc(phase1.get("summary", ""))
    reasoning = esc(phase1.get("reasoning", ""))
    ucd = esc(phase2.get("ucd", ""))
    fields = phase1.get("fields_map", [])
    fields_html = ", ".join(esc(f) for f in fields) if fields else "Not classified"
    gen_at = data.get("generated_at", "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Implied Invention Report — {title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; color: #1e293b; line-height: 1.6; }}
h1 {{ font-size: 1.5rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
.banner {{ background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem 1.25rem; margin: 1.5rem 0; }}
.banner strong {{ color: #92400e; }}
.section {{ margin: 1.5rem 0; }}
.section h2 {{ font-size: 1.1rem; color: #475569; margin-bottom: 0.5rem; }}
.meta {{ color: #64748b; font-size: 0.85rem; }}
pre {{ background: #f8fafc; padding: 1rem; border-radius: 6px; white-space: pre-wrap; font-size: 0.9rem; overflow-x: auto; }}
</style></head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {esc(gen_at)} · Fields: {fields_html}</p>

<div class="banner">
<strong>Implied Invention</strong> — This document describes an inventive concept but does not
provide concrete implementation details. No prior art search was conducted because there are
insufficient technical details to search against.
</div>

<div class="section"><h2>Why "Implied"?</h2><p>{reasoning}</p></div>

<div class="section"><h2>Invention Summary</h2><p>{summary}</p></div>

<div class="section"><h2>Technical Elements Identified</h2><pre>{ucd if ucd else '(No decomposition available)'}</pre></div>

<div class="section"><h2>Recommendation</h2>
<p>To proceed with a prior art search, the document would need to include concrete
implementation details — specific architectures, algorithms, materials, experimental results,
or system designs. Consider revising the document to include these details before resubmitting.</p>
</div>
</body></html>"""


def _render_confidence_banner(entropy_profile: dict | None) -> str:
    if not entropy_profile:
        return ""
    conf = entropy_profile.get("overall_confidence", "medium")
    degs = entropy_profile.get("degradation_points", [])
    if conf == "high":
        bg, border, color = "#dcfce7", "#16a34a", "#166534"
        label = "High confidence"
        msg = "Analysis grounded in strong evidence across all phases."
    elif conf == "low":
        bg, border, color = "#fef2f2", "#dc2626", "#991b1b"
        label = "Low confidence"
        msg = "Some analysis steps lack direct evidence."
    else:
        bg, border, color = "#fef3c7", "#f59e0b", "#92400e"
        label = "Medium confidence"
        msg = "Some evaluation criteria lack direct evidence."
    deg_html = ""
    if degs:
        items = "".join(f"<li>{esc(d)}</li>" for d in degs)
        deg_html = f'<ul style="margin:0.5rem 0 0;padding-left:1.2rem">{items}</ul>'
    return (
        f'<div style="background:{bg};border:1px solid {border};'
        f'border-radius:8px;padding:1rem 1.25rem;margin:1.5rem 0">'
        f'<strong style="color:{color}">{label}</strong> — {esc(msg)}'
        f'{deg_html}</div>'
    )


# Inline SVG download icon (20x20, arrow pointing down into tray)
_DL_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>'
)

_NO_PDF_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" opacity="0.4"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>'
    '<line x1="2" y1="2" x2="22" y2="22"/></svg>'
)


def generate_html(data: dict) -> str:
    # Implied invention: abbreviated report
    if data.get("implied_invention"):
        return _generate_implied_html(data)

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
    source_filename = data.get("source_filename", "")
    job_id = data.get("job_id", "")
    source_title = data.get("source_title", "")

    exact_find = [m for m in scoring_report if m.get("is_self_match")]
    non_self = [m for m in scoring_report if not m.get("is_self_match")]
    hits_docs = sorted(
        [m for m in non_self if (m.get("similarity_score", 0) or 0) > 0],
        key=lambda x: x.get("similarity_score", 0), reverse=True)
    no_hits_docs = [m for m in non_self if (m.get("similarity_score", 0) or 0) == 0]

    def build_card(m: dict, idx: int) -> str:
        title = esc(m.get("title", "Unknown"))
        mid = esc(m.get("id", ""))
        mtype = m.get("manuscript_type", "Document")
        url = m.get("url", "")
        abstract = m.get("abstract", "")
        assessment = m.get("anticipation_assessment", "")
        raw_snippet = m.get("snippet", "")
        if abstract:
            display_text = abstract
        elif assessment:
            display_text = assessment
        elif raw_snippet and raw_snippet.count("…") <= 1:
            display_text = raw_snippet
        else:
            display_text = ""
        display_text = esc(display_text)
        snippet_short = display_text[:200] + "..." if len(display_text) > 200 else display_text
        snippet = display_text

        evals = m.get("similarity_categories", m.get("evaluations", {}))
        matched = []
        unmatched = []
        md_lines = []

        def _cl_text(item) -> str:
            if isinstance(item, dict):
                return item.get("criterion", str(item))
            return str(item)

        def resolve_req(key: str) -> str:
            k = str(key).strip()
            # Strip leading "N." or "N. [w=X.XX]" prefix and resolve to checklist
            k_clean = re.sub(r'^\[w=[\d.]+\]\s*', '', k)
            m_num = re.match(r"^(?:item\s*)?(\d+)\.?\s*(?:\[w=[\d.]+\]\s*)?(.*)$", k_clean, re.IGNORECASE)
            if m_num:
                idx = int(m_num.group(1)) - 1
                if 0 <= idx < len(checklist):
                    return _cl_text(checklist[idx])
                # If index out of range but there's trailing text, return it cleaned
                trailing = m_num.group(2).strip()
                if trailing:
                    return trailing
            # Fallback: strip any [w=...] from the raw key
            return re.sub(r'\[w=[\d.]+\]\s*', '', k).strip()

        partial = []
        for req, ev in (evals or {}).items():
            if not isinstance(ev, dict):
                continue
            score_val = ev.get("score")
            is_match = ev.get("match", False)
            analysis = esc(ev.get("analysis", ""))
            full_req = _clean_req_text(resolve_req(req))
            short = esc(shorten_checklist(full_req))
            if score_val is not None:
                if score_val >= 2:
                    matched.append((short, full_req, analysis, 2))
                elif score_val == 1:
                    partial.append((short, full_req, analysis, 1))
                else:
                    unmatched.append((short, full_req, analysis, 0))
                label = {2: "PRESENT", 1: "PARTIAL", 0: "ABSENT"}.get(score_val, "?")
                md_lines.append(f"- [{label}] **{esc(full_req)}**: {analysis}")
            else:
                if is_match:
                    matched.append((short, full_req, analysis, 2))
                else:
                    unmatched.append((short, full_req, analysis, 0))
                md_lines.append(f"- [{'MATCH' if is_match else 'NO MATCH'}] **{esc(full_req)}**: {analysis}")

        hit_count = len(matched) + len(partial)
        total = len(evals) if evals else len(checklist)
        css_val = m.get("css", 0) or 0
        ewss_val = m.get("ewss", 0) or 0
        adj_ewss = m.get("adjusted_ewss", ewss_val) or ewss_val
        ewss_denom = m.get("ewss_denom_count", 0)
        score_num = adj_ewss if adj_ewss else (
            hit_count / total if total > 0 else 0)
        low_sample = ewss_denom > 0 and ewss_denom < 3

        hover_html = ""
        if matched or partial:
            hover_html = '<div class="hover-hits">'
            for short, full, analysis, *_ in matched:
                hover_html += f'<div class="hv-item"><span class="hv-icon hv-present">&#x25C9;</span> {esc(full)}</div>'
            for short, full, analysis, *_ in partial:
                hover_html += f'<div class="hv-item"><span class="hv-icon hv-partial">&#x25D1;</span> {esc(full)}</div>'
            hover_html += '</div>'

        detail_html = '<div class="detail-panel">'
        if matched:
            detail_html += '<div class="detail-group"><div class="detail-label">Present (score=2)</div>'
            for short, full, analysis, *_ in matched:
                detail_html += f'<div class="d-item d-hit"><span class="d-icon">&#x25C9;</span><div><div class="d-req">{esc(full)}</div>'
                if analysis:
                    detail_html += f'<div class="d-analysis">{analysis}</div>'
                detail_html += '</div></div>'
            detail_html += '</div>'
        if partial:
            detail_html += '<div class="detail-group"><div class="detail-label">Partial (score=1)</div>'
            for short, full, analysis, *_ in partial:
                detail_html += f'<div class="d-item d-partial"><span class="d-icon">&#x25D1;</span><div><div class="d-req">{esc(full)}</div>'
                if analysis:
                    detail_html += f'<div class="d-analysis">{analysis}</div>'
                detail_html += '</div></div>'
            detail_html += '</div>'
        if unmatched:
            detail_html += '<div class="detail-group"><div class="detail-label">Absent (score=0)</div>'
            for short, full, analysis, *_ in unmatched:
                detail_html += f'<div class="d-item d-miss"><span class="d-icon">&#x25CB;</span><div><div class="d-req">{esc(full)}</div>'
                if analysis:
                    detail_html += f'<div class="d-analysis">{analysis}</div>'
                detail_html += '</div></div>'
            detail_html += '</div>'
        detail_html += '</div>'

        # Markdown
        md = f"# {title}\n- **ID**: {mid}\n- **Type**: {mtype}\n- **Matched**: {hit_count}/{total}\n"
        if url:
            md += f"- **URL**: {url}\n"
        md += "\n## Evaluation\n" + "\n".join(md_lines)
        md_esc = esc(md).replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

        badge_type = "Patent" if mtype == "Patent" else "Paper"
        card_cls = "card card-hit" if hit_count > 0 else "card card-none"
        eval_source = m.get("eval_source", "pdf")
        abstract_only = (eval_source == "abstract")
        source_badge = ('<span class="src-badge src-abs" '
                        'title="PDF unavailable — evaluated against abstract/snippet only. '
                        'Consider verifying with full text.">abstract only</span>'
                        if abstract_only else '')

        filing = esc(m.get("filing_date", ""))
        grant = esc(m.get("grant_date", ""))
        inventor = esc(m.get("inventor", ""))
        assignee = esc(m.get("assignee", ""))
        authors_str = esc(m.get("authors", ""))
        year = m.get("year", "")
        patent_link = m.get("patent_link", "")

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

        pdf_direct = m.get("pdf_link", "") or ""
        if isinstance(pdf_direct, list):
            pdf_direct = pdf_direct[0] if pdf_direct else ""
        source_url = patent_link if (patent_link and badge_type == "Patent") else url
        download_btn = (
            f'<a class="dl-btn" href="{esc(pdf_direct)}" target="_blank" download '
            f'title="Download PDF" onclick="event.stopPropagation()">{_DL_ICON_SVG}</a>'
            if pdf_direct else
            f'<span class="dl-btn dl-disabled" title="No direct PDF available — expand card for source link">{_NO_PDF_ICON_SVG}</span>'
        )
        title_el = (
            f'<a class="card-title" href="{esc(source_url)}" '
            f'target="_blank" onclick="event.stopPropagation()">{title}</a>'
            if source_url else f'<div class="card-title">{title}</div>'
        )
        self_badge = ('<span class="src-badge src-self" title="This is your source paper indexed online">your paper</span>'
                      if m.get("is_self_match") else "")

        return f'''
<div class="{card_cls}" data-hits="{hit_count}" data-md="{md_esc}">
  <div class="card-top" onclick="toggle(this.parentElement)">
    <div class="card-left">
      <span class="type-dot {'dot-patent' if badge_type == 'Patent' else 'dot-paper'}"></span>
      <div class="card-title-wrap">
        {title_el}
        <div class="card-id">{subtitle} {source_badge} {self_badge}</div>
      </div>
    </div>
    <div class="card-right">
      {download_btn}
      <button class="md-btn" onclick="event.stopPropagation();showMd(this.closest(\'.card\'))" title="View as Markdown">MD</button>
      {f'<span class="hit-pct" style="color:{score_color(score_num)}" title="CSS={css_val:.0%} EWSS={ewss_val:.0%}{" (low sample: " + str(ewss_denom) + " criteria)" if low_sample else ""}">{score_num:.0%}{"*" if low_sample else ""}</span>' if hit_count > 0 else '<span class="no-badge">&mdash;</span>'}
    </div>
  </div>
  {hover_html}
  {f'<div class="card-snippet" onclick="toggle(this.parentElement)"><span class="snip-short">{snippet_short}</span><span class="snip-full">{snippet}</span></div>' if snippet else ''}
  {detail_html}
</div>'''

    # Build cards
    hits_html = "".join(build_card(m, i) for i, m in enumerate(hits_docs))
    nohits_html = "".join(build_card(m, i) for i, m in enumerate(no_hits_docs))
    exact_find_html = "".join(build_card(m, i) for i, m in enumerate(exact_find))

    # Build checklist HTML with weight badges
    cl_html_items = ""
    for item in checklist:
        criterion, weight_html = _format_cl_item(item)
        cl_html_items += f'<li class="cl-item">{esc(criterion)}{weight_html}</li>'

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
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}},{{left:'\\\\(',right:'\\\\)',display:false}},{{left:'\\\\[',right:'\\\\]',display:true}}]}})"></script>
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
.hdr{{text-align:center;padding:1.5rem 0 2rem;border-bottom:1px solid var(--border);margin-bottom:2rem;position:relative}}
.hdr h1{{font-size:1.4rem;font-weight:700;color:var(--text)}}
.hdr .sub{{font-size:0.82rem;color:var(--text2);margin:0.3rem 0 1rem}}
.source-title{{font-size:0.95rem;color:var(--text);margin:0.5rem 0 0.2rem}}
.source-file{{font-size:0.75rem;color:var(--text2);font-family:"SF Mono",Monaco,monospace;margin-bottom:0.5rem}}
.pills{{display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap}}
.pill{{font-size:0.75rem;padding:0.25rem 0.7rem;border-radius:99px;background:#eef1f5;color:var(--text2)}}
.pill b{{color:var(--text)}}
.hdr-actions{{position:absolute;top:1.5rem;right:0}}
.hdr-actions .fbtn-accent{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.hdr-actions .fbtn-accent:hover{{background:#1d4ed8;border-color:#1d4ed8}}

/* Sections */
.sec{{margin:1.5rem 0;padding:1rem 1.25rem;border:1px solid var(--border);border-radius:10px;background:var(--card)}}
.sec-t{{font-size:0.95rem;font-weight:600;margin-bottom:0.5rem;color:var(--text)}}
.sec-t-lg{{font-size:1.15rem;font-weight:700;letter-spacing:-0.01em}}
.sec-count{{font-size:0.78rem;font-weight:400;color:var(--text2);margin-left:0.4rem}}
.sec-note{{font-size:0.8rem;color:var(--text2);margin-bottom:0.75rem;line-height:1.5;font-style:italic}}
.sec-b{{font-size:0.88rem;color:var(--text2);line-height:1.7}}
.sec-b p{{margin-bottom:0.5rem}}

/* Markdown rendered content */
.md-heading{{margin:1rem 0 0.4rem;color:var(--text)}}
h3.md-heading{{font-size:1rem;font-weight:600;border-bottom:1px solid var(--border);padding-bottom:0.3rem}}
h4.md-heading{{font-size:0.92rem;font-weight:600}}
h5.md-heading{{font-size:0.88rem;font-weight:600}}
.md-para{{margin-bottom:0.6rem;line-height:1.75;color:var(--text);font-size:0.9rem}}
.md-list{{padding-left:1.5rem;margin:0.4rem 0 0.8rem}}
.md-list li{{margin-bottom:0.4rem;line-height:1.65;font-size:0.9rem;color:var(--text)}}
.md-list li code,.md-para code{{background:#f1f5f9;padding:0.1rem 0.35rem;border-radius:3px;font-size:0.85em;color:#be185d}}

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
.cl-weight{{display:inline-block;font-size:0.68rem;padding:0.05rem 0.4rem;border-radius:99px;background:#eef1f5;color:var(--text2);margin-left:0.4rem;vertical-align:middle;font-weight:500}}
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
.card-left{{display:flex;align-items:center;gap:0.6rem;flex:1;min-width:0;max-width:calc(100% - 140px)}}
.card-title-wrap{{min-width:0;flex:1}}
.type-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.dot-patent{{background:var(--patent)}}
.dot-paper{{background:var(--paper)}}
.card-title{{font-size:0.88rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-decoration:none;color:var(--text);display:block}}
a.card-title:hover{{color:var(--accent);text-decoration:underline}}
.card-id{{font-size:0.72rem;color:var(--text2);font-family:"SF Mono",Monaco,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card-right{{display:flex;align-items:center;gap:0.5rem;flex-shrink:0}}
.hit-pct{{font-size:0.88rem;font-weight:700;white-space:nowrap}}
.no-badge{{color:var(--miss);font-size:0.78rem}}
.pdf-link{{font-size:0.72rem;color:var(--accent);text-decoration:none}}
.src-badge{{display:inline-block;font-size:0.65rem;padding:0.05rem 0.4rem;border-radius:99px;margin-left:0.35rem;vertical-align:middle;font-weight:500;letter-spacing:0.02em;text-transform:uppercase}}
.src-abs{{background:#fef3c7;color:#92400e;border:1px solid #fde68a}}
.src-self{{background:#e0e7ff;color:#3730a3;border:1px solid #c7d2fe}}
/* Download icon button */
.dl-btn{{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:6px;background:var(--accent);color:#fff;text-decoration:none;border:1px solid var(--accent);cursor:pointer;flex-shrink:0}}
.dl-btn:hover{{background:#1d4ed8;border-color:#1d4ed8}}
.dl-btn svg{{pointer-events:none}}
.dl-disabled{{background:#e5e7eb;color:#9ca3af;border-color:#e5e7eb;cursor:not-allowed}}
.dl-disabled:hover{{background:#e5e7eb;border-color:#e5e7eb}}
/* Exact Find section — source paper indexed online, shown separately */
.exact-find-sec{{border:2px solid #818cf8;background:#eef2ff;margin:1.5rem 0;padding:1rem 1.25rem;border-radius:10px}}
.exact-find-sec .sec-t{{color:#3730a3}}
.exact-find-sec .sec-note{{color:#4338ca;font-style:normal}}
/* Feedback */
.fb-sec{{background:#f8fafc;border-left:3px solid var(--accent);margin-top:2rem}}
.fb-row{{display:flex;gap:1rem;align-items:center;margin:0.6rem 0;flex-wrap:wrap}}
.fb-label{{font-size:0.82rem;color:var(--text2);min-width:5.5rem;font-weight:500}}
.fb-stars{{display:flex;gap:0.2rem}}
.fb-star{{background:none;border:none;font-size:1.4rem;color:#d1d5db;cursor:pointer;padding:0 0.1rem;line-height:1;transition:color 0.15s}}
.fb-star.on,.fb-star:hover,.fb-star:hover ~ .fb-star{{color:#fbbf24}}
.fb-stars.hov .fb-star{{color:#d1d5db}}
.fb-stars.hov .fb-star.hov,.fb-stars.hov .fb-star.hov ~ .fb-star{{color:#fbbf24}}
.fb-tags{{display:flex;gap:0.4rem;flex-wrap:wrap}}
.fb-tag{{font-size:0.78rem;padding:0.25rem 0.7rem;border-radius:99px;background:#eef1f5;color:var(--text2);cursor:pointer;display:inline-flex;align-items:center;gap:0.3rem;user-select:none}}
.fb-tag input{{margin:0}}
.fb-tag:has(input:checked){{background:#dbeafe;color:#1e40af}}
.fb-input{{width:100%;padding:0.6rem 0.8rem;border:1px solid var(--border);border-radius:8px;font-family:inherit;font-size:0.88rem;resize:vertical;background:#fff}}
.fb-input:focus{{outline:none;border-color:var(--accent)}}
.fb-actions{{display:flex;gap:0.8rem;align-items:center;margin-top:0.6rem}}
.fb-submit{{background:var(--accent);color:#fff;border:none;padding:0.5rem 1.2rem;border-radius:6px;font-size:0.85rem;cursor:pointer;font-weight:500}}
.fb-submit:hover{{background:#1d4ed8}}
.fb-submit:disabled{{opacity:0.5;cursor:not-allowed}}
.fb-status{{font-size:0.8rem;color:var(--text2)}}
.md-btn{{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:6px;font-size:0.68rem;font-weight:600;color:#fff;background:var(--accent);border:1px solid var(--accent);cursor:pointer;flex-shrink:0}}
.md-btn:hover{{background:#1d4ed8;border-color:#1d4ed8}}

/* Hover checklist — full lines, 300ms delay */
.hover-hits{{padding:0 1rem 0.5rem;opacity:0;max-height:0;overflow:hidden;transition:opacity .2s ease .3s,max-height .25s ease .3s}}
.card:hover .hover-hits{{opacity:1;max-height:500px}}
.card.open .hover-hits{{opacity:0;max-height:0;transition:none}}
.hv-item{{font-size:0.82rem;color:var(--text);padding:0.2rem 0;display:flex;align-items:flex-start;gap:0.3rem;line-height:1.5}}
.hv-icon{{flex-shrink:0}}
.hv-present{{color:#dc2626}}
.hv-partial{{color:#ea580c}}

.card-snippet{{font-size:0.82rem;color:var(--text2);padding:0.3rem 1rem 0.6rem;line-height:1.55;border-top:1px solid var(--border);cursor:pointer}}
.snip-full{{display:none}}
.card:hover .snip-short{{display:none}}
.card:hover .snip-full{{display:inline}}
.card.open .snip-short{{display:none}}
.card.open .snip-full{{display:inline}}

/* Detail panel (click to expand) */
.detail-panel{{display:none;padding:0.5rem 1rem 1rem;border-top:1px solid var(--border);max-height:60vh;overflow-y:auto}}
.card.open .detail-panel{{display:block}}
.detail-group{{margin-bottom:0.75rem}}
.detail-label{{font-size:0.75rem;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:0.03em;margin-bottom:0.35rem}}
.d-item{{display:flex;gap:0.4rem;padding:0.3rem 0;align-items:flex-start}}
.d-icon{{flex-shrink:0;font-size:0.8rem}}
.d-hit .d-icon{{color:#dc2626}}
.d-partial .d-icon{{color:#ea580c}}
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
  <div class="hdr-actions">
    <div class="dropdown">
      <button class="fbtn fbtn-accent" onclick="event.stopPropagation();this.parentElement.classList.toggle('open')">Export &#x25BE;</button>
      <div class="dropdown-menu">
        <button onclick="copyFullMd('all')" data-orig="Copy full report">Copy full report</button>
        <button onclick="copyFullMd('hits')" data-orig="Copy matches only">Copy matches only</button>
        <hr>
        <button onclick="dlFullMd('all')">Download .md (full)</button>
        <button onclick="dlFullMd('hits')">Download .md (matches only)</button>
      </div>
    </div>
  </div>
  <h1>Prior Art Search Report</h1>
  {f'<div class="source-title">Source manuscript: <b>{esc(source_title)}</b></div>' if source_title else ''}
  {f'<div class="source-file">File: {esc(source_filename)}</div>' if source_filename else ''}
  <div class="sub">Possible matches for professional review &mdash; not a legal determination</div>
  <div class="pills">
    <span class="pill"><b>{esc(doc_mode.title())}</b> &middot; {esc(invention_type)}</span>
    <span class="pill">Searched <b>{total_patents}</b> patents + <b>{total_papers}</b> papers</span>
    <span class="pill"><b>{len(hits_docs)}</b> possible matches</span>
  </div>
</div>

{_render_confidence_banner(data.get("entropy_profile"))}

<div class="sec">
  <div class="sec-t">Invention Summary</div>
  <div class="sec-b">{format_summary(summary_text)}</div>
</div>

{f'<div class="sec eval-sec"><div class="sec-t sec-t-lg">Novelty Assessment</div><div class="sec-b">{render_markdown(overall_summary)}</div></div>' if overall_summary else ''}

<div class="sec checklist-sec">
  <div class="sec-t sec-t-lg">Evaluation Checklist <span class="sec-count">{len(checklist)} items</span></div>
  <div class="sec-note">Each item is an atomic, testable requirement derived from the invention disclosure. Prior art is evaluated against every item.</div>
  <ol class="cl-list">
    {cl_html_items}
  </ol>
</div>

{f"""<div class="sec">
  <div class="sec-t tog" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">Search Groups ({len(search_groups)})</div>
  <div class="tog-body">{sg_html}</div>
</div>""" if sg_html else ''}

{f"""<div class="exact-find-sec">
  <div class="sec-t sec-t-lg">Exact Find <span class="sec-count">{len(exact_find)} found</span></div>
  <div class="sec-note">Your source paper was indexed online and retrieved by the search. Listed here for awareness; excluded from the novelty ranking below.</div>
  {exact_find_html}
</div>""" if exact_find else ''}

<div class="bar">
  <div class="bar-title">Possible Matches</div>
  <div class="bar-actions">
    <button class="fbtn" onclick="showAll(this)">All ({len(hits_docs) + len(no_hits_docs)})</button>
    <button class="fbtn on" onclick="showHits(this)">With Overlap ({len(hits_docs)})</button>
    <button class="fbtn" onclick="showNone(this)">No Overlap ({len(no_hits_docs)})</button>
  </div>
</div>
<div class="bar-note">Hover for matched points &middot; Click to expand full evaluation</div>

<div id="hits">{hits_html}</div>
<div id="nohits" style="display:none">{nohits_html}</div>

<div class="sec fb-sec">
  <div class="sec-t sec-t-lg">Your feedback</div>
  <div class="sec-note">Help us refine the pipeline. We pair your feedback with the run's event log for offline analysis.</div>
  <div class="fb-row">
    <span class="fb-label">Quality</span>
    <div class="fb-stars" id="fbStars">
      <button class="fb-star" data-r="1">&#9733;</button>
      <button class="fb-star" data-r="2">&#9733;</button>
      <button class="fb-star" data-r="3">&#9733;</button>
      <button class="fb-star" data-r="4">&#9733;</button>
      <button class="fb-star" data-r="5">&#9733;</button>
    </div>
  </div>
  <div class="fb-row">
    <span class="fb-label">Issues (optional)</span>
    <div class="fb-tags" id="fbTags">
      <label class="fb-tag"><input type="checkbox" value="hallucinated_prior_art"> Hallucinated prior art</label>
      <label class="fb-tag"><input type="checkbox" value="missed_prior_art"> Missed obvious prior art</label>
      <label class="fb-tag"><input type="checkbox" value="wrong_scope"> Wrong checklist scope</label>
      <label class="fb-tag"><input type="checkbox" value="self_match"> Source leaked as prior art</label>
      <label class="fb-tag"><input type="checkbox" value="paywalled"> Key docs paywalled</label>
      <label class="fb-tag"><input type="checkbox" value="other"> Other</label>
    </div>
  </div>
  <textarea id="fbComment" class="fb-input" rows="4" placeholder="What went well / what went wrong. Paste quotes, cite specific cards. This feeds the next refine cycle."></textarea>
  <div class="fb-actions">
    <button class="fb-submit" onclick="submitFeedback()">Submit feedback</button>
    <span class="fb-status" id="fbStatus"></span>
  </div>
</div>

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
function toggle(c){{
  const wasOpen=c.classList.contains('open');
  document.querySelectorAll('.card.open').forEach(x=>x.classList.remove('open'));
  if(!wasOpen)c.classList.add('open');
}}
document.addEventListener('click',e=>{{
  if(!e.target.closest('.card')&&!e.target.closest('.overlay'))
    document.querySelectorAll('.card.open').forEach(x=>x.classList.remove('open'));
}});
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

// ── Feedback capture ──
const _JOB_ID={json.dumps(job_id)};
let _fbRating=0;
document.querySelectorAll('#fbStars .fb-star').forEach(btn=>{{
  btn.addEventListener('mouseenter',()=>{{
    const r=+btn.dataset.r;
    document.querySelectorAll('#fbStars .fb-star').forEach(b=>b.classList.toggle('hov',+b.dataset.r<=r));
    document.getElementById('fbStars').classList.add('hov');
  }});
  btn.addEventListener('mouseleave',()=>document.getElementById('fbStars').classList.remove('hov'));
  btn.addEventListener('click',()=>{{
    _fbRating=+btn.dataset.r;
    document.querySelectorAll('#fbStars .fb-star').forEach(b=>b.classList.toggle('on',+b.dataset.r<=_fbRating));
  }});
}});

async function submitFeedback(){{
  const btn=document.querySelector('.fb-submit');
  const st=document.getElementById('fbStatus');
  if(!_JOB_ID){{st.textContent='No job_id in report.';st.style.color='#b91c1c';return;}}
  const comment=document.getElementById('fbComment').value.trim();
  const tags=[...document.querySelectorAll('#fbTags input:checked')].map(i=>i.value);
  if(!_fbRating && !comment && tags.length===0){{st.textContent='Pick a rating or write something first.';st.style.color='#b91c1c';return;}}
  btn.disabled=true;st.textContent='Sending...';st.style.color='';
  try{{
    const isFile=location.protocol==='file:';
    const base=isFile?'http://localhost:8000':'';
    const prefix=isFile?'':'/api';
    const url=`${{base}}${{prefix}}/feedback/${{_JOB_ID}}`;
    const r=await fetch(url,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{rating:_fbRating||null,comment,tags}})}});
    if(!r.ok)throw new Error(`HTTP ${{r.status}}`);
    const j=await r.json();
    st.textContent=`Thanks — recorded (${{j.total_feedback||1}} total for this job).`;
    st.style.color='#065f46';
    document.getElementById('fbComment').value='';
    document.querySelectorAll('#fbTags input:checked').forEach(i=>i.checked=false);
  }}catch(e){{
    st.textContent='Send failed: '+e.message;
    st.style.color='#b91c1c';
  }}finally{{btn.disabled=false}}
}}
</script>
</body></html>'''


def generate_markdown(data: dict) -> str:
    if data.get("implied_invention"):
        return _generate_implied_markdown(data)

    phase1 = data.get("phase1", {})
    phase2 = data.get("phase2", {})
    evaluation = data.get("evaluation", {})
    search = data.get("search", {})

    source_title = data.get("source_title", data.get("source_filename", "Document"))
    summary_text = phase1.get("summary", "")
    invention_type = phase1.get("invention_type", "Unknown")
    doc_mode = phase1.get("doc_mode", "unknown")
    fields_map = phase1.get("fields_map", [])
    cpc = phase1.get("cpc_subclass", "")
    gen_at = data.get("generated_at", "")

    checklist = phase2.get("checklist", [])
    scoring_report = evaluation.get("scoring_report", [])
    overall_summary = evaluation.get("summary", "")
    stats = evaluation.get("stats", {})
    risk_level = stats.get("risk_level", "unknown")
    top_score = stats.get("top_score", 0)

    total_patents = search.get("summary", {}).get("total_patents", 0)
    total_papers = search.get("summary", {}).get("total_papers", 0)

    exact_find = [m for m in scoring_report if m.get("is_self_match")]
    non_self = [m for m in scoring_report if not m.get("is_self_match")]
    hits_docs = [m for m in non_self if (m.get("similarity_score", 0) or 0) > 0]

    lines = []
    lines.append(f"# Patent Novelty Report: {source_title}")
    lines.append("")
    lines.append(f"**Generated:** {gen_at}  ")
    lines.append(f"**Category:** {invention_type} · **Type:** {doc_mode}  ")
    if fields_map:
        lines.append(f"**Fields:** {', '.join(fields_map)}  ")
    if cpc:
        lines.append(f"**CPC:** {cpc}  ")
    lines.append(f"**Risk Level:** {risk_level} · **Top Score:** {top_score:.2%}  ")
    lines.append(f"**Prior Art Found:** {total_patents} patents, {total_papers} papers  ")
    lines.append("")

    ep = data.get("entropy_profile")
    if ep:
        conf = ep.get("overall_confidence", "medium").upper()
        lines.append(f"**Analysis Confidence:** {conf}  ")
        for dp in ep.get("degradation_points", []):
            lines.append(f"- ⚠ {dp}")
        lines.append("")

    lines.append("## Invention Summary")
    lines.append("")
    lines.append(summary_text)
    lines.append("")

    if overall_summary:
        lines.append("## Novelty Assessment")
        lines.append("")
        lines.append(overall_summary)
        lines.append("")

    if checklist:
        lines.append("## Evaluation Criteria")
        lines.append("")
        for i, item in enumerate(checklist, 1):
            if isinstance(item, dict):
                w = item.get("weight", 0)
                lines.append(f"{i}. [w={w:.2f}] {item.get('criterion', str(item))}")
            else:
                lines.append(f"{i}. {item}")
        lines.append("")

    if exact_find:
        lines.append("## Exact Find (Source Self-Match)")
        lines.append("")
        for m in exact_find:
            title = m.get("title", "Unknown")
            url = m.get("url", "")
            lines.append(f"- **{title}**" + (f" — [link]({url})" if url else ""))
        lines.append("")

    if hits_docs:
        lines.append("## Prior Art Matches (Final Reference Table)")
        lines.append("")
        has_css = any(m.get("css") is not None for m in hits_docs)
        if has_css:
            lines.append("| # | Title | Type | CSS | EWSS | RS Synopsis |")
            lines.append("|---|-------|------|-----|------|-------------|")
        else:
            lines.append("| # | Title | Type | Matched | Score |")
            lines.append("|---|-------|------|---------|-------|")
        for i, m in enumerate(hits_docs, 1):
            title = m.get("title", "Unknown")[:80]
            mtype = m.get("manuscript_type", "Doc")
            if has_css:
                css_v = m.get("css", 0) or 0
                ewss_v = m.get("ewss", 0) or 0
                rs_syn = (m.get("rs_synopsis", "") or "")[:100]
                lines.append(
                    f"| {i} | {title} | {mtype} | "
                    f"{css_v:.0%} | {ewss_v:.0%} | {rs_syn} |")
            else:
                evals = m.get("similarity_categories", m.get("evaluations", {}))
                hit_count = sum(1 for ev in (evals or {}).values()
                               if isinstance(ev, dict) and ev.get("match"))
                total = len(evals) if evals else len(checklist)
                score = hit_count / total if total > 0 else 0
                lines.append(
                    f"| {i} | {title} | {mtype} | "
                    f"{hit_count}/{total} | {score:.0%} |")
        lines.append("")

        lines.append("### Detailed Evaluations")
        lines.append("")
        for m in hits_docs:
            title = m.get("title", "Unknown")
            url = m.get("url", "")
            mtype = m.get("manuscript_type", "Document")
            evals = m.get("similarity_categories", m.get("evaluations", {}))
            css_v = m.get("css", 0) or 0
            ewss_v = m.get("ewss", 0) or 0
            lines.append(f"#### {title}")
            if url:
                lines.append(f"**URL:** {url}  ")
            lines.append(f"**Type:** {mtype}")
            if css_v or ewss_v:
                lines.append(f"**CSS:** {css_v:.0%} · **EWSS:** {ewss_v:.0%}")
            lines.append("")
            for req, ev in (evals or {}).items():
                if not isinstance(ev, dict):
                    continue
                score_val = ev.get("score")
                analysis = ev.get("analysis", "")
                if score_val is not None:
                    label = {2: "PRESENT", 1: "PARTIAL", 0: "ABSENT"}.get(score_val, "?")
                else:
                    label = "MATCH" if ev.get("match", False) else "NO MATCH"
                lines.append(f"- **[{label}]** {req}")
                if analysis:
                    lines.append(f"  - {analysis}")
            lines.append("")

    lines.append("---")
    lines.append("*Report generated by Patent Novelty Analyzer*")
    return "\n".join(lines)


def _generate_implied_markdown(data: dict) -> str:
    phase1 = data.get("phase1", {})
    phase2 = data.get("phase2", {})
    title = data.get("source_title", data.get("source_filename", "Document"))
    summary = phase1.get("summary", "")
    reasoning = phase1.get("reasoning", "")
    ucd = phase2.get("ucd", "")
    fields = phase1.get("fields_map", [])
    gen_at = data.get("generated_at", "")

    lines = [
        f"# Implied Invention Report: {title}",
        "",
        f"**Generated:** {gen_at}  ",
        f"**Fields:** {', '.join(fields) if fields else 'Not classified'}  ",
        "**Status:** Implied Invention",
        "",
        "> **Implied Invention** — This document describes an inventive concept but does not "
        "provide concrete implementation details. No prior art search was conducted.",
        "",
        "## Why \"Implied\"?",
        "",
        reasoning,
        "",
        "## Invention Summary",
        "",
        summary,
        "",
        "## Technical Elements Identified",
        "",
        f"```\n{ucd if ucd else '(No decomposition available)'}\n```",
        "",
        "## Recommendation",
        "",
        "To proceed with a prior art search, the document would need to include concrete "
        "implementation details — specific architectures, algorithms, materials, experimental "
        "results, or system designs.",
        "",
        "---",
        "*Report generated by Patent Novelty Analyzer*",
    ]
    return "\n".join(lines)


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
