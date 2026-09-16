"""Re-render a report from a saved results.json — no pipeline, no LLM, no cost.

The point is to see a wording change against a real job instead of a fixture:
`python backend/evals/scratch_rerender_report.py results.json out.html out.md`.
Everything it needs is already in results.json, which is what nodes/report.py
hands to the generator.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patent_analyzer.report_generator import generate_html, generate_markdown
from patent_analyzer.report_sections import inject_html, inject_md


def main() -> None:
    src = Path(sys.argv[1])
    out_html = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".rendered.html")
    out_md = Path(sys.argv[3]) if len(sys.argv) > 3 else src.with_suffix(".rendered.md")
    r = json.loads(src.read_text())

    adj = r.get("adjudication") or {}
    ev = r.get("evaluation") or {}
    scoring = ev.get("scoring_report") or []
    checklist = (r.get("phase2") or {}).get("checklist") or []
    kw = dict(adjudication=adj, draft=r.get("draft_claims"), cost=r.get("cost"),
              read_gap=(r.get("eval_stats") or {}).get("read_gap") or {})
    summary = (r.get("search") or {}).get("summary") or {}
    ext = r.get("extraction") or {}

    out_html.write_text(inject_html(generate_html(r), ext, summary, scoring, checklist, **kw), encoding="utf-8")
    out_md.write_text(inject_md(generate_markdown(r), ext, summary, scoring, checklist, **kw), encoding="utf-8")
    print(f"{out_html}  {out_html.stat().st_size} B")
    print(f"{out_md}  {out_md.stat().st_size} B")


if __name__ == "__main__":
    main()
