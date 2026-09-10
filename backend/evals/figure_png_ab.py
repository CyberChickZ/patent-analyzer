"""A/B: does the PNG bypass in the deep read buy anything?

`evaluate_single_document` attaches the prior-art PDF *and*, separately, every
page of it that looks like it has a figure, re-rendered at 150 dpi as a PNG
image part. The pages were therefore sent twice: once
inside the PDF part, once as pixels. N4 measured the second copy at 2.03x the
tokens of the first for zero additional figure numerals on one document. This
runs the same comparison on the real population -- prior-art PDFs the pipeline
downloaded on earlier jobs, evaluated against that job's own checklist -- so
the decision to drop the bypass rests on more than one document.

Arm A: the pipeline as it stands (PDF + PNG image parts).
Arm B: the PDF alone.

Reported per arm: which criteria matched, whether the evidence quotes survive
verification against the PDF's text layer, tokens and dollars. B is scored
against A: a criterion A found and B missed is what the PNGs would be buying.

  PYTHONPATH=$(pwd) python3 evals/figure_png_ab.py --n 24 --out /tmp/fig_ab.json

Scope note for any F1 read off this: the quote check can only verify evidence
that exists in the text layer. Features disclosed only in a drawing (11.15% of
examiner-cited features, E-series) are outside what either arm can be scored
on here, and outside the F1.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.common import load_env_yaml

load_env_yaml()


# The renderer this A/B exists to justify deleting. It lived in app/llm.py as
# `_render_figure_pages` until 2026-09-18; the copy is kept here, verbatim, so
# the comparison stays reproducible after the pipeline stopped doing it.
def render_figure_pages(path: str, dpi: int = 150, max_screenshots: int = 10) -> list[tuple[int, bytes]]:
    import fitz
    try:
        doc = fitz.open(path)
    except Exception:
        return []
    try:
        figure_indices = []
        for i, page in enumerate(doc):
            try:
                images = page.get_images(full=True)
                if any(img[2] > 100 and img[3] > 100 for img in images):
                    figure_indices.append(i)
                    continue
                if len(page.get_drawings()) > 50:
                    figure_indices.append(i)
            except Exception:
                pass
        results = []
        for i in figure_indices[:max_screenshots]:
            try:
                pix = doc[i].get_pixmap(dpi=dpi)
                results.append((i + 1, pix.tobytes("png")))
            except Exception:
                pass
        return results
    finally:
        doc.close()


def collect(job_root: Path, n: int, min_figs: int) -> list[dict]:
    """(pdf, checklist, summary) triples from finished jobs on this box.

    Only documents whose PDF actually renders figure pages are useful: where
    the renderer returns nothing the two arms are the same call.
    """
    out = []
    for job_dir in sorted(job_root.glob("*/")):
        res = job_dir / "results.json"
        pdfs = sorted(job_dir.glob("prior_art_*.pdf"))
        if not res.exists() or not pdfs:
            continue
        try:
            data = json.loads(res.read_text())
        except Exception:
            continue
        checklist = (data.get("phase2") or {}).get("checklist") or []
        summary = (data.get("phase1") or {}).get("summary") or ""
        if not checklist or not summary:
            continue
        for pdf in pdfs:
            figs = render_figure_pages(str(pdf))
            if len(figs) < min_figs:
                continue
            out.append({"job": job_dir.name, "pdf": str(pdf), "n_fig_pages": len(figs),
                        "fig_bytes": sum(len(b) for _, b in figs),
                        "checklist": checklist, "summary": summary})
            if len(out) >= n:
                return out
    return out


def _positives(result: dict) -> set[str]:
    out = set()
    for k, v in (result.get("checklist_results") or {}).items():
        if not isinstance(v, dict):
            continue
        if v.get("score") is not None:
            if int(v.get("score") or 0) >= 2:
                out.add(k)
        elif v.get("match"):
            out.add(k)
    return out


def _quote_survival(result: dict, pdf: str) -> tuple[int, int]:
    """(verified, total) evidence quotes, checked against the PDF text layer."""
    from patent_analyzer.quote_verify import pdf_text, verify_checklist_results
    text = pdf_text(pdf)
    if not text:
        return 0, 0
    qv = verify_checklist_results(result.get("checklist_results") or {}, text)
    return int(qv.get("verified", 0)), int(qv.get("quotes", 0))


async def run_arm(case: dict, with_figs: bool) -> dict:
    import app.llm as llm

    before = json.loads(json.dumps(llm.usage))
    figs = render_figure_pages(case["pdf"]) if with_figs else []
    real = llm.call_llm_with_pdfs

    async def with_images(system, user, pdf_paths, *a, **kw):
        kw["image_parts"] = [b for _, b in figs] or None
        return await real(system, user, pdf_paths, *a, **kw)

    if with_figs:
        llm.call_llm_with_pdfs = with_images
    t0 = time.monotonic()
    try:
        result = await llm.evaluate_single_document(
            case["summary"], case["checklist"], case["pdf"],
            Path(case["pdf"]).name, "Patent")
    except Exception as exc:
        result = {"checklist_results": {}, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        llm.call_llm_with_pdfs = real
    secs = time.monotonic() - t0

    from patent_analyzer.metering import cost_usd
    delta, cost = {}, 0.0
    for model, after in llm.usage.items():
        b = before.get(model) or {}
        d = {k: after.get(k, 0) - b.get(k, 0) for k in ("calls", "prompt_tokens", "output_tokens", "thought_tokens")}
        if d["calls"]:
            delta[model] = d
            cost += cost_usd(model, d["prompt_tokens"], d["output_tokens"], d["thought_tokens"])
    verified, quotes = _quote_survival(result, case["pdf"])
    return {"positives": sorted(_positives(result)), "n_criteria": len(result.get("checklist_results") or {}),
            "verified": verified, "quotes": quotes, "seconds": round(secs, 1),
            "usage": delta, "cost_usd": round(cost, 6), "error": result.get("error", "")}


# ── the direct question: does the PNG carry anything the PDF part does not? ──
#
# The A/B above measures the end effect, where the positive rate is a few
# percent and 24 documents cannot separate a small effect from noise. This asks
# the narrow question with a large per-document count instead: every reference
# numeral the model can read off the drawings. N4 ran it on one document (54
# numerals, identical both ways); this runs it on the same population.

_NUMERAL_SYS = "You read patent and paper figures. Output JSON only."
_NUMERAL_PROMPT = """List EVERY reference numeral that appears in the figures of the attached document,
with the part name each one labels. Read the drawings themselves, not only the
text. Do not invent numerals you cannot see.

JSON: {"numerals": [{"n": "112", "part": "hinge assembly"}, ...]}"""


def _numerals(resp: str) -> set[str]:
    import json as _json
    import re as _re
    m = _re.search(r"\{.*\}", resp or "", _re.DOTALL)
    if not m:
        return set()
    try:
        data = _json.loads(m.group())
    except Exception:
        return set()
    out = set()
    for row in data.get("numerals") or []:
        n = str((row or {}).get("n") if isinstance(row, dict) else row).strip()
        if n:
            out.add(n)
    return out


async def numeral_arm(case: dict, with_figs: bool) -> dict:
    import app.llm as llm
    from patent_analyzer.metering import cost_usd

    figs = render_figure_pages(case["pdf"]) if with_figs else []
    before = json.loads(json.dumps(llm.usage))
    try:
        resp = await llm.call_llm_with_pdfs(
            _NUMERAL_SYS, _NUMERAL_PROMPT, [case["pdf"]], thinking_budget=0,
            image_parts=[b for _, b in figs] or None, model=llm.stage_model("screen"))
    except Exception as exc:
        return {"numerals": [], "error": f"{type(exc).__name__}: {exc}", "prompt_tokens": 0, "cost_usd": 0.0}
    cost, ptok = 0.0, 0
    for model, after in llm.usage.items():
        bfr = before.get(model) or {}
        d = {k: after.get(k, 0) - bfr.get(k, 0) for k in ("calls", "prompt_tokens", "output_tokens", "thought_tokens")}
        if d["calls"]:
            ptok += d["prompt_tokens"]
            cost += cost_usd(model, d["prompt_tokens"], d["output_tokens"], d["thought_tokens"])
    return {"numerals": sorted(_numerals(resp)), "prompt_tokens": ptok, "cost_usd": round(cost, 6), "error": ""}


async def run_numerals(args):
    cases = collect(Path(args.jobs), args.n, max(args.min_figs, 1))
    print(f"{len(cases)} documents")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i, case):
        async with sem:
            a = await numeral_arm(case, True)
            b = await numeral_arm(case, False)
        sa, sb = set(a["numerals"]), set(b["numerals"])
        print(f"[{i:02d}] {Path(case['pdf']).name[:24]:24s} figs={case['n_fig_pages']:2d} "
              f"A={len(sa):3d} B={len(sb):3d} onlyA={len(sa - sb):3d} onlyB={len(sb - sa):3d} "
              f"tok {a['prompt_tokens']}/{b['prompt_tokens']} ${a['cost_usd']:.4f}/${b['cost_usd']:.4f}")
        return {"i": i, "pdf": Path(case["pdf"]).name, "n_fig_pages": case["n_fig_pages"],
                "A": a, "B": b, "only_A": sorted(sa - sb), "only_B": sorted(sb - sa),
                "both": sorted(sa & sb)}

    rows = await asyncio.gather(*(one(i, c) for i, c in enumerate(cases)))
    both = sum(len(r["both"]) for r in rows)
    oa = sum(len(r["only_A"]) for r in rows)
    ob = sum(len(r["only_B"]) for r in rows)
    summary = {"mode": "numerals", "n_docs": len(rows),
               "numerals_A": both + oa, "numerals_B": both + ob, "both": both,
               "only_A": oa, "only_B": ob,
               "jaccard": round(both / max(both + oa + ob, 1), 4),
               "prompt_tokens_A": sum(r["A"]["prompt_tokens"] for r in rows),
               "prompt_tokens_B": sum(r["B"]["prompt_tokens"] for r in rows),
               "cost_A": round(sum(r["A"]["cost_usd"] for r in rows), 4),
               "cost_B": round(sum(r["B"]["cost_usd"] for r in rows), 4)}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--min-figs", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--jobs", default="/tmp/outputs")
    ap.add_argument("--out", default="/tmp/fig_png_ab.json")
    ap.add_argument("--mode", choices=("eval", "numerals"), default="eval")
    args = ap.parse_args()

    if args.mode == "numerals":
        await run_numerals(args)
        return

    cases = collect(Path(args.jobs), args.n, args.min_figs)
    print(f"{len(cases)} documents with >= {args.min_figs} figure page(s)")
    if not cases:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def one(i, case):
        async with sem:
            a = await run_arm(case, with_figs=True)
            b = await run_arm(case, with_figs=False)
        pa, pb = set(a["positives"]), set(b["positives"])
        row = {"i": i, "job": case["job"], "pdf": Path(case["pdf"]).name,
               "n_fig_pages": case["n_fig_pages"], "fig_bytes": case["fig_bytes"],
               "n_checklist": len(case["checklist"]), "A": a, "B": b,
               "only_A": sorted(pa - pb), "only_B": sorted(pb - pa), "both": sorted(pa & pb)}
        print(f"[{i:02d}] {row['pdf'][:28]:28s} figs={case['n_fig_pages']:2d} "
              f"A+{len(pa)} B+{len(pb)} onlyA={len(pa - pb)} onlyB={len(pb - pa)} "
              f"quotes A {a['verified']}/{a['quotes']} B {b['verified']}/{b['quotes']} "
              f"tok A {sum(d['prompt_tokens'] for d in a['usage'].values())} "
              f"B {sum(d['prompt_tokens'] for d in b['usage'].values())} "
              f"${a['cost_usd']:.4f}/${b['cost_usd']:.4f}")
        return row

    rows = await asyncio.gather(*(one(i, c) for i, c in enumerate(cases)))

    tp = sum(len(set(r["A"]["positives"]) & set(r["B"]["positives"])) for r in rows)
    fn = sum(len(r["only_A"]) for r in rows)      # A found it, B did not: what the PNGs buy
    fp = sum(len(r["only_B"]) for r in rows)      # B found it, A did not
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    def agg(arm, key):
        return sum(r[arm][key] for r in rows)

    def toks(arm):
        return sum(sum(d["prompt_tokens"] for d in r[arm]["usage"].values()) for r in rows)

    summary = {
        "n_docs": len(rows),
        "fig_pages_total": sum(r["n_fig_pages"] for r in rows),
        "positives_A": tp + fn, "positives_B": tp + fp,
        "only_A": fn, "only_B": fp,
        "feature_f1_B_vs_A": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
        "quotes_A": agg("A", "quotes"), "verified_A": agg("A", "verified"),
        "quotes_B": agg("B", "quotes"), "verified_B": agg("B", "verified"),
        "survival_A": round(agg("A", "verified") / max(agg("A", "quotes"), 1), 4),
        "survival_B": round(agg("B", "verified") / max(agg("B", "quotes"), 1), 4),
        "prompt_tokens_A": toks("A"), "prompt_tokens_B": toks("B"),
        "cost_A": round(agg("A", "cost_usd"), 4), "cost_B": round(agg("B", "cost_usd"), 4),
        "seconds_A": round(agg("A", "seconds"), 1), "seconds_B": round(agg("B", "seconds"), 1),
        "scope_note": ("F1 is over criteria the model reported; a feature disclosed only in a "
                       "drawing is outside what the text-layer quote check can score, and outside "
                       "this number (11.15% of examiner-cited features, E-series)."),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
