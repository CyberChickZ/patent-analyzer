#!/usr/bin/env python3
"""J4 (b): quote localization of the extraction subgraph per text layer.

Same PDF, same IDCA summary, three text layers for A1/A2/A3:
  docjson  production: state.document_text = rendered Gemini Doc JSON (+ fitz fallback in A3)
  fitz     legacy: no doc_json -> resolve_doc_text reads _extract_pdf_text(pdf)
  text     reference: Pap2Pat paper.json rendered (openworld_eval.render_paper), a .txt input

Per layer and paper: n_elements, n_unsupported (A3 snap_quote fails), Doc JSON /
fallback hits, LLM calls; element lists are matched across layers with the
te005 embedding + greedy 1:1 rule at 0.7. All Gemini calls go through
evals/llm_cache (IDCA + Doc JSON are shared by the layers).

Usage: python3 evals/pdf_layer_extraction.py [--pairs a,b] [--layers docjson,fitz,text]
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

PDF_DIR = Path(__file__).parent.parent / "eval_data" / "pdfs" / "j4"
RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "j4"
MATCH_TAU = 0.7


def _elements(p2: dict) -> list[dict]:
    out = []
    for c in (p2.get("extraction") or {}).get("candidate_inventions") or []:
        for e in c.get("elements") or []:
            out.append({"id": e.get("id"), "cand": c.get("id"), "level": c.get("level"), "text": e.get("text", ""),
                        "quote": e.get("evidence_quote", ""), "unsupported": bool(e.get("unsupported")),
                        "loc": e.get("evidence_loc")})
    return out


async def run_one(pdf: Path, layers: list[str]) -> dict:
    from graph.extraction_subgraph import build_extraction_subgraph
    from nodes.idca import idca_node
    from openworld_eval import render_paper

    out_path = RUN_DIR / "extraction" / f"{pdf.stem}.json"
    rec = json.loads(out_path.read_text()) if out_path.exists() else {"pair_id": pdf.stem, "layers": {}}
    todo = [l for l in layers if l not in rec["layers"]]
    if not todo:
        return rec

    p1 = await idca_node({"input_local_path": str(pdf)})
    rec["idca"] = {"status_determination": p1.get("status_determination"), "input_mode": p1.get("input_mode"),
                   "doc_json_stats": p1.get("doc_json_stats"), "summary": (p1.get("summary") or "")[:400]}
    if p1.get("status_determination") != "Present":
        rec["skipped"] = f"status={p1.get('status_determination')}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        return rec
    base = {"summary": p1["summary"], "input_mode": p1.get("input_mode", "academic_paper"),
            "cpc_subclass": p1.get("cpc_subclass", "")}
    g = build_extraction_subgraph()
    for layer in todo:
        if layer == "docjson":
            if not p1.get("doc_json"):
                rec["layers"][layer] = {"error": "no Doc JSON from IDCA"}
                continue
            state = {**base, "document_text": p1["document_text"], "doc_json": p1["doc_json"], "input_local_path": str(pdf)}
        elif layer == "fitz":
            state = {**base, "document_text": "", "doc_json": None, "input_local_path": str(pdf)}
        elif layer == "text":
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write(render_paper(pdf.stem))
                tmp = f.name
            state = {**base, "document_text": "", "doc_json": None, "input_local_path": tmp}
        else:
            continue
        p2 = await g.ainvoke(state)
        errs = p2.get("errors") or {}
        rec["layers"][layer] = {
            "n_elements": errs.get("n_elements"), "n_unsupported": errs.get("n_unsupported"),
            "quote_survival": errs.get("quote_survival"), "doc_json_hits": errs.get("doc_json_hits"),
            "fallback_hits": errs.get("fallback_hits"), "claim_ratio_min": errs.get("claim_ratio_min"),
            "llm_calls": errs.get("llm_calls"), "text_chars": len(p2.get("full_text") or ""),
            "n_candidates": len((p2.get("extraction") or {}).get("candidate_inventions") or []),
            "checklist": len(p2.get("checklist") or []),
            "self_check_events": [e["message"] for e in p2.get("events") or [] if "self_check" in e.get("kind", "")],
            "elements": _elements(p2)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def match_layers(rec: dict, a: str, b: str) -> dict | None:
    """Elements of layer a matched 1:1 to layer b (embedding cosine >= 0.7)."""
    import numpy as np
    from extraction_eval import embed, greedy_match
    ea = (rec["layers"].get(a) or {}).get("elements") or []
    eb = (rec["layers"].get(b) or {}).get("elements") or []
    if not ea or not eb:
        return None
    va, vb = embed([e["text"] for e in ea]), embed([e["text"] for e in eb])
    sim = va @ vb.T
    pairs = [(i, j, s) for i, j, s in greedy_match(sim) if s >= MATCH_TAU]
    only_a = [ea[i]["text"] for i in range(len(ea)) if i not in {p[0] for p in pairs}]
    only_b = [eb[j]["text"] for j in range(len(eb)) if j not in {p[1] for p in pairs}]
    return {"n_a": len(ea), "n_b": len(eb), "matched": len(pairs), "only_a": only_a, "only_b": only_b}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="")
    ap.add_argument("--layers", default="docjson,fitz")
    ap.add_argument("--no_match", action="store_true")
    from common import add_budget_arg, arm_budget
    add_budget_arg(ap)
    args = ap.parse_args()
    arm_budget(args)
    import llm_cache
    llm_cache.install()
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.pairs:
        keep = set(args.pairs.split(","))
        pdfs = [p for p in pdfs if p.stem in keep]
    layers = args.layers.split(",")

    recs = []
    for pdf in pdfs:
        rec = await run_one(pdf, layers)
        recs.append(rec)
        cells = []
        for l in layers:
            r = rec["layers"].get(l) or {}
            if not r or r.get("error"):
                cells.append(f"{l}: {r.get('error', '-')}")
                continue
            extra = f" (docjson {r['doc_json_hits']} + fallback {r['fallback_hits']})" if r.get("doc_json_hits") is not None else ""
            cells.append(f"{l}: {r['n_elements'] - r['n_unsupported']}/{r['n_elements']} located{extra}, calls={r['llm_calls']}")
        print(f"{pdf.stem}: " + " | ".join(cells) + (f"  [{rec['skipped']}]" if rec.get("skipped") else ""))
        print("   ", llm_cache.summary())

    if not args.no_match and len(layers) >= 2:
        for rec in recs:
            for a, b in [(layers[0], l) for l in layers[1:]]:
                m = match_layers(rec, a, b)
                if m:
                    rec.setdefault("match", {})[f"{a}~{b}"] = m
                    print(f"{rec['pair_id']} {a}~{b}: matched {m['matched']} of {m['n_a']}/{m['n_b']}; "
                          f"only in {a}: {len(m['only_a'])}, only in {b}: {len(m['only_b'])}")
            (RUN_DIR / "extraction" / f"{rec['pair_id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))

    tot = {}
    for l in layers:
        rs = [rec["layers"][l] for rec in recs if rec["layers"].get(l) and not rec["layers"][l].get("error")]
        n = sum(r["n_elements"] for r in rs)
        u = sum(r["n_unsupported"] for r in rs)
        dj = sum(r["doc_json_hits"] or 0 for r in rs)
        fb = sum(r["fallback_hits"] or 0 for r in rs)
        tot[l] = {"papers": len(rs), "elements": n, "located": n - u, "unsupported": u, "doc_json_hits": dj, "fallback_hits": fb}
    print("TOTAL", json.dumps(tot))
    (RUN_DIR / "extraction_summary.json").write_text(json.dumps(tot, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
