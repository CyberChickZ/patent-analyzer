#!/usr/bin/env python3
"""Coverage eval (product question 2, evaluator ceiling).

Given the examiner-cited prior art D1 (oracle retrieval), can our evaluator
cover the features the examiner marked as disclosed — with a verbatim
quote that actually exists in D1? Also reports where the verified quote
landed relative to the paragraphs the examiner cited.

Checklist variants: oracle (examiner features) isolates the evaluator;
regex (our claim-1 splitter) is what serves in claim mode.

Usage:
    python3 evals/coverage_eval.py --limit 20 --checklist oracle --doc_mode full_text
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common import disclosed_features, format_cited, load_app, load_fixture

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "s4"
_PARA = re.compile(r"\[(\d{4})\]")


def build_checklist(app_data: dict, kind: str) -> list[dict]:
    if kind == "oracle":
        feats = [d["feature"] for d in disclosed_features(app_data)]
    else:
        from nodes.claim_mode import _parse_claim_limitations, _split_preamble
        claim1 = (app_data["rejected_patent"].get("claims") or [""])[0]
        parsed = _parse_claim_limitations(claim1)
        feats = _split_preamble(parsed.get("preamble", "")) + parsed["limitations"]
    feats = [f for f in feats if len(f) > 15]
    return [{"id": f"c{i + 1}", "criterion": f, "weight": 1.0 / max(len(feats), 1)}
            for i, f in enumerate(feats)]


def quote_paragraph(quote: str, doc: str) -> int | None:
    """Index of the numbered paragraph the (normalized) quote sits in."""
    from patent_analyzer.quote_verify import normalize
    q = normalize(quote)
    if len(q) < 10:
        return None
    cur = None
    for line in doc.split("\n"):
        m = _PARA.match(line)
        if m:
            cur = int(m.group(1))
        if q[:60] in normalize(line):
            return cur
    return None


async def run_one(app: str, checklist_kind: str, doc_mode: str) -> dict:
    from app.llm import evaluate_single_document_text
    from patent_analyzer.quote_verify import verify_checklist_results

    out_path = RUN_DIR / f"{app}_{checklist_kind}_{doc_mode}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    data = load_app(app)
    checklist = build_checklist(data, checklist_kind)
    claim1 = (data["rejected_patent"].get("claims") or [""])[0]
    doc = format_cited(data["cited_patent"])
    res = await evaluate_single_document_text(
        claim1, checklist, doc, data["cited_patent"].get("title") or "", "Patent",
        doc_mode=doc_mode)
    cr = res.get("checklist_results", {})
    stats = verify_checklist_results(cr, doc) if doc_mode == "full_text" else {}
    for item in cr.values():
        if isinstance(item, dict) and item.get("evidence_quote"):
            item["quote_paragraph"] = quote_paragraph(item["evidence_quote"], doc)
    result = {"app": app, "checklist_kind": checklist_kind, "doc_mode": doc_mode,
              "checklist": checklist, "checklist_results": cr, "verify": stats,
              "source": res.get("source")}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def score(result: dict, app_data: dict) -> dict:
    """Feature-level coverage against examiner-disclosed features."""
    from extraction_eval import embed, greedy_match

    gold = disclosed_features(app_data)
    cr = result["checklist_results"]
    crit = [c["criterion"] for c in result["checklist"]]
    if not gold or not crit:
        return {"gold": len(gold), "covered": 0, "covered_verified": 0, "para_hit": 0, "para_eval": 0,
                "key_match": 0, "n_crit": len(crit)}
    # align our criteria to gold features (identity for oracle, embedding for regex)
    if result["checklist_kind"] == "oracle":
        pairs = [(i, i) for i in range(len(gold))]
    else:
        sim = embed([g["feature"] for g in gold]) @ embed(crit).T
        pairs = [(i, j) for i, j, s in greedy_match(sim) if s >= 0.7]
    key_match = sum(1 for c in crit if c in cr)
    covered = covered_verified = para_hit = para_eval = 0
    for gi, cj in pairs:
        item = cr.get(crit[cj]) or {}
        sc = item.get("score")
        if sc is None:
            sc = 2 if item.get("match") else 0
        raw_positive = sc > 0 or item.get("quote_unverified")
        if raw_positive:
            covered += 1
        if sc > 0:
            covered_verified += 1
            qp = item.get("quote_paragraph")
            if gold[gi]["paragraphs"] and qp is not None:
                para_eval += 1
                para_hit += int(qp in gold[gi]["paragraphs"])
    return {"gold": len(gold), "covered": covered, "covered_verified": covered_verified,
            "para_hit": para_hit, "para_eval": para_eval, "key_match": key_match, "n_crit": len(crit)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--checklist", default="oracle", choices=["oracle", "regex", "both"])
    ap.add_argument("--doc_mode", default="full_text", choices=["full_text", "abstract", "both"])
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    import llm_cache
    llm_cache.install()
    apps = load_fixture()["stage"][:args.limit]
    kinds = ["oracle", "regex"] if args.checklist == "both" else [args.checklist]
    modes = ["full_text", "abstract"] if args.doc_mode == "both" else [args.doc_mode]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(app, kind, mode):
        async with sem:
            try:
                return await run_one(app, kind, mode)
            except Exception as exc:
                print(f"[{app}/{kind}/{mode}] FAILED {type(exc).__name__}: {exc}")
                return None

    print(f"{'variant':<22}{'n':>4}{'gold':>6}{'cov_raw':>9}{'cov_verified':>14}{'para_hit':>10}{'key_match':>11}")
    for kind in kinds:
        for mode in modes:
            results = [r for r in await asyncio.gather(*(one(a, kind, mode) for a in apps)) if r]
            agg = {"gold": 0, "covered": 0, "covered_verified": 0, "para_hit": 0, "para_eval": 0,
                   "key_match": 0, "n_crit": 0}
            for r in results:
                for k, v in score(r, load_app(r["app"])).items():
                    agg[k] += v
            g = agg["gold"] or 1
            print(f"{kind + '/' + mode:<22}{len(results):>4}{agg['gold']:>6}"
                  f"{agg['covered'] / g:>9.3f}{agg['covered_verified'] / g:>14.3f}"
                  f"{(agg['para_hit'] / agg['para_eval']) if agg['para_eval'] else 0:>10.3f}"
                  f"{agg['key_match'] / (agg['n_crit'] or 1):>11.3f}")
    print("\n" + llm_cache.summary())


if __name__ == "__main__":
    asyncio.run(main())
