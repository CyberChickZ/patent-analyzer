#!/usr/bin/env python3
"""Stage 1 eval: full document -> IDCA -> SSR checklist, scored against the
examiner's feature breakdown of claim 1 (FiNE-Patents).

This is the real product scenario: the pipeline never sees claim 1, only the
rendered description (desc_only) — or, as an upper bound, the description
plus claims (with_claims). The regex claim-1 splitter (extraction_eval.py)
is the reference column.

Usage:
    python3 evals/extraction_fulldoc_eval.py --limit 20 --mode desc_only
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common import examiner_features, load_app, load_fixture, render_doc
from extraction_eval import embed, greedy_match

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "s1"
THRESHOLDS = (0.7, 0.8)


async def run_stage1(app: str, with_claims: bool) -> dict:
    """IDCA + SSR on a rendered FiNE application; cached per (app, mode)."""
    from graph.ssr_subgraph import build_ssr_subgraph
    from nodes.idca import idca_node

    mode = "with_claims" if with_claims else "desc_only"
    out_path = RUN_DIR / f"{app}_{mode}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    data = load_app(app)
    text = render_doc(data["rejected_patent"], with_claims=with_claims)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp = f.name

    p1 = await idca_node({"input_local_path": tmp})
    result = {
        "app": app, "mode": mode,
        "status_determination": p1.get("status_determination"),
        "doc_type": p1.get("doc_type"), "input_mode": p1.get("input_mode"),
        "summary": p1.get("summary", ""), "checklist": [], "delegation": {},
        "retry_count": 0,
    }
    if p1.get("status_determination") == "Present":
        sg = build_ssr_subgraph()
        p2 = await sg.ainvoke({
            "summary": p1["summary"], "fields_map": p1.get("fields_map", []),
            "cpc_subclass": p1.get("cpc_subclass", ""), "personas": p1.get("personas", {}),
        })
        result["checklist"] = p2.get("checklist", [])
        result["delegation"] = p2.get("delegation", {})
        result["retry_count"] = p2.get("retry_count", 0)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def score(gold: list[str], preds: list[str]) -> dict:
    if not preds:
        return {t: (0, len(gold), 0) for t in THRESHOLDS}
    sim = embed(gold) @ embed(preds).T
    matches = greedy_match(sim)
    return {t: (sum(1 for _, _, s in matches if s >= t), len(gold), len(preds)) for t in THRESHOLDS}


def limitation_coverage(summary: str, claim1: str, tau: float = 0.75) -> float:
    """Share of claim-1 limitations whose meaning survives into the IDCA summary."""
    from nodes.claim_mode import _parse_claim_limitations
    lims = [l for l in _parse_claim_limitations(claim1)["limitations"] if len(l) > 15]
    sents = [s.strip() for s in summary.replace("\n", " ").split(". ") if len(s.strip()) > 20]
    if not lims or not sents:
        return 0.0
    sim = embed(lims) @ embed(sents).T
    return float((sim.max(axis=1) >= tau).mean())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--mode", default="desc_only", choices=["desc_only", "with_claims", "both"])
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    import llm_cache
    llm_cache.install()

    apps = load_fixture()["stage"][:args.limit]
    modes = ["desc_only", "with_claims"] if args.mode == "both" else [args.mode]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(app, mode):
        async with sem:
            try:
                return await run_stage1(app, with_claims=(mode == "with_claims"))
            except Exception as exc:
                print(f"[{app}/{mode}] FAILED {type(exc).__name__}: {exc}")
                return None

    for mode in modes:
        results = [r for r in await asyncio.gather(*(one(a, mode) for a in apps)) if r]
        agg = {t: [0, 0, 0] for t in THRESHOLDS}
        cov, gran, present, retries = [], [], 0, 0
        for r in results:
            data = load_app(r["app"])
            gold = examiner_features(data)
            preds = [c.get("criterion", "") for c in r["checklist"] if c.get("criterion")]
            for t, (tp, ng, npred) in score(gold, preds).items():
                agg[t][0] += tp; agg[t][1] += ng; agg[t][2] += npred
            if r["status_determination"] == "Present":
                present += 1
            retries += 1 if r.get("retry_count") else 0
            claim1 = (data["rejected_patent"].get("claims") or [""])[0]
            if r["summary"] and claim1:
                cov.append(limitation_coverage(r["summary"], claim1))
            if gold:
                gran.append(len(preds) / len(gold))

        print(f"\n== mode={mode}  n={len(results)}  Present={present}/{len(results)}  "
              f"SSR retries={retries}  granularity(pred/gold)={np.mean(gran):.2f}  "
              f"limitation_coverage@0.75={np.mean(cov) if cov else 0:.3f}")
        print(f"{'threshold':<12}{'recall':>8}{'precision':>11}{'F1':>7}")
        for t in THRESHOLDS:
            tp, ng, npred = agg[t]
            rec = tp / ng if ng else 0.0
            prec = tp / npred if npred else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            print(f"{t:<12}{rec:>8.3f}{prec:>11.3f}{f1:>7.3f}")
    print("\n" + llm_cache.summary())


if __name__ == "__main__":
    asyncio.run(main())
