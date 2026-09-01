#!/usr/bin/env python3
"""Stage 1 eval: full document -> IDCA -> SSR checklist, scored against the
examiner's feature breakdown of claim 1 (FiNE-Patents).

This is the real product scenario: the pipeline never sees claim 1, only the
rendered description (desc_only) — or, as an upper bound, the description
plus claims (with_claims). The regex claim-1 splitter (extraction_eval.py)
is the reference column.

Usage:
    python3 evals/extraction_fulldoc_eval.py --limit 20 --mode desc_only

Phase 2 is IDCA -> graph.extraction_subgraph. Next to F1 it reports the
omission / fabrication / misclassification columns (evals/extraction_errors.py)
and the quote survival of the core candidate. (The legacy SSR extractor and its
--extractor flag were removed on 2026-09-18; cached runs from before that carry
"extractor": "ssr" and are no longer comparable.)
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

from common import examiner_features, load_app, load_fixture, model_tag, render_doc
from extraction_errors import classify_errors, error_rates
from extraction_eval import embed, greedy_match

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "s1"
THRESHOLDS = (0.7, 0.8)


class BudgetExceeded(RuntimeError):
    pass


def _live_calls() -> int:
    import llm_cache
    return llm_cache.stats["misses"]


async def run_stage1(app: str, with_claims: bool, extractor: str = "new", max_live_calls: int | None = None) -> dict:
    """IDCA + the extraction subgraph on a rendered FiNE application; cached per
    (app, mode, extractor)."""
    from nodes.idca import idca_node

    mode = "with_claims" if with_claims else "desc_only"
    suffix = "" if extractor == "ssr" else f"_{extractor}"
    out_path = RUN_DIR / f"{app}_{mode}{suffix}{model_tag()}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    if max_live_calls is not None and _live_calls() >= max_live_calls:
        raise BudgetExceeded(f"live LLM calls reached {max_live_calls}")

    data = load_app(app)
    text = render_doc(data["rejected_patent"], with_claims=with_claims)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp = f.name

    import llm_cache
    calls_before = llm_cache.stats["hits"] + llm_cache.stats["misses"]
    p1 = await idca_node({"input_local_path": tmp})
    result = {
        "app": app, "mode": mode, "extractor": extractor,
        "status_determination": p1.get("status_determination"),
        "doc_type": p1.get("doc_type"), "input_mode": p1.get("input_mode"),
        "summary": p1.get("summary", ""), "checklist": [], "delegation": {},
        "retry_count": 0, "extraction": None, "errors": None, "llm_calls": None,
    }
    if p1.get("status_determination") == "Present":
        from graph.extraction_subgraph import build_extraction_subgraph
        p2 = await build_extraction_subgraph().ainvoke({
            "summary": p1["summary"], "document_text": text, "input_local_path": tmp,
            "input_mode": p1.get("input_mode", "academic_paper"), "cpc_subclass": p1.get("cpc_subclass", ""),
        })
        result["checklist"] = p2.get("checklist", [])
        result["extraction"] = p2.get("extraction")
        result["errors"] = p2.get("errors")
        result["retry_count"] = p2.get("retry_count", 0)
        result["llm_calls"] = p2.get("llm_calls")
    result["llm_calls_total"] = llm_cache.stats["hits"] + llm_cache.stats["misses"] - calls_before
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def score(gold: list[str], preds: list[str]) -> dict:
    if not preds:
        return {t: (0, len(gold), 0) for t in THRESHOLDS}
    sim = embed(gold) @ embed(preds).T
    matches = greedy_match(sim)
    return {t: (sum(1 for _, _, s in matches if s >= t), len(gold), len(preds)) for t in THRESHOLDS}


def core_elements(result: dict) -> list[dict]:
    """Elements of the core candidate (all, incl. unsupported) for the error taxonomy."""
    cands = ((result.get("extraction") or {}).get("candidate_inventions") or [])
    core = next((c for c in cands if c.get("level") == "core"), cands[0] if cands else None)
    return [{"id": e["id"], "text": e["text"], "evidence_quote": e.get("evidence_quote", ""),
             "kind": e.get("kind"), "level": core.get("level")} for e in (core or {}).get("elements", [])]


def errors_for(result: dict, gold: list[str], doc_text: str) -> dict:
    if result.get("extractor", "ssr") == "new":
        return classify_errors(core_elements(result), gold, doc_text, require_quote=True)
    preds = [{"id": f"c{i}", "text": c.get("criterion", "")} for i, c in enumerate(result["checklist"]) if c.get("criterion")]
    return classify_errors(preds, gold, doc_text, require_quote=False)


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
    ap.add_argument("--extractor", default="new", choices=["new"],
                    help="kept so old command lines still parse; the SSR extractor is gone")
    ap.add_argument("--max-live-calls", type=int, default=60, help="stop launching new samples past this many live Gemini calls")
    ap.add_argument("--no-errors", action="store_true", help="skip the omission/fabrication/misclassification columns")
    args = ap.parse_args()

    import llm_cache
    llm_cache.install()

    apps = load_fixture()["stage"][:args.limit]
    modes = ["desc_only", "with_claims"] if args.mode == "both" else [args.mode]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(app, mode):
        async with sem:
            try:
                return await run_stage1(app, with_claims=(mode == "with_claims"), extractor=args.extractor,
                                        max_live_calls=args.max_live_calls)
            except Exception as exc:
                print(f"[{app}/{mode}] FAILED {type(exc).__name__}: {exc}")
                return None

    for mode in modes:
        results = [r for r in await asyncio.gather(*(one(a, mode) for a in apps)) if r]
        agg = {t: [0, 0, 0] for t in THRESHOLDS}
        cov, gran, present, retries = [], [], 0, 0
        err_agg = {"omission": [0, 0], "fabrication": [0, 0], "misclassification": [0, 0]}
        survival, calls, n_cands = [], [], []
        for r in results:
            data = load_app(r["app"])
            gold = examiner_features(data)
            preds = [c.get("criterion", "") for c in r["checklist"] if c.get("criterion")]
            for t, (tp, ng, npred) in score(gold, preds).items():
                agg[t][0] += tp; agg[t][1] += ng; agg[t][2] += npred
            if r["status_determination"] == "Present":
                present += 1
            retries += 1 if r.get("retry_count") else 0
            if r.get("llm_calls") is not None:
                calls.append(r["llm_calls"])
            elif r.get("llm_calls_total") is not None:
                calls.append(r["llm_calls_total"])
            if r.get("errors") and r["errors"].get("quote_survival") is not None:
                survival.append(r["errors"]["quote_survival"])
            if r.get("extraction"):
                n_cands.append(len(r["extraction"].get("candidate_inventions") or []))
            if not args.no_errors and gold and (preds or r.get("extraction")):
                doc_text = render_doc(data["rejected_patent"], with_claims=(mode == "with_claims"))
                err = errors_for(r, gold, doc_text)
                err_agg["omission"][0] += len(err["omission"]); err_agg["omission"][1] += err["n_gold"]
                err_agg["fabrication"][0] += len(err["fabrication"]); err_agg["fabrication"][1] += err["n_pred"]
                err_agg["misclassification"][0] += len(err["misclassification"]); err_agg["misclassification"][1] += len(err["matched"])
            claim1 = (data["rejected_patent"].get("claims") or [""])[0]
            if r["summary"] and claim1:
                cov.append(limitation_coverage(r["summary"], claim1))
            if gold:
                gran.append(len(preds) / len(gold))

        print(f"\n== mode={mode}  extractor={args.extractor}  n={len(results)}  Present={present}/{len(results)}  "
              f"retries={retries}  granularity(pred/gold)={np.mean(gran) if gran else 0:.2f}  "
              f"limitation_coverage@0.75={np.mean(cov) if cov else 0:.3f}")
        if calls:
            print(f"   llm calls/sample={np.mean(calls):.1f}  "
                  + (f"quote_survival={np.mean(survival):.3f}  " if survival else "")
                  + (f"candidates/sample={np.mean(n_cands):.1f}" if n_cands else ""))
        if not args.no_errors:
            def _rate(k):
                n, d = err_agg[k]
                return f"{n / d:.3f} ({n}/{d})" if d else "n/a"
            print(f"   errors: omission={_rate('omission')}  fabrication={_rate('fabrication')}  "
                  f"misclassification={_rate('misclassification')}")
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
