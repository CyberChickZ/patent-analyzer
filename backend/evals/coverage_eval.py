#!/usr/bin/env python3
"""Coverage eval (product question 2, evaluator ceiling).

Given the examiner-cited prior art D1 (oracle retrieval), can our evaluator
cover the features the examiner marked as disclosed — with a verbatim
quote that actually exists in D1? Also reports where the verified quote
landed relative to the paragraphs the examiner cited.

Checklist variants: oracle (examiner features) isolates the evaluator;
regex (our claim-1 splitter) is what serves in claim mode.

Protocols:
  legacy  — feature coverage / verified coverage / para_hit (README table).
  fine    — FiNE-Patents Table 2 protocol: predicted features are aligned
            to examiner features by argmax edit similarity (many-to-one, no
            threshold), every verified quote is located to a passage
            (kind, number) and feature-level / claim-level passage P/R/F1
            are computed exactly as evaluate.py::compute_retrieval_metrics.
Baselines (no Gemini): --baseline rougeL | embed reproduce FiNE's
RougeSimilarity / EmbeddingSimilarity (top-5, tau 0.4 / 0.5).

Usage:
    python3 evals/coverage_eval.py --limit 20 --checklist oracle --doc_mode full_text
    python3 evals/coverage_eval.py --protocol fine --checklist both --limit 100
    python3 evals/coverage_eval.py --baseline rougeL --limit 100
"""

import argparse
import asyncio
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common import breakdown_features, disclosed_features, format_cited, load_app, load_fixture

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "s4"
_PARA = re.compile(r"\[(\d{4})\]")
BASELINE_TAU = {"rougeL": 0.4, "embed": 0.5}
BASELINE_TOPK = 5


def select_apps(limit: int, sample: str = "test") -> list[str]:
    """stage fixture ids; sample=test keeps FiNE test split with a rejected version."""
    apps = load_fixture()["stage"]
    if sample == "test":
        keep = []
        for a in apps:
            meta = load_app(a)["metadata"] or {}
            if meta.get("split") == "test" and "rejected" in (meta.get("include_versions") or []):
                keep.append(a)
        apps = keep
    return apps[:limit]


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


def cited_passages(patent: dict) -> list[tuple[str, int | None, str]]:
    """FiNE valid_references: non-empty paragraphs, non-empty claims, abstract."""
    refs = [("paragraph", i + 1, p) for i, p in enumerate(patent.get("description") or []) if p]
    refs += [("claim", i + 1, c) for i, c in enumerate(patent.get("claims") or []) if c]
    refs.append(("abstract", None, patent.get("abstract") or ""))
    return refs


def quote_location(quote: str, patent: dict) -> tuple[str, int | None] | None:
    """(kind, number) of the cited passage holding the quote. Exact normalized
    prefix match first (paragraphs, claims, abstract — FiNE's reference order),
    otherwise the best fuzzy locate_quote hit at the verifier's threshold."""
    from patent_analyzer.quote_verify import locate_quote, normalize
    q = normalize(quote)
    if len(q) < 10:
        return None
    refs = cited_passages(patent)
    head = q[:60]
    for kind, num, text in refs:
        if head in normalize(text):
            return (kind, num)
    best, best_ref = 0.0, None
    for kind, num, text in refs:
        found, sim = locate_quote(quote, text)
        if found and sim > best:
            best, best_ref = sim, (kind, num)
    return best_ref


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


async def run_one_multi(app: str, checklist_kind: str) -> dict:
    """Multi-quote variant (evals/eval_prompts.evaluate_multi_quote): up to 5
    verbatim quotes per criterion, no in-place downgrade — verification is
    done per quote at scoring time by quote_predictions()."""
    from eval_prompts import evaluate_multi_quote

    out_path = RUN_DIR / f"{app}_{checklist_kind}_full_text_multi.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    data = load_app(app)
    checklist = build_checklist(data, checklist_kind)
    claim1 = (data["rejected_patent"].get("claims") or [""])[0]
    doc = format_cited(data["cited_patent"])
    res = await evaluate_multi_quote(
        claim1, checklist, doc, data["cited_patent"].get("title") or "", "Patent")
    result = {"app": app, "checklist_kind": checklist_kind, "doc_mode": "full_text", "quotes": "multi",
              "checklist": checklist, "checklist_results": res.get("checklist_results", {}),
              "source": res.get("source"), "error": res.get("error")}
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
    # align our criteria to gold features (exact text for oracle, embedding for regex)
    if result["checklist_kind"] == "oracle":
        pairs = [(i, crit.index(g["feature"])) for i, g in enumerate(gold) if g["feature"] in crit]
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


# ---------------------------------------------------------------- FiNE protocol

def edit_similarity(string: str, substring: str) -> float:
    """Verbatim port of FiNE evaluate.py::edit_similarity."""
    sub = substring.lower()
    s = string.lower()
    if not sub:
        return 0.0
    sm = SequenceMatcher(None, sub, s, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    return matched / len(sub)


def _prf(pred: set, gold: set) -> tuple[float, float, float]:
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def fine_score(preds: list[tuple[str, set]], app_data: dict) -> dict:
    """FiNE compute_retrieval_metrics on (pred_feature, {(kind, number)}) pairs.

    Each predicted feature maps to the examiner feature with the highest
    edit_similarity (many-to-one, no threshold); its passages accumulate
    there. Feature-level P/R/F1 are per examiner feature with gold passages
    (others skipped), averaged within the sample. Claim-level compares the
    union of predicted passages with the union of gold passages."""
    universe = breakdown_features(app_data)
    mapped = [set() for _ in universe]
    for feat, passages in preds:
        if not universe:
            break
        sims = [(j, edit_similarity(u["feature"], feat)) for j, u in enumerate(universe)]
        best_j, _ = max(sims, key=lambda x: x[1])
        mapped[best_j] |= passages
    feat_rows = [_prf(pred, u["passages"]) for u, pred in zip(universe, mapped) if u["passages"]]
    gold_all = set().union(*(u["passages"] for u in universe)) if universe else set()
    pred_all = set().union(*(p for _, p in preds)) if preds else set()
    return {
        "feature": tuple(float(np.mean(col)) for col in zip(*feat_rows)) if feat_rows else None,
        "claim": _prf(pred_all, gold_all) if gold_all else None,
        "n_feat": len(feat_rows),
        "n_pred_passages": len(pred_all),
    }


def llm_predictions(result: dict, app_data: dict) -> list[tuple[str, set]]:
    """Checklist items with a verified quote -> the passage the quote sits in."""
    cr = result["checklist_results"]
    patent = app_data["cited_patent"]
    preds = []
    for c in result["checklist"]:
        item = cr.get(c["criterion"]) or {}
        sc = item.get("score")
        if sc is None:
            sc = 2 if item.get("match") else 0
        passages = set()
        if sc > 0 and item.get("evidence_quote"):
            loc = quote_location(item["evidence_quote"], patent)
            if loc:
                passages.add(loc)
        preds.append((c["criterion"], passages))
    return preds


def fine_split_features(claim: str) -> list[str]:
    """FiNE baselines' feature segmentation (baselines.py)."""
    return [f.strip() for f in re.split(r"[;\n]", claim) if len(f) > 20]


def baseline_predictions(app_data: dict, kind: str) -> list[tuple[str, set]]:
    """FiNE RougeSimilarity / EmbeddingSimilarity: score every feature against
    every passage, keep top-5 passages, mark those >= tau as disclosed."""
    claim1 = (app_data["rejected_patent"].get("claims") or [""])[0] or ""
    features = fine_split_features(claim1)
    refs = cited_passages(app_data["cited_patent"])
    passages = [t for _, _, t in refs]
    if not features or not passages:
        return [(f, set()) for f in features]
    if kind == "rougeL":
        from rouge_score import rouge_scorer
        rs = rouge_scorer.RougeScorer(["rougeL"])
        scores = np.array([[rs.score(prediction=p, target=f)["rougeL"].recall for p in passages]
                           for f in features])
    else:
        from extraction_eval import embed
        scores = embed(features) @ embed(passages).T
    tau = BASELINE_TAU[kind]
    preds = []
    for i, f in enumerate(features):
        top = np.argsort(-scores[i], kind="stable")[:BASELINE_TOPK]
        preds.append((f, {(refs[j][0], refs[j][1]) for j in top if scores[i, j] >= tau}))
    return preds


def aggregate_fine(rows: list[dict]) -> dict:
    feat = [r["feature"] for r in rows if r["feature"] is not None]
    claim = [r["claim"] for r in rows if r["claim"] is not None]
    mean3 = lambda xs: tuple(float(np.mean(col)) for col in zip(*xs)) if xs else (0.0, 0.0, 0.0)
    return {"n": len(rows), "n_feat": sum(r["n_feat"] for r in rows),
            "n_pred_passages": sum(r["n_pred_passages"] for r in rows),
            "feature": mean3(feat), "claim": mean3(claim)}


def _print_fine_header():
    print(f"{'variant':<22}{'n':>4}{'feats':>6}{'feat_P':>8}{'feat_R':>8}{'feat_F1':>8}"
          f"{'claim_P':>9}{'claim_R':>9}{'claim_F1':>9}{'pred_psg':>9}")


def _print_fine_row(name: str, agg: dict):
    fp, fr, ff = agg["feature"]
    cp, cr, cf = agg["claim"]
    print(f"{name:<22}{agg['n']:>4}{agg['n_feat']:>6}{fp:>8.3f}{fr:>8.3f}{ff:>8.3f}"
          f"{cp:>9.3f}{cr:>9.3f}{cf:>9.3f}{agg['n_pred_passages']:>9}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--checklist", default="oracle", choices=["oracle", "regex", "both"])
    ap.add_argument("--doc_mode", default="full_text", choices=["full_text", "abstract", "both"])
    ap.add_argument("--protocol", default="legacy", choices=["legacy", "fine"])
    ap.add_argument("--baseline", default=None, choices=["rougeL", "embed"])
    ap.add_argument("--sample", default="test", choices=["test", "stage"])
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--summary", default=None, help="append aggregate rows to this json")
    args = ap.parse_args()

    apps = select_apps(args.limit, args.sample)
    summary = {}

    if args.baseline:
        rows = [fine_score(baseline_predictions(load_app(a), args.baseline), load_app(a)) for a in apps]
        agg = aggregate_fine(rows)
        _print_fine_header()
        _print_fine_row(f"baseline/{args.baseline}", agg)
        summary[f"baseline/{args.baseline}"] = agg
    else:
        import llm_cache
        llm_cache.install()
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

        if args.protocol == "fine":
            _print_fine_header()
        else:
            print(f"{'variant':<22}{'n':>4}{'gold':>6}{'cov_raw':>9}{'cov_verified':>14}{'para_hit':>10}{'key_match':>11}")
        for kind in kinds:
            for mode in modes:
                results = [r for r in await asyncio.gather(*(one(a, kind, mode) for a in apps)) if r]
                name = f"{kind}/{mode}"
                if args.protocol == "fine":
                    rows = [fine_score(llm_predictions(r, load_app(r["app"])), load_app(r["app"])) for r in results]
                    agg = aggregate_fine(rows)
                    _print_fine_row(name, agg)
                    summary[name] = agg
                    continue
                agg = {"gold": 0, "covered": 0, "covered_verified": 0, "para_hit": 0, "para_eval": 0,
                       "key_match": 0, "n_crit": 0}
                for r in results:
                    for k, v in score(r, load_app(r["app"])).items():
                        agg[k] += v
                g = agg["gold"] or 1
                print(f"{name:<22}{len(results):>4}{agg['gold']:>6}"
                      f"{agg['covered'] / g:>9.3f}{agg['covered_verified'] / g:>14.3f}"
                      f"{(agg['para_hit'] / agg['para_eval']) if agg['para_eval'] else 0:>10.3f}"
                      f"{agg['key_match'] / (agg['n_crit'] or 1):>11.3f}")
        print("\n" + llm_cache.summary())

    if args.summary:
        p = Path(args.summary)
        prev = json.loads(p.read_text()) if p.exists() else {}
        prev.update({k: {**v, "apps": len(apps), "sample": args.sample} for k, v in summary.items()})
        p.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
