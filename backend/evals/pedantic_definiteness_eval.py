#!/usr/bin/env python3
"""Draft gate 2: indefiniteness rate of the drafted claims, with the detector
calibrated on PEDANTIC first.

Detector D = (a) patent_analyzer.draft.definiteness rules (antecedent basis,
relative terms, exemplary phrasing, 112(f)) and (b) app.llm.definiteness_advisory
— the PEDANTIC examination prompt (Knappich et al., arXiv 2505.21342;
github.com/boschresearch/pedantic-patentsemtech, MIT) with their category list
and likelihood expressions verbatim, run on Gemini instead of their served
model; their `get_claim` / `search_description` tools are replaced by the
parent claims and the description inline. A claim counts as indefinite when
p_indefinite >= --threshold (0.5, their convention) or, in the "rules+llm"
mode, when a rule flag stands.

  calibrate  N claims from the PEDANTIC test split (seed 42) -> P/R/F1 and
             AUROC against their binary labels, next to the paper's baselines
             (Logistic Regression F1 56.3 / AUROC 59.5; Qwen-2.5-72B + LR
             ensemble F1 58.8 / AUROC 60.3). D must beat the LR baseline to be
             used as a gate; otherwise the draft numbers are reported as
             numbers, not as a gate.
  draft      pool every claim of the draft runs written by evals/draft_eval.py
             (and any --runs dir), sample --limit, report the rule-flag rate
             before / after the node's reword pass and D's indefinite rate,
             with the A2 pre-search draft as the comparison column.

Usage:
    python3 evals/pedantic_definiteness_eval.py calibrate --limit 100 --max-live-calls 120
    python3 evals/pedantic_definiteness_eval.py draft --limit 100
"""

import argparse
import asyncio
import json
import pickle
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common import load_env_yaml

DATA = Path(__file__).parent.parent / "eval_data" / "pedantic" / "dataset.pkl"
DRAFT_RUNS = Path(__file__).parent.parent / "eval_data" / "runs" / "draft_eval"
OUT_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "pedantic_draft"
LR_BASELINE = {"f1": 0.563, "auroc": 0.595}          # paper Table 3, Logistic Regression
ENSEMBLE_BASELINE = {"f1": 0.588, "auroc": 0.603}    # Qwen-2.5-72B + LR


def _shim():
    """Minimal `pedantic` package so their pickled Sample objects load without
    their dependencies (pyrootutils / thefuzz / bm25s)."""
    if "pedantic" in sys.modules:
        return
    import datetime
    import types
    from enum import Enum
    from typing import Literal

    from pydantic import BaseModel

    pkg = types.ModuleType("pedantic")
    dc = types.ModuleType("pedantic.dataset_creation")
    rs = types.ModuleType("pedantic.dataset_creation.rejection_schema")
    dt = types.ModuleType("pedantic.datatypes")

    class IndefinitenessCategory(BaseModel):
        key: str
        description: str = ""
        example: str = ""

    from patent_analyzer.draft.pedantic_categories import CATEGORY_KEYS
    keys = CATEGORY_KEYS + ["dependence", "other"]
    rs.indefiniteness_categories = [IndefinitenessCategory(key=k) for k in keys]
    rs.IndefinitenessCategoryEnum = Enum("IndefinitenessCategoryEnum", {k: k.replace("_", " ") for k in keys})
    rs.IndefinitenessCategory = IndefinitenessCategory

    class ClaimRecitationOccurrence(BaseModel):
        start: int
        stop: int
        text: str

    class ClaimRecitation(BaseModel):
        cited_text: str
        occurrences: list[ClaimRecitationOccurrence]

    class RejectionReason(BaseModel):
        usc_code: str
        category: rs.IndefinitenessCategoryEnum
        reason_text: str
        claim_recitations: list[ClaimRecitation]
        p_indefinite: float
        model_config = {"use_enum_values": True}

    class ClaimExamination(BaseModel):
        claim: int
        rejection_reasons: list[RejectionReason]
        p_indefinite: float

    class Claim(BaseModel):
        number: int
        text: str
        parents: list[int]
        independent: bool
        canceled: bool

    class Section(BaseModel):
        heading: str
        paragraphs: list[str]

    class Application(BaseModel):
        number: str
        claims: dict[int, Claim]
        specification: list[Section]
        rejections: dict[int, ClaimExamination]
        dataset_split: Literal["train", "test", "val"]
        filing_date: datetime.datetime

    class Sample(BaseModel):
        application: Application
        claim_number: int

    for name, obj in (("ClaimRecitationOccurrence", ClaimRecitationOccurrence), ("ClaimRecitation", ClaimRecitation),
                      ("RejectionReason", RejectionReason), ("ClaimExamination", ClaimExamination), ("Claim", Claim),
                      ("Section", Section), ("Application", Application), ("Sample", Sample)):
        setattr(dt, name, obj)
    dt.IndefinitenessCategoryEnum = rs.IndefinitenessCategoryEnum
    sys.modules.update({"pedantic": pkg, "pedantic.dataset_creation": dc,
                        "pedantic.dataset_creation.rejection_schema": rs, "pedantic.datatypes": dt})
    pkg.datatypes, pkg.dataset_creation = dt, dc


def load_pedantic(split: str = "test", limit: int = 100, seed: int = 42) -> list[dict]:
    """[{id, claim_text, parents[text], description, label, gold_categories}]."""
    _shim()
    samples = pickle.loads(DATA.read_bytes())
    ids = sorted(sid for sid, s in samples.items() if s.application.dataset_split == split)
    random.Random(seed).shuffle(ids)
    out = []
    for sid in ids[:limit]:
        s = samples[sid]
        app, no = s.application, s.claim_number
        claim = app.claims[no]
        rej = app.rejections.get(no)
        out.append({"id": sid, "claim_no": no, "claim_text": claim.text, "independent": claim.independent,
                    "parents": [{"no": p, "text": app.claims[p].text} for p in claim.parents if p in app.claims],
                    "description": "\n\n".join(p for sec in app.specification for p in sec.paragraphs),
                    "label": 1 if rej else 0,
                    "gold_categories": sorted({r.category if isinstance(r.category, str) else r.category.value
                                               for r in (rej.rejection_reasons if rej else [])})})
    return out


def rule_flags(claim_text: str, parents: list[dict]) -> list[dict]:
    """The draft's rule check over a raw claim string (parents seed the antecedent set)."""
    from nodes.claim_mode import _parse_claim_limitations
    from patent_analyzer.draft import definiteness as D

    def _claim(text, no):
        parsed = _parse_claim_limitations(text)
        lims = parsed["limitations"] or [text]
        return {"no": no, "preamble": re.sub(r"^\s*\d+\s*\.\s*", "", parsed["preamble"] or ""),
                "limitations": [{"lid": f"c{no}.l{i}", "text": t, "basis": []} for i, t in enumerate(lims, 1)]}
    ps = [_claim(p["text"], p["no"]) for p in parents]
    return D.check(_claim(claim_text, 0), ps)


async def detect(rows: list[dict], threshold: float = 0.5, use_llm: bool = True, batch: int = 4) -> list[dict]:
    from app.llm import definiteness_advisory
    out = []
    for i in range(0, len(rows), batch if not rows or rows[0].get("_single") else batch):
        chunk = rows[i:i + batch]
        for r in chunk:
            flags = rule_flags(r["claim_text"], r.get("parents") or [])
            rec = {"id": r["id"], "label": r.get("label"), "rule_flags": [{"category": f["category"], "span": f["span"]} for f in flags],
                   "rule_hit": bool(flags), "p_llm": None, "llm_categories": []}
            if use_llm:
                claims = [{"no": p["no"], "text": p["text"], "depends_on": None} for p in r.get("parents") or []]
                claims.append({"no": r.get("claim_no", 1), "text": r["claim_text"],
                               "depends_on": (r["parents"][-1]["no"] if r.get("parents") else None)})
                got = await definiteness_advisory(claims, r.get("description", ""))
                a = got.get(r.get("claim_no", 1)) or {}
                rec["p_llm"] = a.get("p_indefinite")
                rec["llm_categories"] = sorted({x.get("category") for x in a.get("reasons") or []})
            rec["pred_llm"] = int((rec["p_llm"] or 0.0) >= threshold)
            rec["pred_rules"] = int(rec["rule_hit"])
            rec["pred_union"] = int(rec["pred_llm"] or rec["pred_rules"])
            out.append(rec)
            print(f"  {rec['id']:<16} label={rec['label']} p_llm={rec['p_llm']} rules={[f['category'] for f in rec['rule_flags']]}", flush=True)
    return out


def prf(preds: list[int], labels: list[int]) -> dict:
    tp = sum(1 for p, l in zip(preds, labels) if p and l)
    fp = sum(1 for p, l in zip(preds, labels) if p and not l)
    fn = sum(1 for p, l in zip(preds, labels) if not p and l)
    tn = sum(1 for p, l in zip(preds, labels) if not p and not l)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "accuracy": (tp + tn) / max(1, len(labels)), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auroc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


async def cmd_calibrate(args):
    rows = load_pedantic("test", args.limit)
    n_pos = sum(r["label"] for r in rows)
    print(f"PEDANTIC test, seed 42: n={len(rows)} ({n_pos} indefinite / {len(rows) - n_pos} definite), "
          f"{sum(1 for r in rows if r['independent'])} independent")
    recs = await detect(rows, threshold=args.threshold, use_llm=not args.rules_only)
    labels = [r["label"] for r in recs]
    print(f"\n{'detector':<22}{'P':>7}{'R':>7}{'F1':>7}{'acc':>7}{'AUROC':>8}")
    for name, key in (("rules only", "pred_rules"), ("LLM examiner", "pred_llm"), ("rules OR LLM", "pred_union")):
        m = prf([r[key] for r in recs], labels)
        a = auroc([float(r["p_llm"] or 0.0) for r in recs], labels) if key == "pred_llm" else (
            auroc([float(max(r["p_llm"] or 0.0, 0.6 if r["rule_hit"] else 0.0)) for r in recs], labels) if key == "pred_union" else
            auroc([1.0 if r["rule_hit"] else 0.0 for r in recs], labels))
        print(f"{name:<22}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{m['accuracy']:>7.3f}{(a if a is not None else float('nan')):>8.3f}")
    print(f"{'paper LR baseline':<22}{'':>7}{'':>7}{LR_BASELINE['f1']:>7.3f}{'':>7}{LR_BASELINE['auroc']:>8.3f}")
    print(f"{'paper 72B+LR ensemble':<22}{'':>7}{'':>7}{ENSEMBLE_BASELINE['f1']:>7.3f}{'':>7}{ENSEMBLE_BASELINE['auroc']:>8.3f}")
    # per-category: does a rule fire where the examiner cited that category?
    print("\nrule category vs examiner-cited category (on the labelled claims)")
    for cat in ("antecedent_basis", "relative_term", "exemplary_phrasing", "functional_claiming"):
        gold = [1 if cat.replace("_", " ") in [g.replace("_", " ") for g in r0["gold_categories"]] else 0
                for r0 in rows]
        pred = [1 if any(f["category"] == cat for f in r["rule_flags"]) else 0 for r in recs]
        m = prf(pred, gold)
        print(f"  {cat:<22} gold={sum(gold):>3}  fired={sum(pred):>3}  P={m['precision']:.3f} R={m['recall']:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"calibration_{len(recs)}.json").write_text(json.dumps({"rows": recs, "n": len(recs)}, indent=1))


def draft_claim_pool(runs_dir: Path) -> list[dict]:
    """Every claim of every draft run: {id, claim_text, parents, description, source}."""
    from patent_analyzer.draft.assemble import render_claim
    out = []
    for f in sorted(runs_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        draft = rec.get("draft") or {}
        by_no = {c.get("no"): c for c in draft.get("claims") or []}
        for c in draft.get("claims") or []:
            parents = []
            d = c.get("depends_on")
            while d is not None and d in by_no:
                parents.insert(0, {"no": d, "text": render_claim(by_no[d])})
                d = by_no[d].get("depends_on")
            out.append({"id": f"{rec.get('publication_number', f.stem)}-c{c.get('no')}", "claim_no": c.get("no"),
                        "claim_text": render_claim(c), "parents": parents, "description": "",
                        "rule_text_claim": {**c, "limitations": [{**l, "text": l.get("rule_text") or l.get("text")} for l in c.get("limitations") or []]},
                        "open_flags": [f_ for f_ in (draft.get("definiteness") or {}).get("open_flags") or [] if str(f_.get("lid", "")).startswith(f"c{c.get('no')}.")],
                        "source": rec.get("publication_number", f.stem), "independent": c.get("depends_on") is None})
        core = next((x for x in ((rec.get("extraction") or {}).get("candidate_inventions") or []) if x.get("level") == "core"), None) or {}
        a2 = (core.get("independent_claim_draft") or {}).get(core.get("primary_form") or "method") or ""
        if a2:
            out.append({"id": f"{rec.get('publication_number', f.stem)}-a2", "claim_no": 1, "claim_text": a2, "parents": [],
                        "description": "", "source": rec.get("publication_number", f.stem), "independent": True, "is_a2": True})
    return out


async def cmd_draft(args):
    from patent_analyzer.draft import definiteness as D
    pool = draft_claim_pool(Path(args.runs) if args.runs else DRAFT_RUNS)
    drafted = [r for r in pool if not r.get("is_a2")]
    a2 = [r for r in pool if r.get("is_a2")]
    if not drafted:
        print(f"no draft runs in {args.runs or DRAFT_RUNS}; run evals/draft_eval.py first")
        return
    random.Random(42).shuffle(drafted)
    sample = drafted[:args.limit]
    print(f"draft claims: {len(drafted)} from {len({r['source'] for r in drafted})} documents; sampling {len(sample)}; "
          f"A2 comparison claims: {len(a2)}")
    # rule flags before the node's reword pass (rule_text) and after (text)
    before = after = 0
    cats_after = {}
    for r in sample:
        rc = r.get("rule_text_claim")
        if rc:
            before += bool(D.check(rc, [{"limitations": [{"text": p["text"], "basis": []}], "preamble": ""} for p in r["parents"]]))
        flags = rule_flags(r["claim_text"], r["parents"])
        after += bool(flags)
        for f in flags:
            cats_after[f["category"]] = cats_after.get(f["category"], 0) + 1
    print(f"rule flag rate: before reword {before}/{len(sample)} = {before / len(sample):.3f}; after {after}/{len(sample)} = {after / len(sample):.3f}")
    print(f"remaining rule categories: {cats_after or 'none'}")
    print(f"open flags recorded by the node: {sum(len(r.get('open_flags') or []) for r in sample)}")
    if args.rules_only:
        return
    recs = await detect(sample, threshold=args.threshold)
    ind = sum(r["pred_llm"] for r in recs)
    print(f"\nD (LLM examiner, p >= {args.threshold}) indefinite: {ind}/{len(recs)} = {ind / len(recs):.3f}")
    print(f"D (rules OR LLM) indefinite: {sum(r['pred_union'] for r in recs)}/{len(recs)} = {sum(r['pred_union'] for r in recs) / len(recs):.3f}")
    cats = {}
    for r in recs:
        for c in r["llm_categories"]:
            cats[c] = cats.get(c, 0) + 1
    print(f"LLM categories: {dict(sorted(cats.items(), key=lambda kv: -kv[1]))}")
    if a2:
        a2_recs = await detect(a2[:args.limit], threshold=args.threshold)
        ia = sum(r["pred_llm"] for r in a2_recs)
        ra = sum(r["pred_rules"] for r in a2_recs)
        print(f"A2 pre-search draft: D indefinite {ia}/{len(a2_recs)} = {ia / len(a2_recs):.3f}; rule flags {ra}/{len(a2_recs)} = {ra / len(a2_recs):.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "draft_claims.json").write_text(json.dumps({"rows": recs}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--limit", type=int, default=100)
    c.add_argument("--threshold", type=float, default=0.5)
    c.add_argument("--rules-only", action="store_true")
    c.add_argument("--max-live-calls", type=int, default=150)
    d = sub.add_parser("draft")
    d.add_argument("--limit", type=int, default=100)
    d.add_argument("--threshold", type=float, default=0.5)
    d.add_argument("--runs", default="")
    d.add_argument("--rules-only", action="store_true")
    d.add_argument("--max-live-calls", type=int, default=150)
    args = ap.parse_args()
    load_env_yaml()
    from draft_eval import install_budget
    install_budget(args.max_live_calls)
    asyncio.run(cmd_calibrate(args) if args.cmd == "calibrate" else cmd_draft(args))


if __name__ == "__main__":
    main()
