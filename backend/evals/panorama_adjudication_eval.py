#!/usr/bin/env python3
"""H2: does the determination stage reproduce US examiner §102 / §103 / ALLOW
calls when handed the examiner's own references?

Instances: evals/panorama_data.py (eval_data/runs/h2/samples.json — chain-
resolved claim text, label, examiner-cited references with their text).
Per instance: elements = nodes.claim_mode._parse_claim_limitations (plus the
preamble features unless --no-preamble); each cited reference is scored by
app.llm.evaluate_single_document_text(doc_mode="full_text"), the quotes are
verified against the reference text, and patent_analyzer.adjudicate turns
the verified coverage into a label. Stage 1 (per-document scoring) is cached
in eval_data/runs/h2/doc_results.json, so rule variants are free to re-score
(--variants). Control: --claim-only asks Gemini for the label from the claim
text alone, no references (pseudo-feature check).

Reference points (PANORAMA arXiv 2510.24774, NOC4PC, Table 14/15): best LLM
Claude 3.7 Sonnet CoT 45.40 custom score / 48.27 accuracy; random 32.33.
Their task gives the model the application-level union of references, ours
gives the claim's own references, so the numbers are indicative, not
comparable.

Usage:
    python3 evals/panorama_adjudication_eval.py --limit 100 --max-live-calls 260
    python3 evals/panorama_adjudication_eval.py --variants          # re-score only
    python3 evals/panorama_adjudication_eval.py --claim-only --limit 100
    python3 evals/panorama_adjudication_eval.py --run h2_holdout --variants            # hold-out set
    EVAL_BRI=1 python3 evals/panorama_adjudication_eval.py --run h2_holdout --tag bri --variants
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "h2"
LABELS = ("102", "103", "ALLOW")
MAX_DOC_CHARS = 600_000


class BudgetExceeded(RuntimeError):
    pass


def install_budget(max_live: int):
    import app.llm as llm
    import llm_cache
    llm_cache.install()
    cached_text = llm.call_llm

    async def text(*a, **k):
        if llm_cache.stats["misses"] >= max_live:
            raise BudgetExceeded(f"live LLM calls reached {max_live}")
        return await cached_text(*a, **k)

    llm.call_llm = text


def claim_elements(chain_text: str, with_preamble: bool = True) -> list[dict]:
    """Checklist in claim-mode format: [{id, criterion, weight}]."""
    from nodes.claim_mode import _parse_claim_limitations, _split_preamble
    parsed = _parse_claim_limitations(chain_text)
    pre = [re.sub(r"^\s*\d+\s*\.\s*", "", t) for t in _split_preamble(parsed["preamble"])] if with_preamble else []
    texts, seen = [], set()
    for t in pre + parsed["limitations"]:
        t = t.strip(" ,;.")
        if len(t) > 15 and t.lower() not in seen:
            seen.add(t.lower())
            texts.append(t)
    n = max(len(texts), 1)
    return [{"id": f"lim{i + 1}", "criterion": t, "weight": 1.0 / n, "preamble": i < len(pre)}
            for i, t in enumerate(texts)]


async def score_document(claim_text: str, checklist: list[dict], doc: dict) -> dict:
    from app.llm import evaluate_single_document_text
    from patent_analyzer.quote_verify import verify_checklist_results
    text = (doc.get("text") or "")[:MAX_DOC_CHARS]
    if len(text) < 120:
        return {"pub_num": doc["pub"], "title": doc.get("title", ""), "checklist_results": {},
                "source": "no_content", "text_mode": doc.get("text_mode")}
    res = await evaluate_single_document_text(claim_text, checklist, text, doc.get("title", ""),
                                              "Patent", doc_mode="full_text")
    res["quote_verification"] = verify_checklist_results(res.get("checklist_results", {}), text)
    res["pub_num"] = doc["pub"]
    res["text_mode"] = doc.get("text_mode")
    return res


async def run_stage1(samples: list[dict], docs: dict[str, dict], with_preamble: bool,
                     concurrency: int = 3, run_dir: Path = RUN_DIR, tag: str = "") -> dict[str, dict]:
    """Per instance: elements + one scored result per cited reference.
    Results go to <run_dir>/doc_results<tag>.json (tag e.g. "_bri" keeps a
    prompt variant's stage 1 apart from the default prompt's)."""
    out_path = run_dir / f"doc_results{tag}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    sem = asyncio.Semaphore(concurrency)

    async def one(s):
        key = f"{s['app']}:{s['claimNumber']}"
        cited = [docs[c["pub"]] for c in s["cited"] if c["pub"] in docs]
        shas = [hashlib.sha1((d.get("text") or "").encode()).hexdigest() for d in cited]
        prev = done.get(key)
        if prev and prev.get("with_preamble") == with_preamble and prev.get("text_shas") == shas:
            return
        checklist = claim_elements(s["chain_text"], with_preamble)
        async with sem:
            results = await asyncio.gather(*(score_document(s["chain_text"], checklist, d) for d in cited))
        done[key] = {"label": s["label"], "is_dependent": s["is_dependent"], "checklist": checklist,
                     "docs": results, "with_preamble": with_preamble, "text_shas": shas}
        _flush()

    def _flush():
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(done, ensure_ascii=False))
        tmp.replace(out_path)

    try:
        await asyncio.gather(*(one(s) for s in samples))
    finally:
        _flush()
    return done


def confusion(pairs: list[tuple[str, str]]) -> dict:
    """pairs of (gold, pred) -> accuracy, macro-F1, per-class P/R/F1, matrix."""
    m = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in pairs:
        m[g][p] += 1
    per = {}
    for c in LABELS:
        tp = m[c][c]
        fp = sum(m[g][c] for g in LABELS if g != c)
        fn = sum(m[c][p] for p in LABELS if p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": round(pr, 4), "recall": round(rc, 4),
                  "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0, "support": tp + fn}
    n = len(pairs)
    return {"n": n, "accuracy": round(sum(1 for g, p in pairs if g == p) / n, 4) if n else 0.0,
            "macro_f1": round(sum(v["f1"] for v in per.values()) / len(LABELS), 4),
            "per_class": per, "matrix": m}


async def run_findings(stage1: dict, docs: dict[str, dict], concurrency: int = 2,
                       run_dir: Path = RUN_DIR, tag: str = "") -> dict[str, dict]:
    """Stage 1.5: the four MPEP findings per instance, cached like stage 1.

    One call per instance, on the references the combination would rely on.
    The model never sees the label or the claim's own description — only the
    elements and the references' text (see app.llm.obviousness_findings and
    MPEP 2142 on hindsight). Every quote it returns is located before it counts.
    """
    from app.llm import obviousness_findings
    from patent_analyzer.obviousness import verify
    out_path = run_dir / f"obv_findings{tag}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    sem = asyncio.Semaphore(concurrency)

    def _texts(r: dict) -> dict[str, str]:
        return {d["pub_num"]: (docs.get(d["pub_num"], {}).get("text") or "")[:MAX_DOC_CHARS]
                for d in r["docs"] if d.get("pub_num")}

    async def one(key, r):
        if key in done:
            return
        texts = {k: t for k, t in _texts(r).items() if len(t) >= 120}
        if len(texts) < 1:
            done[key] = {"skipped": "no reference text"}
            return
        els = [c.get("criterion") or c.get("text") or "" for c in r["checklist"]]
        async with sem:
            raw = await obviousness_findings(els, texts)
        done[key] = {"raw": raw, "verified": verify(raw, texts, sorted(texts))}
        _flush()

    def _flush():
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(done, ensure_ascii=False))
        tmp.replace(out_path)

    try:
        await asyncio.gather(*(one(k, r) for k, r in stage1.items()))
    finally:
        _flush()
    return done


def score_rules(stage1: dict, params: dict, drop_preamble: bool = False,
                findings: dict | None = None) -> tuple[dict, list[dict]]:
    from patent_analyzer.adjudicate import adjudicate
    pairs, rows = [], []
    for key, r in stage1.items():
        checklist = [c for c in r["checklist"] if not (drop_preamble and c.get("preamble"))]
        f = ((findings or {}).get(key) or {}).get("verified") if findings else None
        adj = adjudicate(checklist, r["docs"], findings=f, **params)
        pairs.append((r["label"], adj["label"]))
        rows.append({"key": key, "gold": r["label"], "pred": adj["label"], "risk": adj["risk"],
                     "is_dependent": r["is_dependent"], "n_elements": adj["n_elements"],
                     "best_coverage": adj["best_coverage"],
                     "combo_coverage": adj["combo"]["coverage"] if adj["combo"] else 0.0,
                     "basis": adj["basis"],
                     "findings": {k: (f or {}).get(k, {}).get("status") for k in
                                  ("motivation", "expectation_of_success")} if f else None,
                     "text_modes": [d.get("text_mode") for d in r["docs"]]})
    return confusion(pairs), rows


VARIANTS = {
    "base (min_cover 1.0, allow_missing 0, score>0, preamble in)": {},
    "score>=2 only (Present, not Partial)": {"min_score": 2},
    "allow_missing 1": {"allow_missing": 1},
    "min_cover 0.8": {"min_cover": 0.8},
    "single_partial_103 0.7 (PANORAMA C.5.3 a)": {"single_partial_103": 0.7},
    "require_quotes off (score alone counts)": {"require_quotes": False},
    "require_quotes off + single_partial_103 0.7": {"require_quotes": False, "single_partial_103": 0.7},
    "no preamble elements": {"_drop_preamble": True},
    "no preamble + allow_missing 1": {"_drop_preamble": True, "allow_missing": 1},
    "no preamble + single_partial_103 0.7": {"_drop_preamble": True, "single_partial_103": 0.7},
}


def fmt_table(name: str, c: dict) -> str:
    per = c["per_class"]
    return (f"| {name} | {c['accuracy']:.3f} | {c['macro_f1']:.3f} | "
            + " | ".join(f"{per[l]['f1']:.2f}" for l in LABELS) + " |")


def fmt_matrix(c: dict) -> str:
    lines = ["| gold \\ pred | " + " | ".join(LABELS) + " |", "|---|---|---|---|"]
    for g in LABELS:
        lines.append(f"| {g} | " + " | ".join(str(c["matrix"][g][p]) for p in LABELS) + " |")
    return "\n".join(lines)


async def claim_only(samples: list[dict]) -> dict:
    """Control: label from the claim text alone, one Gemini call per claim."""
    from app.llm import call_llm
    system = ("You are a US patent examiner. Output JSON only.")
    pairs, rows = [], []
    for s in samples:
        prompt = (
            "Without any prior art in front of you, predict how the first office action treated this claim: "
            "\"102\" (anticipated by a single reference), \"103\" (obvious over a combination), or \"ALLOW\" "
            "(no prior-art rejection).\n\nCLAIM:\n" + s["chain_text"] +
            "\n\nJSON: {\"label\": \"102\"|\"103\"|\"ALLOW\", \"why\": \"one sentence\"}")
        pred = "ALLOW"
        try:
            resp = await call_llm(system, prompt, thinking_budget=0)
            m = re.search(r'"label"\s*:\s*"(102|103|ALLOW)"', resp)
            if m:
                pred = m.group(1)
        except BudgetExceeded:
            break
        except Exception:
            pass
        pairs.append((s["label"], pred))
        rows.append({"key": f"{s['app']}:{s['claimNumber']}", "gold": s["label"], "pred": pred,
                     "is_dependent": s["is_dependent"]})
    return {"confusion": confusion(pairs), "rows": rows}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-live-calls", type=int, default=260)
    ap.add_argument("--no-preamble", action="store_true", help="elements without preamble features (stage 1)")
    ap.add_argument("--variants", action="store_true", help="re-score cached stage-1 results under every rule variant")
    ap.add_argument("--claim-only", action="store_true", help="run the claim-only Gemini control")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--run", default="h2", help="run name under eval_data/runs/ (samples.json + docs.json)")
    ap.add_argument("--tag", default="", help="suffix for doc_results / adjudication_result (e.g. bri)")
    ap.add_argument("--findings-tag", default="",
                    help="separate cache suffix for stage 1.5 only, so a changed findings prompt does not "
                         "force stage 1 (the expensive one) to run again")
    ap.add_argument("--findings", action="store_true",
                    help="stage 1.5: one call per instance for the MPEP 2143.01 / 2143.02 / 2141.01(a) findings, "
                         "then score every variant a second time with the rule gated on them")
    args = ap.parse_args()

    import llm_cache
    install_budget(args.max_live_calls)
    run_dir = RUN_DIR.parent / args.run
    tag = f"_{args.tag}" if args.tag else ""
    samples = json.loads((run_dir / "samples.json").read_text())[:args.limit]
    docs = json.loads((run_dir / "docs.json").read_text())

    if args.claim_only:
        res = await claim_only(samples)
        (run_dir / "claim_only.json").write_text(json.dumps(res, indent=1))
        c = res["confusion"]
        print("claim-only control:", fmt_table("claim-only", c))
        print(fmt_matrix(c))
        dep = Counter((r["is_dependent"], r["pred"]) for r in res["rows"])
        print("pred by dependent:", dict(dep))
        print(llm_cache.summary())
        return

    try:
        stage1 = await run_stage1(samples, docs, with_preamble=not args.no_preamble, concurrency=args.concurrency,
                                  run_dir=run_dir, tag=tag)
    except BudgetExceeded as exc:
        print(f"stopped: {exc}")
        stage1 = json.loads((run_dir / f"doc_results{tag}.json").read_text())
    stage1 = {k: v for k, v in stage1.items() if k in {f"{s['app']}:{s['claimNumber']}" for s in samples}}
    print(f"stage 1: {len(stage1)} instances, {sum(len(v['docs']) for v in stage1.values())} scored documents; "
          + llm_cache.summary())

    findings = None
    if args.findings:
        findings = await run_findings(stage1, docs, concurrency=args.concurrency, run_dir=run_dir,
                                      tag=(f"_{args.findings_tag}" if args.findings_tag else tag))
        got = [f for f in findings.values() if f.get("verified")]
        loc = sum(f["verified"]["quotes_located"] for f in got)
        chk = sum(f["verified"]["quotes_checked"] for f in got)
        print(f"stage 1.5: findings for {len(got)}/{len(findings)} instances; "
              f"{loc}/{chk} quotes located ({loc / max(1, chk):.1%}) — an unlocated quote fails its finding")

    header = "| rule | acc | macro-F1 | F1 102 | F1 103 | F1 ALLOW |\n|---|---|---|---|---|---|"
    report = {"n": len(stage1), "variants": {}}
    print(header)
    for name, p in (VARIANTS if args.variants else {list(VARIANTS)[0]: {}}).items():
        p = dict(p)
        drop = p.pop("_drop_preamble", False)
        for suffix, f in (("", None),) + ((("  + MPEP findings gate", findings),) if findings else ()):
            c, rows = score_rules(stage1, p, drop_preamble=drop, findings=f)
            report["variants"][name + suffix] = {"params": p, "drop_preamble": drop,
                                                 "findings_gate": bool(f), "confusion": c, "rows": rows}
            print(fmt_table(name + suffix, c))
    base = list(report["variants"].values())[0]
    print(fmt_matrix(base["confusion"]))
    dep = Counter((r["is_dependent"], r["gold"] == r["pred"]) for r in base["rows"])
    print("base correct by dependent:", dict(dep))
    modes = Counter(m for r in base["rows"] for m in r["text_modes"])
    print("document text modes:", dict(modes))
    (run_dir / f"adjudication_result{tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
