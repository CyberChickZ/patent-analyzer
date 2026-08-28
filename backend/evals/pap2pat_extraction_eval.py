#!/usr/bin/env python3
"""Gate 2: Pap2Pat Coverage@n — do the extracted candidate inventions cover the
elements of the real independent claims filed for the same paper?

Sample: Pap2Pat test pairs, seed 42, same 50 as E4 (openworld_eval.sample_pairs);
--limit runs the first k of them (smoke default 5). Input: paper.json rendered
through adapters.paper (with [S1.P3] markers) — no field of patent.json is fed
to the pipeline. Gold: independent claims of patent.json (numbered, not
referring back to another claim) cut with nodes.claim_mode._parse_claim_limitations.

Coverage@n: union of the supported elements of the first n candidates vs each
independent claim's elements, greedy 1:1 at tau=0.7 (te005), recall pooled over
claims; full coverage = share of independent claims whose elements are all
matched. Control column: the current SSR checklist (n = all). Upper bound: the
regex-cut gold matched against itself = 1.0. Error columns from
extraction_errors.classify_errors (level misclassification: gold claim 1 = core).

Usage:
    python3 evals/pap2pat_extraction_eval.py --limit 5 --max-live-calls 40
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common import model_tag
from extraction_errors import classify_errors
from extraction_eval import embed, greedy_match

PAP2PAT_ROOT = Path(os.environ.get("PAP2PAT_ROOT", "/tmp/pap2pat"))
RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "p2p_extract"
TAU = 0.7
NS = (1, 2, 3)
_DEPENDENT = re.compile(r"\b(?:of|according to|as claimed in|as recited in|as defined in|as in)\s+"
                        r"(?:any\s+(?:one\s+)?of\s+)?claims?\s+\d", re.I)


class BudgetExceeded(RuntimeError):
    pass


def ensure_data() -> Path:
    data = PAP2PAT_ROOT / "Pap2Pat" / "data"
    if not data.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/boschresearch/Pap2Pat",
                        str(PAP2PAT_ROOT)], check=True)
    import openworld_eval
    openworld_eval.PAP2PAT = data
    return data


def install_budget(max_live: int):
    """Hard cap: any call that would go live past `max_live` raises instead."""
    import app.llm as llm
    import llm_cache
    llm_cache.install()
    cached_text, cached_pdfs = llm.call_llm, llm.call_llm_with_pdfs

    def guard():
        if llm_cache.stats["misses"] >= max_live:
            raise BudgetExceeded(f"live LLM calls reached {max_live}")

    async def text(*a, **k):
        guard()
        return await cached_text(*a, **k)

    async def pdfs(*a, **k):
        guard()
        return await cached_pdfs(*a, **k)

    llm.call_llm, llm.call_llm_with_pdfs = text, pdfs


def independent_claims(claims: list[str]) -> list[tuple[int, str]]:
    out = []
    for c in claims or []:
        m = re.match(r"^\s*(\d+)\s*\.\s", c or "")
        if m and not _DEPENDENT.search(c):
            out.append((int(m.group(1)), c.strip()))
    return out


def gold_claims(patent: dict) -> list[dict]:
    """[{claim_no, elements: [{text, level}]}] — level 'core' for claim 1 only."""
    from nodes.claim_mode import _parse_claim_limitations, _split_preamble
    out = []
    for no, claim in independent_claims(patent.get("claims") or []):
        parsed = _parse_claim_limitations(claim)
        texts = [re.sub(r"^\s*\d+\s*\.\s*", "", t) for t in _split_preamble(parsed["preamble"]) + parsed["limitations"]]
        texts = [t for t in texts if len(t.strip()) > 15]
        if texts:
            out.append({"claim_no": no, "elements": [{"text": t, "level": "core" if no == 1 else None} for t in texts]})
    return out


def paper_texts(pair_id: str, data: Path, input_mode: str = "academic_paper") -> tuple[str, str]:
    """(IDCA text = openworld_eval.render_paper, marker text via adapters.paper).

    input_mode='manuscript' is the submission-draft scenario: the marker text
    is cut by the production adapter (patent_analyzer.adapters.manuscript
    .strip_related_work — drops Related Work / Background / Prior Art at any
    depth and clears the abstract), exactly what nodes/idca.py:72 does to the
    Doc JSON in manuscript mode. The IDCA text stays whole, because IDCA is
    what does that cut in production.
    """
    from openworld_eval import render_paper
    from patent_analyzer.adapters.paper import doc_from_sections, render_doc
    paper = json.loads((data / pair_id / "paper.json").read_text())
    doc = doc_from_sections(paper.get("title", ""), paper.get("abstract", ""), paper.get("sections"))
    if input_mode == "manuscript":
        from patent_analyzer.adapters.manuscript import strip_related_work
        doc = strip_related_work(doc)
    return render_paper(pair_id), render_doc(doc)


async def run_pair(pair: dict, extractor: str, data: Path, input_mode: str = "academic_paper",
                   run_tag: str = "") -> dict:
    from nodes.idca import idca_node
    import llm_cache

    key = pair["pair_id"]
    mode_tag = "" if input_mode == "academic_paper" else f"_{input_mode}"
    out_path = RUN_DIR / f"{key}_{extractor}{mode_tag}{model_tag()}{('_' + run_tag) if run_tag else ''}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    idca_text, marker_text = paper_texts(key, data, input_mode)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(idca_text)
        tmp = f.name
    before = llm_cache.stats["hits"] + llm_cache.stats["misses"]
    p1 = await idca_node({"input_local_path": tmp, "input_mode": input_mode})
    rec = {"pair_id": key, "extractor": extractor, "status_determination": p1.get("status_determination"),
           "input_mode": p1.get("input_mode"), "summary": p1.get("summary", ""),
           "checklist": [], "extraction": None, "errors": None, "llm_calls": None}
    if p1.get("status_determination") == "Present":
        if extractor == "new":
            from graph.extraction_subgraph import build_extraction_subgraph
            p2 = await build_extraction_subgraph().ainvoke({
                "summary": p1["summary"], "document_text": marker_text,
                "input_mode": p1.get("input_mode", "academic_paper"), "cpc_subclass": p1.get("cpc_subclass", "")})
            rec.update(checklist=p2.get("checklist", []), extraction=p2.get("extraction"),
                       errors=p2.get("errors"), llm_calls=p2.get("llm_calls"), retry_count=p2.get("retry_count", 0))
        else:
            from graph.ssr_subgraph import build_ssr_subgraph
            p2 = await build_ssr_subgraph().ainvoke({
                "summary": p1["summary"], "fields_map": p1.get("fields_map", []),
                "cpc_subclass": p1.get("cpc_subclass", ""), "personas": p1.get("personas", {})})
            rec.update(checklist=p2.get("checklist", []), retry_count=p2.get("retry_count", 0))
    rec["llm_calls_total"] = llm_cache.stats["hits"] + llm_cache.stats["misses"] - before
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def candidate_elements(rec: dict, n: int | None, supported_only: bool = True) -> list[dict]:
    cands = ((rec.get("extraction") or {}).get("candidate_inventions") or [])
    cands = cands if n is None else cands[:n]
    return [{"id": e["id"], "text": e["text"], "evidence_quote": e.get("evidence_quote", ""),
             "kind": e.get("kind"), "level": c.get("level")}
            for c in cands for e in c.get("elements") or []
            if not (supported_only and e.get("unsupported"))]


def checklist_elements(rec: dict) -> list[dict]:
    return [{"id": f"c{i}", "text": c["criterion"]} for i, c in enumerate(rec.get("checklist") or []) if c.get("criterion")]


def coverage(preds: list[str], claims: list[dict], tau: float = TAU) -> dict:
    """Per independent claim, greedy 1:1 at tau; recall pooled over claims;
    full = share of claims with every element matched."""
    n_gold = sum(len(c["elements"]) for c in claims)
    if not preds or not n_gold:
        return {"recall": 0.0, "full": 0.0, "n_gold": n_gold, "n_pred": len(preds)}
    pv = embed(preds)
    tp, full = 0, 0
    for c in claims:
        sim = embed([e["text"] for e in c["elements"]]) @ pv.T
        hit = sum(1 for _, _, s in greedy_match(sim) if s >= tau)
        tp += hit
        full += hit == len(c["elements"])
    return {"recall": tp / n_gold, "full": full / len(claims), "n_gold": n_gold, "n_pred": len(preds)}


def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5, help="first k of the 50 seed-42 pairs")
    ap.add_argument("--pairs", default="", help="comma-separated pair_ids (or the US pub prefix) instead of --limit")
    ap.add_argument("--input-mode", default="academic_paper", choices=["academic_paper", "manuscript"],
                    help="manuscript = submission draft: Related Work / Background dropped, abstract cleared")
    ap.add_argument("--run-tag", default="", help="suffix on the run files; model_tag() only names a stage "
                    "override, so a run under a new *global* model would otherwise replay an old model's record")
    ap.add_argument("--max-live-calls", type=int, default=40)
    ap.add_argument("--no-control", action="store_true", help="skip the SSR control column")
    ap.add_argument("--no-errors", action="store_true")
    args = ap.parse_args()

    data = ensure_data()
    install_budget(args.max_live_calls)
    import llm_cache
    from openworld_eval import sample_pairs
    pairs = sample_pairs(50, seed=42)
    if args.pairs:
        want = [x.strip() for x in args.pairs.split(",") if x.strip()]
        pairs = [p for p in pairs if any(w in p["pair_id"] or w in p["patent_pub"] for w in want)]
    else:
        pairs = pairs[:args.limit]

    rows = []
    for pair in pairs:
        patent = json.loads((data / pair["pair_id"] / "patent.json").read_text())
        claims = gold_claims(patent)
        if not claims:
            print(f"[{pair['pair_id']}] no independent claims parsed — skipped")
            continue
        row = {"pair_id": pair["pair_id"], "claims": claims, "new": None, "ssr": None}
        for extractor in (["new"] if args.no_control else ["new", "ssr"]):
            try:
                row[extractor] = await run_pair(pair, extractor, data, args.input_mode, args.run_tag)
            except BudgetExceeded as exc:
                print(f"[{pair['pair_id']}/{extractor}] BUDGET: {exc}")
            except Exception as exc:
                print(f"[{pair['pair_id']}/{extractor}] FAILED {type(exc).__name__}: {exc}")
        rows.append(row)

    print(f"\n== Pap2Pat extraction  pairs={len(rows)}  seed=42  tau={TAU}  input_mode={args.input_mode}")
    hdr = f"{'pair':<28}{'#claims':>8}{'#gold':>6}" + "".join(f"{'cov@' + str(n):>8}" for n in NS) + \
          f"{'full@3':>8}{'#cand':>6}{'surv':>7}{'calls':>6}{'| ssr cov':>10}{'full':>6}{'#items':>7}"
    print(hdr)
    agg = {f"new@{n}": [] for n in NS}
    agg.update({"new_full": [], "ssr": [], "ssr_full": [], "surv": [], "calls": [], "ncand": [], "ssr_items": []})
    err_agg = {k: {"omission": [0, 0], "fabrication": [0, 0], "misclassification": [0, 0]} for k in ("new", "ssr")}
    for row in rows:
        claims, new, ssr = row["claims"], row["new"], row["ssr"]
        n_gold = sum(len(c["elements"]) for c in claims)
        line = f"{row['pair_id']:<28}{len(claims):>8}{n_gold:>6}"
        covs = {}
        if new and new.get("extraction"):
            for n in NS:
                covs[n] = coverage([e["text"] for e in candidate_elements(new, n)], claims)
                agg[f"new@{n}"].append(covs[n]["recall"])
            agg["new_full"].append(covs[3]["full"])
            agg["ncand"].append(len(new["extraction"].get("candidate_inventions") or []))
            surv = (new.get("errors") or {}).get("quote_survival")
            if surv is not None:
                agg["surv"].append(surv)
            if new.get("llm_calls") is not None:
                agg["calls"].append(new["llm_calls"])
            line += "".join(f"{covs[n]['recall']:>8.3f}" for n in NS)
            line += f"{covs[3]['full']:>8.3f}{agg['ncand'][-1]:>6}{_fmt(surv):>7}{new.get('llm_calls') or '-':>6}"
        else:
            line += f"{'-':>8}" * (len(NS) + 1) + f"{'-':>6}{'-':>7}{'-':>6}"
        if ssr and ssr.get("checklist"):
            cv = coverage([e["text"] for e in checklist_elements(ssr)], claims)
            agg["ssr"].append(cv["recall"])
            agg["ssr_full"].append(cv["full"])
            agg["ssr_items"].append(cv["n_pred"])
            line += f"{cv['recall']:>10.3f}{cv['full']:>6.2f}{cv['n_pred']:>7}"
        else:
            line += f"{'-':>10}{'-':>6}{'-':>7}"
        print(line)
        if not args.no_errors:
            gold = [e for c in claims for e in c["elements"]]
            _, marker_text = paper_texts(row["pair_id"], data, args.input_mode)
            for k, rec, preds, rq in (("new", new, candidate_elements(new, None, supported_only=False) if new else [], True),
                                      ("ssr", ssr, checklist_elements(ssr) if ssr else [], False)):
                if not preds:
                    continue
                err = classify_errors(preds, gold, marker_text, require_quote=rq)
                e = err_agg[k]
                e["omission"][0] += len(err["omission"]); e["omission"][1] += err["n_gold"]
                e["fabrication"][0] += len(err["fabrication"]); e["fabrication"][1] += err["n_pred"]
                e["misclassification"][0] += len(err["misclassification"]); e["misclassification"][1] += len(err["matched"])

    def mean(xs):
        return f"{np.mean(xs):.3f}" if xs else "  n/a"

    print(f"\n{'mean':<28}{'':>8}{'':>6}" + "".join(f"{mean(agg[f'new@{n}']):>8}" for n in NS)
          + f"{mean(agg['new_full']):>8}{mean(agg['ncand']):>6}{mean(agg['surv']):>7}{mean(agg['calls']):>6}"
          + f"{mean(agg['ssr']):>10}{mean(agg['ssr_full']):>6}{mean(agg['ssr_items']):>7}")
    print("upper bound (regex-cut gold vs itself) = 1.000")
    if not args.no_errors:
        for k in ("new", "ssr"):
            e = err_agg[k]
            def rate(name):
                n, d = e[name]
                return f"{n / d:.3f} ({n}/{d})" if d else "n/a"
            print(f"errors[{k}]: omission={rate('omission')}  fabrication={rate('fabrication')}  "
                  f"misclassification(level)={rate('misclassification')}")
    print("\n" + llm_cache.summary())


if __name__ == "__main__":
    asyncio.run(main())
