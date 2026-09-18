#!/usr/bin/env python3
"""Draft gate 1: Dis2Pat element overlap — do the drafted claims recite the
elements of the claims actually granted for the same disclosure?

Data: HF `lj408/Dis2Pat` test split (943 rows; Jiang, Sun, Goetz, arXiv
2608.21249; CC-BY-SA-4.0), `dis2pat_test.jsonl` cached under
eval_data/dis2pat/. Sample: seed-42 permutation, first --limit rows. Input:
ONLY the seven `disclosure` fields (title, problem, core_idea, how_it_works,
novelty, benefits, optional_variants) rendered through
adapters.disclosure.doc_from_fields + adapters.paper.render_doc — no field of
the patent (title/abstract/claims/specification) is fed to the pipeline.
Pipeline: idca -> extraction subgraph -> draft node with no prior art
(DRAFT_RECHECK=0, DRAFT_ADVISORY=0): gate 1 measures drafting, not search
(the patents are on Google Patents and would find themselves).

Gold: the `claims` string split into numbered claims; independent claims cut
with nodes.claim_mode._parse_claim_limitations + _split_preamble (the same
cutter the extraction evals use); dependent claims reduced to their added
limitation (the "The … of claim N, wherein / further comprising" prefix
removed).

Metrics (te005 embed + extraction_eval.greedy_match, tau=.7, 1:1):
  indep_recall     draft claim 1 (preamble + limitations) vs gold claim 1 elements
  indep_precision  matched / drafted elements
  full             gold claim 1 elements all matched (0/1 per row)
  dep_recall / dep_precision   drafted dependents (primary form) vs gold dependents' added limitations
  controls: A2 independent_claim_draft (same cutter), elements joined directly
  (upper-bound candidate), gold vs itself = 1.0.
Also reported: granularity (drafted / gold elements), pool items dropped as
unsupported, LLM calls.

Usage:
    python3 evals/draft_eval.py --limit 5 --max-live-calls 60
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from common import load_env_yaml, model_tag
from extraction_eval import embed, greedy_match

DATA_DIR = Path(__file__).parent.parent / "eval_data" / "dis2pat"
DATA_FILE = DATA_DIR / "dis2pat_test.jsonl"
RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "draft_eval"
TAU = 0.7
_DEP_PREFIX = re.compile(r"^\s*\d+\s*\.\s*(?:The|A|An)\b[^,]{0,120}?\b(?:of|according to|as claimed in|as recited in|as in)\s+"
                         r"(?:any\s+(?:one\s+)?of\s+)?claims?\s+\d+\s*,?\s*(?:wherein|further comprising|further including|in which|characterized in that|and further comprising|comprising)?\s*", re.I)


class BudgetExceeded(RuntimeError):
    pass


def ensure_data() -> Path:
    if not DATA_FILE.exists():
        from huggingface_hub import hf_hub_download
        import shutil
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download("lj408/Dis2Pat", "dis2pat_test.jsonl", repo_type="dataset")
        shutil.copy(p, DATA_FILE)
    return DATA_FILE


def load_rows() -> list[dict]:
    return [json.loads(l) for l in ensure_data().read_text().splitlines() if l.strip()]


def sample_rows(rows: list[dict], limit: int, seed: int = 42) -> list[dict]:
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [rows[i] for i in idx[:limit]]


def install_budget(max_live: int):
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


def disclosure_text(row: dict) -> str:
    from patent_analyzer.adapters.disclosure import doc_from_fields
    from patent_analyzer.adapters.paper import render_doc
    d = row.get("disclosure") or {}
    doc = doc_from_fields(problem=d.get("problem", ""), core_idea=d.get("core_idea", ""), how_it_works=d.get("how_it_works", ""),
                          novelty=d.get("novelty", ""), optional_variants=d.get("optional_variants"), title=d.get("title", ""),
                          benefits=d.get("benefits", ""))
    return render_doc(doc)


def split_claims(claims: str) -> list[str]:
    text = re.sub(r"^.{0,80}?(?=\b1\s*\.\s+[A-Z])", "", (claims or "").strip(), count=1, flags=re.S)   # "What is claimed is:" / "The invention claimed is:"
    parts = [p.strip() for p in re.split(r"(?:(?<=\.)|^)\s*(?=\d{1,3}\s*\.\s+(?:A|An|The|In|Method|System|Apparatus|One|Non)\b)", text) if p.strip()]
    return [p for p in parts if re.match(r"^\d{1,3}\s*\.\s", p)]


def gold_claims(row: dict) -> dict:
    """{"independent": [{claim_no, elements[]}], "dependents": [{claim_no, parent, text}]}"""
    from graph.extraction_subgraph import _DEPENDENT
    from nodes.claim_mode import _parse_claim_limitations, _split_preamble
    indep, deps = [], []
    for claim in split_claims(row.get("claims") or ""):
        no = int(re.match(r"^(\d+)", claim).group(1))
        if _DEPENDENT.search(claim):
            m = re.search(r"claims?\s+(\d+)", claim, re.I)
            body = _DEP_PREFIX.sub("", claim).strip().rstrip(".")
            if len(body) > 10:
                deps.append({"claim_no": no, "parent": int(m.group(1)) if m else None, "text": body})
            continue
        parsed = _parse_claim_limitations(claim)
        texts = [re.sub(r"^\s*\d+\s*\.\s*", "", t) for t in _split_preamble(parsed["preamble"]) + parsed["limitations"]]
        texts = [t for t in texts if len(t.strip()) > 10]
        if texts:
            indep.append({"claim_no": no, "elements": texts})
    return {"independent": indep, "dependents": deps}


def draft_elements(draft: dict) -> tuple[list[str], list[str]]:
    """(claim 1 preamble + limitations, dependents of claim 1) from the draft node output."""
    claims = draft.get("claims") or []
    if not claims:
        return [], []
    c1 = claims[0]
    pre = re.sub(r",?\s*comprising:?\s*$", "", c1.get("preamble", "")).strip()
    els = ([pre] if len(pre) > 10 else []) + [l.get("text", "") for l in c1.get("limitations") or []]
    deps = [l.get("text", "") for c in claims if c.get("depends_on") == c1.get("no") for l in c.get("limitations") or []]
    return [e for e in els if e], [d for d in deps if d]


def a2_elements(core: dict) -> list[str]:
    from nodes.claim_mode import _parse_claim_limitations, _split_preamble
    form = core.get("primary_form") or "method"
    text = (core.get("independent_claim_draft") or {}).get(form) or ""
    if not text:
        return []
    parsed = _parse_claim_limitations(text)
    return [t for t in _split_preamble(parsed["preamble"]) + parsed["limitations"] if len(t.strip()) > 10]


def overlap(preds: list[str], gold: list[str], tau: float = TAU) -> dict:
    if not preds or not gold:
        return {"recall": 0.0, "precision": 0.0, "full": 0.0, "n_gold": len(gold), "n_pred": len(preds), "matched": 0}
    sim = embed(gold) @ embed(preds).T
    hit = sum(1 for _, _, s in greedy_match(sim) if s >= tau)
    return {"recall": hit / len(gold), "precision": hit / len(preds), "full": float(hit == len(gold)),
            "n_gold": len(gold), "n_pred": len(preds), "matched": hit}


async def run_row(row: dict, max_live: int) -> dict:
    from graph.extraction_subgraph import build_extraction_subgraph
    from nodes.draft import draft_node
    from nodes.idca import idca_node
    import llm_cache

    key = row.get("publication_number") or f"row{abs(hash(row.get('title', '')))}"
    out_path = RUN_DIR / f"{key}{model_tag()}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    text = disclosure_text(row)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp = f.name
    before = llm_cache.stats["hits"] + llm_cache.stats["misses"]
    p1 = await idca_node({"input_local_path": tmp, "input_mode": "disclosure"})
    rec = {"publication_number": key, "title": row.get("title", ""), "status_determination": p1.get("status_determination"),
           "extraction": None, "draft": None, "llm_calls": None}
    if p1.get("status_determination") != "Absent":
        p2 = await build_extraction_subgraph().ainvoke({"summary": p1.get("summary", ""), "document_text": text,
                                                        "input_mode": "disclosure", "cpc_subclass": p1.get("cpc_subclass", "")})
        rec["extraction"], rec["checklist"], rec["errors"] = p2.get("extraction"), p2.get("checklist", []), p2.get("errors")
        p3 = await draft_node({"job_id": key, "extraction": p2.get("extraction") or {}, "checklist": p2.get("checklist") or [],
                               "scoring_report": [], "adjudication": {}, "ranked_candidates": [], "summary": p1.get("summary", ""),
                               "document_text": text})
        rec["draft"] = p3.get("draft_claims")
        rec["draft_events"] = [e.get("message", "")[:300] for e in p3.get("events") or []]
    rec["llm_calls"] = llm_cache.stats["hits"] + llm_cache.stats["misses"] - before
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def _f(x) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-live-calls", type=int, default=60)
    ap.add_argument("--out", default="")
    from common import add_budget_arg, arm_budget
    add_budget_arg(ap)
    args = ap.parse_args()
    arm_budget(args)
    load_env_yaml()
    os.environ.setdefault("DRAFT_RECHECK", "0")
    os.environ.setdefault("DRAFT_ADVISORY", "0")
    os.environ.setdefault("IDCA_DOC_JSON", "0")
    install_budget(args.max_live_calls)
    import llm_cache

    rows = sample_rows(load_rows(), args.limit)
    results = []
    for row in rows:
        try:
            rec = await run_row(row, args.max_live_calls)
        except BudgetExceeded as e:
            print(f"stopped: {e}")
            break
        gold = gold_claims(row)
        g1 = next((c for c in gold["independent"] if c["claim_no"] == 1), gold["independent"][0] if gold["independent"] else None)
        draft = rec.get("draft") or {}
        core = next((c for c in ((rec.get("extraction") or {}).get("candidate_inventions") or []) if c.get("level") == "core"), None) or {}
        d_els, d_deps = draft_elements(draft)
        el_direct = [e["text"] for e in (core.get("elements") or []) if not e.get("unsupported")]
        gold_deps = [d["text"] for d in gold["dependents"] if d.get("parent") == (g1 or {}).get("claim_no")]
        row_out = {"publication_number": rec["publication_number"], "title": rec.get("title", "")[:80], "strategy": draft.get("strategy"),
                   "n_claims": len(draft.get("claims") or []), "llm_calls": rec.get("llm_calls"),
                   "dropped": sum(1 for p in (draft.get("avoidance") or {}).get("pool") or [] if p.get("dropped")),
                   "gold_indep_elements": len((g1 or {}).get("elements") or []), "gold_deps": len(gold_deps),
                   "indep": overlap(d_els, (g1 or {}).get("elements") or []),
                   "indep_a2": overlap(a2_elements(core), (g1 or {}).get("elements") or []),
                   "indep_elements": overlap(el_direct, (g1 or {}).get("elements") or []),
                   "gold_self": overlap((g1 or {}).get("elements") or [], (g1 or {}).get("elements") or []),
                   "dep": overlap(d_deps, gold_deps),
                   "wording_accepted": ((draft.get("wording_check") or {}).get("primary") or {}).get("accepted"),
                   "open_flags": len((draft.get("definiteness") or {}).get("open_flags") or [])}
        results.append(row_out)
        print(f"{row_out['publication_number']:<14} {row_out['title'][:48]:<50} strat={row_out['strategy']} claims={row_out['n_claims']} "
              f"indep R/P={_f(row_out['indep']['recall'])}/{_f(row_out['indep']['precision'])} ({row_out['indep']['n_pred']}/{row_out['gold_indep_elements']}) "
              f"a2 R={_f(row_out['indep_a2']['recall'])} els R={_f(row_out['indep_elements']['recall'])} "
              f"dep R/P={_f(row_out['dep']['recall'])}/{_f(row_out['dep']['precision'])} ({row_out['dep']['n_pred']}/{row_out['gold_deps']}) "
              f"dropped={row_out['dropped']} open112b={row_out['open_flags']} llm={row_out['llm_calls']}")

    if not results:
        return
    def mean(key, sub):
        vals = [r[key][sub] for r in results if r[key]["n_gold"]]
        return sum(vals) / len(vals) if vals else None
    def pooled(key):
        tp = sum(r[key]["matched"] for r in results)
        ng = sum(r[key]["n_gold"] for r in results)
        np_ = sum(r[key]["n_pred"] for r in results)
        return (tp / ng if ng else None, tp / np_ if np_ else None)
    print(f"\nn={len(results)}  tau={TAU}  (macro = mean over rows with gold; pooled = sum tp / sum gold)")
    print(f"{'column':<22}{'recall':>8}{'prec':>8}{'full':>8}{'pooledR':>9}{'pooledP':>9}")
    for label, key in (("draft claim 1", "indep"), ("A2 draft", "indep_a2"), ("elements direct", "indep_elements"),
                       ("gold self", "gold_self"), ("draft dependents", "dep")):
        pr, pp = pooled(key)
        print(f"{label:<22}{_f(mean(key, 'recall')):>8}{_f(mean(key, 'precision')):>8}{_f(mean(key, 'full')):>8}{_f(pr):>9}{_f(pp):>9}")
    gran = [r["indep"]["n_pred"] / r["gold_indep_elements"] for r in results if r["gold_indep_elements"]]
    print(f"granularity (drafted / gold elements): {sum(gran) / len(gran):.2f}   dropped unsupported pool items: {sum(r['dropped'] for r in results)}   "
          f"open 112(b) flags: {sum(r['open_flags'] for r in results)}   LLM calls: {sum(r['llm_calls'] or 0 for r in results)} "
          f"(live {llm_cache.stats['misses']}, cached {llm_cache.stats['hits']})")
    if args.out:
        Path(args.out).write_text(json.dumps({"tau": TAU, "n": len(results), "rows": results}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
