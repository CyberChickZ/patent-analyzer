#!/usr/bin/env python3
"""J5: per-stage model comparison on the existing gates.

    python3 evals/j5_model_gates.py --gate extract --model gemini-3.5-flash
    python3 evals/j5_model_gates.py --gate eval    --model gemini-2.5-pro
    python3 evals/j5_model_gates.py --gate screen  --model gemini-3.1-flash-lite --tag h1d

Sets LLM_MODEL_<STAGE> for the gate's stage only (every other stage stays on the
default LLM_MODEL), points llm_cache at a J5-only cache dir so every model —
including the default — goes live once, and runs the gate's eval in-process:

  extract  extraction_fulldoc_eval (FiNE 5, --extractor new) + pap2pat_extraction_eval (5 pairs)
  eval     coverage_eval --protocol fine --checklist both --limit 20 --sample stage --verifier either
  screen   prune.stage2_llm re-run offline on the stage-1 survivors recorded in
           eval_data/runs/e4/*_search_<tag>.json (abstracts re-fetched from BigQuery
           and cached), gold-family survival via evals.funnel.gold_family_map

Every Vertex round-trip is metered through a client proxy (model, stage by
system prompt, seconds, tokens, 429s) so the stage's cost/latency is separable
from the IDCA calls that run on the default model. Output:
eval_data/runs/j5/<gate>_<model>[_<tag>].json
"""

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "j5"
E4_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "e4"
STAGE_OF = {"extract": "extract", "eval": "eval", "screen": "screen"}

# system-prompt fingerprints → stage (only for attribution in the meter)
SYS_STAGE = [
    ("patent attorney identifying what in a technical document", "extract"),
    ("patent attorney drafting independent claims", "extract"),
    ("verify whether a generated text is faithful", "extract"),
    ("US patent examiner comparing a SOURCE invention", "eval"),
    ("You are a US patent examiner. Output JSON only.", "eval"),
    ("patent examiner screening search results", "screen"),
    ("patent analyst. Read the document and classify", "idca"),
    ("document transcription engine", "docjson"),
    ("patent search facets", "facets"),
]

# USD per 1M tokens, Standard PayGo, global endpoint, <=200K-token prompts
# (cloud.google.com/vertex-ai/generative-ai/pricing, fetched 2026-09-18)
PRICE = {
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
}

calls: list[dict] = []


def _stage_of(system: str) -> str:
    for needle, st in SYS_STAGE:
        if needle in (system or ""):
            return st
    return "other"


def install_meter():
    """Proxy the GenAI client so each generate_content is logged with its model,
    stage, latency and usage — exact even under concurrency."""
    import app.llm as llm
    real_get_client = llm.get_client

    class _Models:
        def __init__(self, inner):
            self._inner = inner

        async def generate_content(self, model, contents, config):
            system = getattr(config, "system_instruction", "") or ""
            t0 = time.monotonic()
            try:
                resp = await self._inner.generate_content(model=model, contents=contents, config=config)
            except Exception as exc:
                calls.append({"model": model, "stage": _stage_of(system), "seconds": round(time.monotonic() - t0, 2),
                              "error": f"{type(exc).__name__}:{getattr(exc, 'code', '')}"})
                raise
            um = getattr(resp, "usage_metadata", None)
            calls.append({"model": model, "stage": _stage_of(system), "seconds": round(time.monotonic() - t0, 2),
                          "prompt_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
                          "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
                          "thought_tokens": int(getattr(um, "thoughts_token_count", 0) or 0),
                          "finish": str(getattr(resp.candidates[0], "finish_reason", "")) if getattr(resp, "candidates", None) else ""})
            return resp

    class _Proxy:
        def __init__(self, inner):
            self._inner = inner
            self.aio = type("A", (), {})()
            self.aio.models = _Models(inner.aio.models)

        def __getattr__(self, k):
            return getattr(self._inner, k)

    proxy = None

    def get_client():
        nonlocal proxy
        if proxy is None:
            proxy = _Proxy(real_get_client())
        return proxy

    llm.get_client = get_client


def meter_summary(stage: str, model: str, n_docs: int) -> dict:
    mine = [c for c in calls if c["stage"] == stage and c["model"] == model and "error" not in c]
    errs = [c for c in calls if c["stage"] == stage and c["model"] == model and "error" in c]
    pin, pout = PRICE.get(model, (0.0, 0.0))
    tok_in = sum(c["prompt_tokens"] for c in mine)
    tok_out = sum(c["output_tokens"] + c["thought_tokens"] for c in mine)
    cost = tok_in / 1e6 * pin + tok_out / 1e6 * pout
    secs = sum(c["seconds"] for c in mine)
    return {"stage": stage, "model": model, "n_docs": n_docs, "calls": len(mine),
            "errors": len(errs), "errors_429": sum(1 for c in errs if c["error"].endswith("429")),
            "prompt_tokens": tok_in, "output_tokens": sum(c["output_tokens"] for c in mine),
            "thought_tokens": sum(c["thought_tokens"] for c in mine),
            "seconds_total": round(secs, 1), "seconds_per_doc": round(secs / n_docs, 1) if n_docs else None,
            "cost_usd": round(cost, 4), "cost_per_doc": round(cost / n_docs, 4) if n_docs else None,
            "max_tokens_hits": sum(1 for c in mine if "MAX_TOKENS" in c.get("finish", "")),
            "other_stage_calls": {st: sum(1 for c in calls if c["stage"] == st)
                                  for st in sorted({c["stage"] for c in calls} - {stage})}}


async def _run_main(module, argv: list[str]) -> str:
    """Run an eval's async main() with argv, echoing and capturing stdout."""
    buf = io.StringIO()

    class _Tee(io.TextIOBase):
        def write(self, s):
            buf.write(s)
            sys.__stdout__.write(s)
            return len(s)

        def flush(self):
            sys.__stdout__.flush()

    old_argv = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        with contextlib.redirect_stdout(_Tee()):
            await module.main()
    finally:
        sys.argv = old_argv
    return buf.getvalue()


def _lines_after(text: str, marker: str, n: int) -> list[str]:
    out = text.splitlines()
    for i, l in enumerate(out):
        if marker in l:
            return out[i:i + n]
    return []


async def gate_extract(model: str) -> dict:
    import extraction_fulldoc_eval as fe
    import pap2pat_extraction_eval as pe
    from common import model_tag
    from extraction_fulldoc_eval import RUN_DIR as S1, load_fixture
    from pap2pat_extraction_eval import RUN_DIR as P2P

    out_fine = await _run_main(fe, ["--limit", "5", "--extractor", "new", "--concurrency", "1", "--max-live-calls", "400"])
    out_p2p = await _run_main(pe, ["--limit", "5", "--no-control", "--max-live-calls", "400"])

    apps = load_fixture()["stage"][:5]
    tag = model_tag()
    fine_runs = [json.loads((S1 / f"{a}_desc_only_new{tag}.json").read_text())
                 for a in apps if (S1 / f"{a}_desc_only_new{tag}.json").exists()]
    from openworld_eval import sample_pairs
    pairs = [p["pair_id"] for p in sample_pairs(50, seed=42)[:5]]
    p2p_runs = [json.loads((P2P / f"{k}_new{tag}.json").read_text())
                for k in pairs if (P2P / f"{k}_new{tag}.json").exists()]

    def _unsup(runs):
        n_el = sum((r.get("errors") or {}).get("n_elements") or 0 for r in runs)
        n_un = sum((r.get("errors") or {}).get("n_unsupported") or 0 for r in runs)
        return {"n_elements": n_el, "n_unsupported": n_un, "unsupported_ratio": round(n_un / n_el, 4) if n_el else None}

    return {"fine_stdout": _lines_after(out_fine, "== mode=", 12), "p2p_stdout": _lines_after(out_p2p, "== Pap2Pat", 12),
            "fine_unsupported": _unsup(fine_runs), "p2p_unsupported": _unsup(p2p_runs),
            "fine_present": sum(1 for r in fine_runs if r.get("status_determination") == "Present"),
            "fine_retries": sum(1 for r in fine_runs if r.get("retry_count")),
            "n_docs": len(fine_runs) + len(p2p_runs)}


async def gate_eval(model: str) -> dict:
    import coverage_eval as ce
    summ = RUN_DIR / f"eval_summary_{model}.json"
    if summ.exists():
        summ.unlink()
    out = await _run_main(ce, ["--protocol", "fine", "--checklist", "both", "--limit", "20", "--sample", "stage",
                               "--verifier", "either", "--concurrency", "1", "--summary", str(summ)])
    summary = json.loads(summ.read_text()) if summ.exists() else {}
    return {"stdout": _lines_after(out, "variant", 6), "summary": summary, "n_docs": 20}


async def _abstracts(key: str, docs: list[dict]) -> dict[str, dict]:
    """title/abstract for the patent survivors, fetched once from BigQuery and cached."""
    from patent_analyzer.recall.bigquery_patents import fetch_by_pub_nums, _canon_pub
    cache = RUN_DIR / f"abstracts_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pubs = [d["pub_num"] for d in docs if d.get("match_type") == "Patent" and d.get("pub_num")]
    got = await fetch_by_pub_nums(pubs, with_claims=False)
    out = {p: {"title": got[_canon_pub(p)].get("title", ""), "abstract": got[_canon_pub(p)].get("abstract", ""),
               "cpc_codes": got[_canon_pub(p)].get("cpc_codes", []),
               "year": (got[_canon_pub(p)].get("publication_date") or "")[:4]}
           for p in pubs if _canon_pub(p) in got}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out


async def gate_screen(model: str, tag: str) -> dict:
    from funnel import gold_family_map, _canon, _fams
    from patent_analyzer.agentic import prune
    gold = json.loads((E4_DIR / "gold.json").read_text())
    rows = []
    for f in sorted(E4_DIR.glob(f"*_search_{tag}.json")):
        rec = json.loads(f.read_text())
        g = gold.get(rec["key"])
        fd = rec.get("funnel_docs") or []
        if not g or not fd or not rec.get("extraction"):
            continue
        per_paper = RUN_DIR / f"screen_{model}_{tag}_{rec['key']}.json"
        if per_paper.exists():
            rows.append(json.loads(per_paper.read_text()))
            continue
        s1 = sorted([d for d in fd if d.get("stage1")], key=lambda d: -float(d.get("cos") or 0))
        abstracts = await _abstracts(rec["key"], s1)
        docs = []
        for d in s1:
            a = abstracts.get(d["pub_num"], {})
            docs.append({"pub_num": d["pub_num"], "title": a.get("title") or d.get("title", ""),
                         "abstract": a.get("abstract", ""), "match_type": d.get("match_type"),
                         # production pool dicts carry no top-level cpc_codes (they sit in raw.bigquery), so none here
                         "cpc_codes": [], "year": a.get("year") or None,
                         "prune_cos": float(d.get("cos") or 0)})
        cands = [{"id": c.get("id"), "concept": c.get("concept", "")}
                 for c in rec["extraction"].get("candidate_inventions") or []]
        elements = [{"id": e["id"], "text": e["text"]} for e in rec.get("loop_elements") or []]
        fam_of = await gold_family_map(g)
        gold_in = _fams([d["pub_num"] for d in docs], fam_of)
        t0 = time.monotonic()
        kept, stats = await prune.stage2_llm(cands, elements, docs, list(range(len(docs))))
        wall = time.monotonic() - t0
        worth = [d["pub_num"] for d in docs if d.get("prune_worth_reading")]
        gold_worth = _fams(worth, fam_of)
        gold_kept = _fams([docs[i]["pub_num"] for i in kept], fam_of)
        # the production verdicts recorded in the run file (default model) for the same docs
        rec_worth = [d["pub_num"] for d in s1 if d.get("worth_reading")]
        rows.append({"key": rec["key"], "n_docs": len(docs), "with_abstract": sum(1 for d in docs if d["abstract"]),
                     "gold_families_in_stage1": len(gold_in), "gold_worth": len(gold_worth), "gold_kept60": len(gold_kept),
                     "worth": len(worth), "kept": len(kept), "unanswered": stats.get("unanswered"),
                     "calls": stats.get("stage2_calls"), "wall_seconds": round(wall, 1),
                     "recorded_run": {"gold_worth": len(_fams(rec_worth, fam_of)), "worth": len(rec_worth),
                                      "gold_kept60": len(_fams([p["pub_num"] for p in rec.get("pruned", [])], fam_of))}})
        rows[-1]["meter"] = meter_summary("screen", model, 1)
        calls.clear()
        per_paper.write_text(json.dumps(rows[-1], ensure_ascii=False, indent=1))
        print(f"[{rec['key']}] {model}: {len(docs)} docs ({rows[-1]['with_abstract']} w/ abstract) → worth {len(worth)} "
              f"kept {len(kept)} | gold fam in {len(gold_in)} → worth {len(gold_worth)} kept {len(gold_kept)} "
              f"(recorded run: worth {rows[-1]['recorded_run']['gold_worth']}) {wall:.0f}s")
    n = len(rows)
    agg = {"papers": n, "gold_in": sum(r["gold_families_in_stage1"] for r in rows),
           "gold_worth": sum(r["gold_worth"] for r in rows), "gold_kept60": sum(r["gold_kept60"] for r in rows),
           "recorded_gold_worth": sum(r["recorded_run"]["gold_worth"] for r in rows),
           "recorded_gold_kept60": sum(r["recorded_run"]["gold_kept60"] for r in rows),
           "worth_total": sum(r["worth"] for r in rows), "unanswered": sum(r["unanswered"] or 0 for r in rows),
           "wall_seconds_per_paper": round(sum(r["wall_seconds"] for r in rows) / n, 1) if n else None}
    return {"rows": rows, "agg": agg, "n_docs": n}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=list(STAGE_OF))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="h1d", help="screen gate: e4 run tag whose stage-1 survivors are re-screened")
    from common import add_budget_arg, arm_budget
    add_budget_arg(ap)
    args = ap.parse_args()
    arm_budget(args)

    stage = STAGE_OF[args.gate]
    os.environ[f"LLM_MODEL_{stage.upper()}"] = args.model
    os.environ.setdefault("LLM_CACHE_DIR", str(Path(__file__).parent.parent / "eval_data" / ".llm_cache_j5"))
    from common import load_env_yaml
    load_env_yaml()
    install_meter()
    import llm_cache
    llm_cache.install()
    import app.llm as llm
    print(f"[j5] gate={args.gate} stage={stage} model={args.model} default={llm.MODEL} cache={os.environ['LLM_CACHE_DIR']}")

    t0 = time.monotonic()
    res: dict = {}
    try:
        if args.gate == "extract":
            res = await gate_extract(args.model)
        elif args.gate == "eval":
            res = await gate_eval(args.model)
        else:
            res = await gate_screen(args.model, args.tag)
    except BaseException as exc:
        # keep the meter even when the eval's scoring crashes — the live calls were paid for
        res = {"failed": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(f"[j5] gate crashed: {res['failed']} — writing meter anyway")
    res["wall_seconds"] = round(time.monotonic() - t0, 1)
    res["meter"] = meter_summary(stage, args.model, res.get("n_docs") or 0)
    if args.gate == "screen" and res.get("rows"):
        ms = [r["meter"] for r in res["rows"] if r.get("meter")]
        n = len(ms) or 1
        tot = {k: sum(m[k] for m in ms) for k in ("calls", "errors", "errors_429", "prompt_tokens", "output_tokens",
                                                   "thought_tokens", "seconds_total", "cost_usd", "max_tokens_hits")}
        res["meter"] = {"stage": stage, "model": args.model, "n_docs": len(ms), **tot,
                        "seconds_per_doc": round(tot["seconds_total"] / n, 1), "cost_per_doc": round(tot["cost_usd"] / n, 4)}
    res["llm_usage"] = llm.usage
    res["cache"] = dict(llm_cache.stats)
    res["calls"] = calls
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    think = f"_think-{os.environ['LLM_THINKING_LEVEL'].lower()}" if os.environ.get("LLM_THINKING_LEVEL") else ""
    out = RUN_DIR / f"{args.gate}_{args.model}{think}{('_' + args.tag) if args.gate == 'screen' else ''}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["meter"], indent=1))
    print(f"[j5] wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
