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

MAX_DOC_CHARS = 120_000



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
                     concurrency: int = 3) -> dict[str, dict]:
    """Per instance: elements + one scored result per cited reference."""
    out_path = RUN_DIR / "doc_results.json"
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

    try:
        await asyncio.gather(*(one(s) for s in samples))
    finally:
        out_path.write_text(json.dumps(done, ensure_ascii=False))
    return done



VARIANTS = {
    "base (min_cover 1.0, allow_missing 0, score>0, preamble in)": {},
    "score>=2 only (Present, not Partial)": {"min_score": 2},
    "allow_missing 1": {"allow_missing": 1},
    "min_cover 0.8": {"min_cover": 0.8},
    "single_partial_103 0.7 (PANORAMA C.5.3 a)": {"single_partial_103": 0.7},
    "no preamble elements": {"_drop_preamble": True},
    "no preamble + allow_missing 1": {"_drop_preamble": True, "allow_missing": 1},
    "no preamble + single_partial_103 0.7": {"_drop_preamble": True, "single_partial_103": 0.7},
}



if __name__ == "__main__":
    asyncio.run(main())
