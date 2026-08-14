"""Precision stage for the wide pool: embedding shortlist, then an LLM screen.

Stage 1 — text-embedding-005 cosine of every element against every pool
document (title + abstract); the union of each element's top-K survives.
Stage 2 — batches of documents are shown to Gemini with the candidate
inventions and their elements; the model says whether the document is
worth reading in full and which elements it appears to touch.

The screen's question follows patent-search-pilot's CITATION_RECALL note:
"Its instruction changed from 'how many core elements does this disclose'
to 'how likely is it that reading this in full would be worth your time',
because a forty-word abstract cannot demonstrate several elements and
scoring it low for that is scoring it for being old." Pointwise batches
(not RankGPT's listwise sliding window, arXiv 2304.09542 §3.2) because the
output is a keep/drop set, not a permutation.
"""

from __future__ import annotations

import json
import os

import numpy as np

STAGE1_TOPK = int(os.environ.get("PRUNE_STAGE1_TOPK", "100"))
STAGE2_BATCH = int(os.environ.get("PRUNE_STAGE2_BATCH", "25"))
KEEP = int(os.environ.get("PRUNE_KEEP", "60"))

SCREEN_SCHEMA = {
    "type": "OBJECT",
    "properties": {"verdicts": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"i": {"type": "INTEGER"}, "worth_reading": {"type": "BOOLEAN"},
                       "elements": {"type": "ARRAY", "items": {"type": "STRING"}},
                       "reason": {"type": "STRING"}},
        "required": ["i", "worth_reading"]}}},
    "required": ["verdicts"],
}


def _doc_text(d: dict) -> str:
    return " ".join(x for x in (d.get("title") or "", (d.get("abstract") or d.get("snippet") or "")[:600]) if x).strip()


def stage1_embed(elements: list[dict], docs: list[dict], topk: int = STAGE1_TOPK,
                 embed_docs=None, embed_queries=None) -> tuple[list[int], dict]:
    """Indices of docs in the union of per-element top-k by cosine; each
    surviving doc gets `prune_cos` (max over elements)."""
    if not docs or not elements:
        return list(range(len(docs))), {"stage1_in": len(docs), "stage1_out": len(docs)}
    if embed_docs is None or embed_queries is None:
        from ..encoders import embed_docs as _ed, embed_queries as _eq
        embed_docs, embed_queries = embed_docs or _ed, embed_queries or _eq
    dv = np.asarray(embed_docs([_doc_text(d) or "untitled" for d in docs]), dtype=np.float32)
    qv = np.asarray(embed_queries([e["text"] for e in elements]), dtype=np.float32)
    dv /= np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
    qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9
    sim = qv @ dv.T                                    # elements × docs
    keep: set[int] = set()
    k = min(topk, len(docs))
    for row in sim:
        keep.update(int(i) for i in np.argpartition(-row, k - 1)[:k])
    best = sim.max(axis=0)
    best_el = sim.argmax(axis=0)
    for i in range(len(docs)):
        docs[i]["prune_cos"] = float(best[i])
        docs[i]["prune_best_element"] = elements[int(best_el[i])].get("id", "")
        docs[i]["prune_stage1"] = i in keep
    out = sorted(keep, key=lambda i: -best[i])
    return out, {"stage1_in": len(docs), "stage1_out": len(out),
                 "stage1_cut_cos": float(min(best[i] for i in keep)) if keep else None}


def _batch_prompt(candidates: list[dict], elements: list[dict], batch: list[tuple[int, dict]]) -> str:
    inv = "\n".join(f'- {c.get("id")}: {c.get("concept", "")[:300]}' for c in candidates) or "- (see elements)"
    els = "\n".join(f'  {e["id"]}: {e["text"][:220]}' for e in elements)
    rows = []
    for i, d in batch:
        meta = " · ".join(x for x in (d.get("pub_num") or "", str(d.get("year") or ""),
                                      ", ".join((d.get("cpc_codes") or [])[:3])) if x)
        rows.append(f'[{i}] {meta}\n  TITLE: {(d.get("title") or "")[:200]}\n  ABSTRACT: {(d.get("abstract") or d.get("snippet") or "")[:600]}')
    return f"""You screen prior-art candidates for a patent search. For EACH document decide whether
reading it in full would be worth an examiner's time for the invention below — an abstract
cannot prove several elements, so do not penalise short or old records; ask whether the
document plausibly discloses or teaches toward ANY element. Then list the element ids it
appears to touch (may be empty) and give a reason of at most 15 words.

CANDIDATE INVENTIONS:
{inv}
ELEMENTS:
{els}

DOCUMENTS:
{chr(10).join(rows)}

Answer with JSON {{"verdicts": [{{"i": <document index>, "worth_reading": true|false, "elements": ["<element id>", ...], "reason": "<=15 words"}}, ...]}} — one entry per document."""


async def stage2_llm(candidates: list[dict], elements: list[dict], docs: list[dict], idxs: list[int],
                     batch_size: int = STAGE2_BATCH, keep: int = KEEP, call=None) -> tuple[list[int], dict]:
    """LLM screen over the stage-1 survivors. Returns indices kept (≤keep),
    ordered by (#elements touched, cosine)."""
    if call is None:
        from app import llm as _llm
        _model = _llm.stage_model("screen")

        async def call(system, user, response_schema=None):
            # looked up at call time so evals/llm_cache.install() still wraps it
            return await _llm.call_llm(system, user, response_schema=response_schema, model=_model)
    import asyncio
    system = "You are a patent examiner screening search results. Output JSON only."
    verdict: dict[int, tuple[bool, list[str], str]] = {}
    batches = [[(i, docs[i]) for i in idxs[start:start + batch_size]] for start in range(0, len(idxs), batch_size)]
    sem = asyncio.Semaphore(int(os.environ.get("PRUNE_CONCURRENCY", "4")))

    async def _one(batch):
        async with sem:
            try:
                resp = await call(system, _batch_prompt(candidates, elements, batch), response_schema=SCREEN_SCHEMA)
                return json.loads(resp)
            except Exception:
                return {}

    results = await asyncio.gather(*(_one(b) for b in batches))
    calls = len(batches)
    for data in results:
        for v in (data.get("verdicts") or []):
            try:
                verdict[int(v["i"])] = (bool(v.get("worth_reading")), [str(x) for x in (v.get("elements") or [])],
                                        str(v.get("reason") or "")[:160])
            except (KeyError, TypeError, ValueError):
                continue
    kept = []
    for i in idxs:
        ok, els, why = verdict.get(i, (False, [], "no verdict returned"))
        docs[i]["prune_worth_reading"] = ok
        docs[i]["prune_elements"] = els
        docs[i]["prune_reason"] = why
        if ok:
            kept.append(i)
    kept.sort(key=lambda i: (-len(docs[i].get("prune_elements") or []), -float(docs[i].get("prune_cos") or 0.0)))
    return kept[:keep], {"stage2_in": len(idxs), "stage2_calls": calls, "stage2_worth": len(kept),
                         "stage2_out": min(len(kept), keep), "unanswered": sum(1 for i in idxs if i not in verdict)}


async def prune(candidates: list[dict], elements: list[dict], docs: list[dict], **kw) -> tuple[list[dict], dict]:
    """Full funnel: pool docs → stage 1 → stage 2 → ≤KEEP docs (dicts get
    prune_* fields). Stats carry every stage's counts."""
    idxs, s1 = stage1_embed(elements, docs, topk=kw.get("topk", STAGE1_TOPK),
                            embed_docs=kw.get("embed_docs"), embed_queries=kw.get("embed_queries"))
    kept, s2 = await stage2_llm(candidates, elements, docs, idxs, batch_size=kw.get("batch_size", STAGE2_BATCH),
                                keep=kw.get("keep", KEEP), call=kw.get("call"))
    return [docs[i] for i in kept], {"pool": len(docs), **s1, **s2}
