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

STAGE1_TOPK = int(os.environ.get("PRUNE_STAGE1_TOPK", "250"))
STAGE1_CAP = int(os.environ.get("PRUNE_STAGE1_CAP", "1200"))   # union cap → ≤ CAP/BATCH screen calls
STAGE2_BATCH = int(os.environ.get("PRUNE_STAGE2_BATCH", "40"))
KEEP = int(os.environ.get("PRUNE_KEEP", "60"))
GRAPH_SOURCES = {"citation_graph", "google_similar", "lens_bridge"}
# 300 publications cost 17.4 GiB of BigQuery (measured 2026-09-18: 9.31 + 8.07 on the bucketed
# pubs/claims tables), so 400 keeps a job's claims fetch near 23 GiB / $0.14 and every query
# under the 30 GiB ceiling; it still reads 6.7x more documents than the 60-cut would show.
STAGE3_IN = int(os.environ.get("PRUNE_STAGE3_IN", "400"))      # abstract survivors that get their claims read
STAGE3_FETCH_CHUNK = int(os.environ.get("PRUNE_STAGE3_CHUNK", "300"))
STAGE3_BATCH = int(os.environ.get("PRUNE_STAGE3_BATCH", "8"))  # claims are long: fewer per call

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
                 embed_docs=None, embed_queries=None, summary: str = "", cap: int = STAGE1_CAP) -> tuple[list[int], dict]:
    """Indices of docs in the union of per-element (and whole-summary) top-k
    by cosine, capped at `cap` by best cosine; each surviving doc gets
    `prune_cos` (max over queries). h1g H1-01: 3 of 4 gold families were
    dropped here at cos .56-.65 with top-100 per element (cut .62)."""
    if not docs or not elements:
        return list(range(len(docs))), {"stage1_in": len(docs), "stage1_out": len(docs)}
    if embed_docs is None or embed_queries is None:
        from ..encoders import embed_docs as _ed, embed_queries as _eq
        embed_docs, embed_queries = embed_docs or _ed, embed_queries or _eq
    dv = np.asarray(embed_docs([_doc_text(d) or "untitled" for d in docs]), dtype=np.float32)
    queries = [e["text"] for e in elements] + ([summary[:2000]] if summary else [])
    qv = np.asarray(embed_queries(queries), dtype=np.float32)
    dv /= np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
    qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9
    sim = qv @ dv.T                                    # elements × docs
    keep: set[int] = set()
    k = min(topk, len(docs))
    for row in sim:
        keep.update(int(i) for i in np.argpartition(-row, k - 1)[:k])
    best = sim.max(axis=0)
    best_el = sim.argmax(axis=0)
    labels = [e.get("id", "") for e in elements] + (["summary"] if summary else [])
    # candidates that arrived through the graph (citations / Google similar / paper→patent bridges)
    # were selected by structure, not text: they skip the embedding cut (leader H7 (3))
    protected = {i for i, d in enumerate(docs) if set(d.get("sources") or []) & GRAPH_SOURCES}
    out = sorted(keep | protected, key=lambda i: -best[i])[:cap + len(protected)]
    keep = set(out)
    for i in range(len(docs)):
        docs[i]["prune_cos"] = float(best[i])
        docs[i]["prune_best_element"] = labels[int(best_el[i])]
        docs[i]["prune_stage1"] = i in keep
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
document plausibly discloses or teaches toward ANY element, in ANY field (examiners cite
across fields: a software method can anticipate a step of a mechanical system and vice versa).
When in doubt, keep it: a wrong keep costs one read, a wrong drop loses the reference. Drop
only when the document clearly concerns a different problem. Then list the element ids it
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
        if not ok and set(docs[i].get("sources") or []) & GRAPH_SOURCES:
            # graph-selected candidates are not dropped on an abstract (h1h H1-01: the screen
            # dropped an examiner-cited reference as "software-based, not physical")
            ok, why = True, (why + " [kept: graph source]").strip()
        docs[i]["prune_worth_reading"] = ok
        docs[i]["prune_elements"] = els
        docs[i]["prune_reason"] = why
        if ok:
            kept.append(i)
    kept.sort(key=lambda i: (-len(docs[i].get("prune_elements") or []), -float(docs[i].get("prune_cos") or 0.0)))
    return kept[:keep], {"stage2_in": len(idxs), "stage2_calls": calls, "stage2_worth": len(kept),
                         "stage2_out": min(len(kept), keep), "unanswered": sum(1 for i in idxs if i not in verdict)}


CLAIMS_SCHEMA = SCREEN_SCHEMA


def _claims_prompt(elements: list[dict], batch: list[tuple[int, dict, str]]) -> str:
    els = "\n".join(f'  {e["id"]}: {e["text"][:220]}' for e in elements)
    rows = []
    for i, d, claims in batch:
        rows.append(f'[{i}] {d.get("pub_num") or ""} · {d.get("title") or ""}\n  CLAIMS: {claims[:4000]}')
    return f"""An examiner cites a reference for what it CLAIMS or DISCLOSES, not for its abstract.
You are reading the claims of each candidate below. For each document decide whether its claims
touch ANY element of the invention — the same subject matter in any wording, in any field, at any
level of generality (a broader claim reads on a narrower element). A document whose claims are in
a different art but recite the same mechanism still counts. When in doubt, keep it.

ELEMENTS:
{els}

DOCUMENTS:
{chr(10).join(rows)}

Answer with JSON {{"verdicts": [{{"i": <document index>, "worth_reading": true|false, "elements": ["<element id>", ...], "reason": "<=15 words"}}, ...]}} — one entry per document."""


async def stage3_claims(elements: list[dict], docs: list[dict], idxs: list[int], keep: int = KEEP,
                        n_in: int = STAGE3_IN, batch_size: int = STAGE3_BATCH,
                        fetch_claims=None, call=None) -> tuple[list[int], dict]:
    """Claims-level screen over the abstract survivors (leader, 2026-09-18).
    The abstract screen keeps thousands; the 60-cut then falls on an ordering
    that a 40-word abstract cannot inform. This reads the claims of the top
    `n_in` survivors — an examiner's own basis for citing — and reorders by
    (#elements the claims touch, cosine). Graph-sourced documents are kept.
    """
    import asyncio
    if not idxs:
        return [], {"stage3_in": 0, "stage3_calls": 0, "stage3_with_claims": 0, "stage3_out": 0}
    head = idxs[:n_in]
    if fetch_claims is None:
        from patent_analyzer.recall.bigquery_patents import fetch_by_pub_nums

        async def fetch_claims(pubs):
            out: dict[str, str] = {}
            for x in range(0, len(pubs), STAGE3_FETCH_CHUNK):
                got = await fetch_by_pub_nums(pubs[x:x + STAGE3_FETCH_CHUNK], with_claims=True)
                out.update({k: (v.get("claims_text") or "") for k, v in got.items()})
            return out
    pubs = [docs[i].get("pub_num") for i in head if docs[i].get("pub_num")]
    if not pubs:                       # papers only: nothing to read claims from
        return head[:keep], {"stage3_in": len(head), "stage3_calls": 0, "stage3_with_claims": 0, "stage3_out": min(len(head), keep)}
    try:
        claims = await fetch_claims(pubs)
    except Exception as exc:
        return head[:keep], {"stage3_in": len(head), "stage3_calls": 0, "stage3_with_claims": 0,
                             "stage3_out": min(len(head), keep), "stage3_error": f"{type(exc).__name__}: {exc}"[:160]}
    from patent_analyzer.recall.bigquery_patents import _canon_pub
    have = [(i, docs[i], claims.get(_canon_pub(docs[i].get("pub_num") or ""), "")) for i in head]
    have = [(i, d, c) for i, d, c in have if c]
    if call is None:
        from app import llm as _llm
        _model = _llm.stage_model("screen")

        async def call(system, user, response_schema=None):
            return await _llm.call_llm(system, user, response_schema=response_schema, model=_model)
    system = "You are a US patent examiner reading claims. Output JSON only."
    batches = [have[x:x + batch_size] for x in range(0, len(have), batch_size)]
    sem = asyncio.Semaphore(int(os.environ.get("PRUNE_CONCURRENCY", "4")))

    async def _one(batch):
        async with sem:
            try:
                return json.loads(await call(system, _claims_prompt(elements, batch), response_schema=CLAIMS_SCHEMA))
            except Exception:
                return {}
    results = await asyncio.gather(*(_one(b) for b in batches))
    verdict: dict[int, tuple[bool, list[str], str]] = {}
    for data in results:
        for v in (data.get("verdicts") or []):
            try:
                verdict[int(v["i"])] = (bool(v.get("worth_reading")), [str(x) for x in (v.get("elements") or [])],
                                        str(v.get("reason") or "")[:160])
            except (KeyError, TypeError, ValueError):
                continue
    for i, d, _c in have:
        ok, els, why = verdict.get(i, (None, [], "no claims verdict"))
        if ok is None:
            continue
        d["claims_worth_reading"] = ok
        d["claims_elements"] = els
        d["claims_reason"] = why
    def _rank(i):
        d = docs[i]
        n_claims = len(d.get("claims_elements") or [])
        graph = 1 if set(d.get("sources") or []) & GRAPH_SOURCES else 0
        read = d.get("claims_worth_reading")
        return (-(1 if read else 0), -n_claims, -graph, -len(d.get("prune_elements") or []), -float(d.get("prune_cos") or 0.0))
    ordered = sorted(idxs, key=_rank)
    return ordered[:keep], {"stage3_in": len(head), "stage3_calls": len(batches),
                            "stage3_with_claims": len(have), "stage3_kept": sum(1 for i in head if docs[i].get("claims_worth_reading")),
                            "stage3_out": min(len(ordered), keep)}


async def prune(candidates: list[dict], elements: list[dict], docs: list[dict], **kw) -> tuple[list[dict], dict]:
    """Full funnel: pool docs → stage 1 → stage 2 → ≤KEEP docs (dicts get
    prune_* fields). Stats carry every stage's counts."""
    idxs, s1 = stage1_embed(elements, docs, topk=kw.get("topk", STAGE1_TOPK),
                            embed_docs=kw.get("embed_docs"), embed_queries=kw.get("embed_queries"),
                            summary=kw.get("summary", ""), cap=kw.get("cap", STAGE1_CAP))
    keep = kw.get("keep", KEEP)
    # stage 2 orders by (#elements, cosine) but keeps everything worth reading, so stage 3 can
    # read further down the list than the 60-cut would have allowed
    kept, s2 = await stage2_llm(candidates, elements, docs, idxs, batch_size=kw.get("batch_size", STAGE2_BATCH),
                                keep=kw.get("stage2_keep", max(keep, STAGE3_IN)), call=kw.get("call"))
    s3 = {}
    if kw.get("claims_screen", os.environ.get("PRUNE_CLAIMS", "1") != "0") and kept:
        kept, s3 = await stage3_claims(elements, docs, kept, keep=keep, n_in=kw.get("stage3_in", STAGE3_IN),
                                       fetch_claims=kw.get("fetch_claims"), call=kw.get("claims_call") or kw.get("call"))
    else:
        kept = kept[:keep]
    return [docs[i] for i in kept], {"pool": len(docs), **s1, **s2, **s3}
