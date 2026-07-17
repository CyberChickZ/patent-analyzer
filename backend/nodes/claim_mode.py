"""Claim Mode: accept raw claim text, search for matching prior art patents.

This is an alternative entry point — instead of uploading a PDF,
the user pastes claim text and we search for existing patents with
similar claims.

Pipeline: parse_claims → search_bigquery + search_existing → evaluate → report
"""

import asyncio
import re
from datetime import datetime, timezone

from state import GraphState


def _parse_claim_limitations(claim_text: str) -> dict:
    """Parse a patent claim into preamble + individual limitations."""
    lines = claim_text.strip().split("\n")
    clean = " ".join(l.strip() for l in lines)

    # Try to split on "comprising:" or "including:" or "wherein:"
    preamble = ""
    body = clean
    for split_word in ["comprising:", "including:", "consisting of:", "wherein:"]:
        if split_word in clean.lower():
            idx = clean.lower().index(split_word)
            preamble = clean[:idx + len(split_word)].strip()
            body = clean[idx + len(split_word):].strip()
            break

    # Split limitations on semicolons or numbered patterns
    limitations = []
    # Try semicolon split first
    parts = re.split(r';\s*', body)
    if len(parts) >= 2:
        limitations = [p.strip().rstrip('.') for p in parts if len(p.strip()) > 10]
    else:
        # Try splitting on "a)", "b)", "(a)", "(b)", or "i)", "ii)"
        parts = re.split(r'(?:^|\s)(?:\(?[a-z]\)|\(?[ivx]+\))\s', body)
        limitations = [p.strip().rstrip('.') for p in parts if len(p.strip()) > 10]

    if not limitations:
        # Fallback: treat the whole body as one limitation
        limitations = [body.strip().rstrip('.')]

    limitations = [sub for lim in limitations for sub in _split_sub_features(lim)]

    return {
        "preamble": preamble,
        "limitations": limitations,
        "full_text": claim_text.strip(),
    }


def _split_sub_features(text: str) -> list[str]:
    """Second-pass split within a limitation, mirroring how EPO examiners
    break features: each wherein clause and each dash-list item stands alone
    (FiNE-Patents extraction eval — recall was the gap, precision had slack)."""
    parts = re.split(r',?\s+(?=wherein\b)', text)
    out = []
    for p in parts:
        for q in re.split(r'(?:^|(?<=\s))-\s+', p):
            out.extend(re.split(r',\s+(?=and\s+\w+ing\b)', q))
    cleaned = [x.strip(' ,;-') for x in out]
    return [x for x in cleaned if len(x) > 10] or ([text.strip()] if text.strip() else [])


def _split_preamble(preamble: str) -> list[str]:
    """Examiners often count the device/method name and the trailing
    'the method comprising…' bridge as separate features."""
    p = re.sub(r'^\s*\d+\s*\.\s*', '', preamble or '').strip()
    if not p:
        return []
    parts = re.split(r',\s+(?=the\s+\w+\s+(?:being|compris|configur|perform))', p)
    parts = [x.strip(' ,') for x in parts if len(x.strip()) > 3]
    return [x for x in parts if not re.match(r'^the\s+\w+\s+comprising:?$', x)]


async def claim_parse_node(state: GraphState) -> dict:
    """Parse claim text into structured limitations."""
    claim_text = state.get("document_text", "")
    if not claim_text:
        return {
            "status": "error",
            "error": "No claim text provided",
            "events": [_evt("error", "No claim text provided")],
        }

    parsed = _parse_claim_limitations(claim_text)
    limitations = parsed["limitations"]

    return {
        "phase": "phase1",
        "status_determination": "Present",
        "doc_type": "patent_draft",
        "input_mode": "claim_text",
        "summary": claim_text[:2000],
        "category": "Process",
        "innovation_axes": [{"axis_name": f"Limitation {i+1}", "axis_description": lim[:200]}
                            for i, lim in enumerate(limitations)],
        "checklist": [{"id": f"lim{i+1}", "criterion": lim, "weight": 1.0 / max(len(limitations), 1)}
                      for i, lim in enumerate(limitations)],
        "events": [
            _evt("info", f"Parsed claim into {len(limitations)} limitations"),
            _evt("info", f"Preamble: {parsed['preamble'][:200]}"),
        ],
        "phase_results": {"phase1": {"status": "completed", "data": {
            "mode": "claim",
            "limitations": len(limitations),
        }}},
    }


async def claim_search_node(state: GraphState) -> dict:
    """Search BigQuery Patents + existing channels for claim-level matches."""
    from patent_analyzer.recall import pool as recall_pool
    from patent_analyzer.recall import semantic_scholar as ch_ss
    from patent_analyzer.recall import serpapi as ch_serpapi
    from patent_analyzer.recall.bigquery_patents import search_by_limitations, search_claims
    from patent_analyzer.semantic_search import rerank_docs
    from patent_analyzer.searcher import download_pdf
    from pathlib import Path

    limitations = [c["criterion"] for c in state.get("checklist", [])]
    summary = state.get("summary", "")
    output_dir = state.get("output_dir", "/tmp/outputs/default")

    events = []

    # 1. BigQuery Patents — claim-level search
    _evt_add(events, "info", "Searching BigQuery Patents by limitations...")
    bq_cands, bq_err = await search_by_limitations(limitations, limit_per_limitation=15)
    if bq_err:
        _evt_add(events, "warning", f"BigQuery: {bq_err[:200]}")
    _evt_add(events, "channel_done", f"BigQuery Patents: {len(bq_cands)} candidates")

    # 2. SerpAPI Google Patents — keyword search from claim text
    serpapi_lock = asyncio.Semaphore(1)

    async def _throttled_patent(q: str):
        async with serpapi_lock:
            res = await ch_serpapi.search_patents(q, max_pages=1)
            await asyncio.sleep(1.5)
            return res

    serp_cands = []
    # Use first 2 limitations as search queries
    for lim in limitations[:2]:
        short_q = " ".join(lim.split()[:15])
        cands, err = await _throttled_patent(short_q)
        serp_cands.extend(cands)
    _evt_add(events, "channel_done", f"SerpAPI Patents: {len(serp_cands)} candidates")

    # 3. Semantic Scholar — for NPL coverage
    ss_query = " ".join(summary.split()[:30])
    ss_cands, ss_err = await ch_ss.search(ss_query, limit=30)
    _evt_add(events, "channel_done", f"Semantic Scholar: {len(ss_cands)} candidates")

    # Pool & dedup
    channel_results = {
        "bigquery_patents": bq_cands,
        "serpapi_patents": serp_cands,
        "semantic_scholar": ss_cands,
    }
    pooled = recall_pool.pool_and_dedupe(channel_results)
    _evt_add(events, "info", f"Pool: {len(pooled)} unique candidates")

    if not pooled:
        return {
            "phase": "phase3",
            "status": "failed_recall",
            "error": "No matching patents found",
            "search_results": [],
            "ranked_candidates": [],
            "events": events,
        }

    all_docs = recall_pool.candidates_to_legacy_docs(pooled)

    # Rerank by claim text similarity
    ranked = rerank_docs(summary, all_docs, limit=30)
    _evt_add(events, "info", f"Ranked: top {len(ranked)} candidates")

    # Download PDFs
    job_dir = Path(output_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    dl_count = 0
    for i, doc in enumerate(ranked[:20]):
        pdf_url = recall_pool.resolve_pdf_url(doc)
        if not pdf_url:
            continue
        try:
            local = download_pdf(pdf_url, job_dir, f"claim_pa_{i:03d}.pdf")
            if local:
                doc["local_pdf"] = local
                dl_count += 1
        except Exception:
            pass

    return {
        "phase": "phase3",
        "search_results": [{"title": d.get("title", ""), "match_type": d.get("match_type", "")}
                           for d in all_docs],
        "ranked_candidates": ranked,
        "search_stats": {"total_unique": len(all_docs), "downloaded": dl_count},
        "events": events,
        "phase_results": {"phase3": {"status": "completed", "data": {
            "total": len(all_docs), "ranked": len(ranked),
        }}},
    }


def _evt(kind: str, message: str) -> dict:
    return {"ts": datetime.now(timezone.utc).isoformat(),
            "phase": "phase1", "kind": kind, "message": message}


def _evt_add(events: list, kind: str, message: str):
    events.append({"ts": datetime.now(timezone.utc).isoformat(),
                    "phase": "phase3", "kind": kind, "message": message})
