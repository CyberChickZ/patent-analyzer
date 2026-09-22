"""What stays in the small per-job records, and what moves to funnel.json.

Two records carry the search funnel and both were unusable because of it,
measured on job 075b99c1:

  results.json   20.4 MB, of which search_stats 12.0 MB
  state.json     18.6 MB, of which phases.phase3 12.0 MB (the same search_stats)
                 plus a single round_done event whose payload is 6.6 MB
                 (cited_by_seed alone: 5.4 MB over 5,073 seeds)

state.json is rewritten to local disk AND to GCS on every heartbeat, so the
second one was the more expensive of the two. The full record lives in
funnel.json, served by GET /api/jobs/{id}/funnel.
"""

from __future__ import annotations

import json

# ── results.json size control ───────────────────────────────────────────────
#
# search_stats is by far the biggest thing a job produces: measured on job
# 075b99c1, results.json is 20.5 MB of which search is 12.0 MB — loop_rounds
# 6.6 MB (cited_by_seed alone is 5.4 MB over 5,073 seeds) and funnel_docs
# 4.5 MB over 10,097 documents. The Express proxy does `await res.json()` and
# re-serialises, so every byte is paid for twice on the way to the browser.
#
# Nothing is thrown away: the complete search_stats is written to funnel.json
# and served by GET /api/jobs/{id}/funnel. results.json keeps what the report
# and the UI actually read.

# Per-round lists of publication numbers — provenance, not display. Every one of
# these has a count beside it that stays.
FUNNEL_ROUND_KEYS = ("cited_by_seed", "bridge_by_paper", "seed_pubs", "bridge_pubs", "pool_pubs",
                      "expanded_pubs", "similar_by_seed", "similar_pubs", "neigh_oa_ids", "lens_pubs",
                      "cited_pubs", "forward_pubs")
FUNNEL_TOP_KEYS = ("pool",)


def slim_search_stats(stats: dict, job_id: str) -> dict:
    """The part of search_stats that results.json keeps.

    funnel_docs is projected down to the three fields the report's per-query
    table looks up (rank, elements) and restricted to documents a query actually
    returned or that were ranked — the other ~9,600 rows are only reachable
    through funnel.json, which is where the full record lives.
    """
    slim = {k: v for k, v in (stats or {}).items() if k not in FUNNEL_TOP_KEYS}

    rounds = []
    for r in slim.get("loop_rounds") or []:
        rr = {k: v for k, v in r.items() if k not in FUNNEL_ROUND_KEYS}
        lens = rr.get("lens")
        if isinstance(lens, dict):
            rr["lens"] = {k: v for k, v in lens.items() if k != "searches"}
        rounds.append(rr)
    if "loop_rounds" in slim:
        slim["loop_rounds"] = rounds

    referenced = {p for r in rounds for q in (r.get("queries") or []) for p in (q.get("pubs") or [])}
    full = stats.get("funnel_docs") or []
    slim["funnel_docs"] = [{"pub_num": d.get("pub_num"), "rank": d.get("rank"),
                            "elements": d.get("elements") or []}
                           for d in full
                           if d.get("pub_num") and (d.get("rank") or d.get("pub_num") in referenced)]
    slim["funnel_ref"] = {"file": "funnel.json", "endpoint": f"/api/jobs/{job_id}/funnel",
                          "n_funnel_docs": len(full), "n_pool": len(stats.get("pool") or []),
                          "note": "full per-document funnel, per-seed citation lists and the "
                                  "candidate pool live in funnel.json"}
    return slim


# ── event payloads ──────────────────────────────────────────────────────────
#
# A single round_done event carried 6.6 MB: cited_by_seed (5,073 seeds),
# seed_pubs (14,109), bridge_pubs (13,297), pool_pubs (10,097). Events are a
# progress feed — /events/{job_id} and the live UI — not a provenance store, and
# the provenance is already in funnel.json. Oversized payload values are
# replaced by a note saying what they were, so nothing looks silently absent.

EVENT_PAYLOAD_MAX = 4096          # bytes per payload value, serialised


def slim_event(evt: dict, limit: int = EVENT_PAYLOAD_MAX) -> dict:
    """A copy of `evt` whose payload values are all small."""
    payload = evt.get("payload")
    if not isinstance(payload, dict) or not payload:
        return evt
    out, dropped = {}, []
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
            continue
        try:
            size = len(json.dumps(v, default=str))
        except Exception:
            size = limit + 1
        if size <= limit:
            out[k] = v
        else:
            n = len(v) if isinstance(v, (list, dict, tuple)) else None
            out[k] = {"_omitted": True, "n": n, "bytes": size,
                      "where": "full record in funnel.json (GET /api/jobs/<id>/funnel)"}
            dropped.append(k)
    if not dropped:
        return evt
    return {**evt, "payload": out, "payload_trimmed": dropped}
