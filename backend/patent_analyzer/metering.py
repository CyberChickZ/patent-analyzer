"""Per-job, per-phase accounting: wall-clock, LLM calls and tokens, external
calls, and an estimated dollar cost.

One process runs one job at a time (app.main._pipeline_worker is a single
worker pulling from a serial queue), so module-level counters are per-job.
`start_run(job_id)` resets them, and only when the job actually changes — a
HITL resume re-enters the pipeline for the *same* job in the same process and
must keep what the earlier phases spent.

Cost is an ESTIMATE from published list prices, not a bill. Vertex discounts,
context-caching credits and the free tiers are not modelled, and a token count
is only as good as what the API reported (`usage_metadata` is occasionally
absent, in which case that call contributes calls but no tokens).
"""

from __future__ import annotations

import copy
import time

# ── the rate card, and where every number comes from ─────────────────────────
#
# All figures read off https://cloud.google.com/vertex-ai/generative-ai/pricing
# ("Agent Platform Pricing"), Standard tier, **Global** region, fetched
# 2026-09-18. Global is the right column because app/llm.py builds its client
# with `location=os.getenv("VERTEX_LOCATION", "global")`; a deployment that
# pins a region pays the Non-global column (+10%) and this table under-counts.
#
# Verbatim from that page, 2026-09-18:
#   "Gemini 3.8 Flash* through December 31, 2026 | Input (text, image, video,
#    audio) Global $0.75 ... Text output (response and reasoning) Global $3.75"
#   "Gemini 3.8 Flash Starting January 1, 2027 | Input ... Global $1.50 ...
#    Text output (response and reasoning) Global $7.50"
#   "Gemini 3.1 Flash-Lite | Input (text, image, video) Global $0.25 ... Text
#    output (response and reasoning) Global $1.50"
#   "Gemini 2.5 Pro | Input (text, image, video, audio) $1.25 [<=200K] $2.50
#    [>200K] ... Text output (response and reasoning) $10.00 $15.00"
# Thought tokens are not a separate line anywhere on the page: every output row
# reads "Text output (response and reasoning)", so reasoning bills as output.
#
# USD per 1,000,000 tokens: (input, output).
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),        # introductory price, ends 2026-12-31 -> (1.50, 7.50)
    "gemini-2.5-pro": (1.25, 10.00),         # the <=200K-context tier; see _TIER_NOTE
    "gemini-3.1-flash-lite": (0.25, 1.50),
}
# When the introductory Gemini 3.x Flash price lapses, this table is wrong by 2x.
PRICE_REVIEW_DATE = "2027-01-01"
_TIER_NOTE = ("gemini-2.5-pro is costed at its <=200K-context tier; the per-model meter "
              "aggregates tokens across calls, so a single >200K call cannot be told apart "
              "and is under-costed (its true rate is $2.50/$15.00 per 1M).")

# https://cloud.google.com/bigquery/pricing, fetched 2026-09-18: "Queries
# (on-demand) | 0 tebibyte to 1 tebibyte Free per 1 month / account | 1 tebibyte
# and above $6.25 / 1 tebibyte, per 1 month / account. The first 1 TiB per month
# is free." The free TiB is account-wide and not modelled here, so a small
# month's BigQuery line is an over-estimate.
BQ_USD_PER_TIB = 6.25

# Same Vertex page, embeddings table: "Embeddings for Text (Excluding Gemini
# Embedding) | Input | Global | Online requests $0.000025 | Batch requests
# $0.00002 | Output | Online requests No charge" — priced per 1,000 *count*
# (input tokens). text-embedding-005 (patent_analyzer/encoders.py) is in that
# row. $0.000025/1k count == $0.025 per 1M.
EMBED_USD_PER_MTOK = 0.025

# SerpAPI is on the free tier: no marginal cost, but the call count is the thing
# that actually runs out, so it is counted like a priced resource.

_UNPRICED_NOTE = "model not in the price table — counted, not costed"

# ── live counters ─────────────────────────────────────────────────────────────

external: dict[str, int] = {}          # channel -> outbound calls that hit the network
bq: dict[str, float] = {"queries": 0, "bytes_billed": 0.0}

_job_id: str | None = None
_phases: dict[str, dict] = {}          # phase -> metrics, in the order they ran
_mark_t0: float = time.monotonic()
_mark_base: dict = {}


def count(channel: str, n: int = 1) -> None:
    """One outbound call on `channel`. Cache hits are counted separately (pass
    e.g. "serpapi:cached") so a cheap re-run is not mistaken for a cheap job."""
    external[channel] = external.get(channel, 0) + n


def count_bq(bytes_billed: float) -> None:
    bq["queries"] += 1
    bq["bytes_billed"] += float(bytes_billed or 0)


def _llm_usage() -> dict:
    from app.llm import usage  # imported late: patent_analyzer must not need app/
    return copy.deepcopy(usage)


def snapshot() -> dict:
    return {"t": time.monotonic(), "llm": _llm_usage(),
            "external": dict(external), "bq": dict(bq)}


def cost_usd(model: str, prompt_tokens: int, output_tokens: int, thought_tokens: int) -> float:
    """Thought tokens bill as output."""
    price = PRICES.get(model)
    if price is None:
        return 0.0
    pin, pout = price
    return (prompt_tokens * pin + (output_tokens + thought_tokens) * pout) / 1_000_000


def _delta(before: dict, after: dict) -> dict:
    """Everything spent between two snapshots, per model and per channel."""
    models: dict[str, dict] = {}
    for model, a in after["llm"].items():
        b = before["llm"].get(model, {})
        d = {k: a.get(k, 0) - b.get(k, 0) for k in
             ("calls", "prompt_tokens", "output_tokens", "thought_tokens", "errors_429")}
        d["seconds"] = round(a.get("seconds", 0.0) - b.get("seconds", 0.0), 1)
        if d["calls"] <= 0 and d["prompt_tokens"] <= 0:
            continue
        d["cost_usd"] = round(cost_usd(model, d["prompt_tokens"], d["output_tokens"], d["thought_tokens"]), 4)
        if model not in PRICES:
            d["note"] = _UNPRICED_NOTE
        models[model] = d

    ext = {k: v - before["external"].get(k, 0) for k, v in after["external"].items()}
    ext = {k: v for k, v in ext.items() if v}

    bq_queries = int(after["bq"]["queries"] - before["bq"]["queries"])
    bq_bytes = after["bq"]["bytes_billed"] - before["bq"]["bytes_billed"]
    bq_cost = round(bq_bytes / 2 ** 40 * BQ_USD_PER_TIB, 4)

    llm_cost = round(sum(m["cost_usd"] for m in models.values()), 4)
    return {
        "seconds": round(after["t"] - before["t"], 1),
        "llm_calls": sum(m["calls"] for m in models.values()),
        "llm": models,
        "external_calls": sum(ext.values()),
        "external": ext,
        "bigquery": {"queries": bq_queries, "gib_billed": round(bq_bytes / 2 ** 30, 3), "cost_usd": bq_cost},
        "cost_usd": round(llm_cost + bq_cost, 4),
    }


# ── phase accounting ──────────────────────────────────────────────────────────

def start_run(job_id: str) -> None:
    """Begin (or re-enter) a job. Re-entering the same job — what a HITL resume
    does — keeps the phases already recorded; a different job starts clean."""
    global _job_id, _phases, _mark_t0, _mark_base
    if job_id == _job_id:
        _mark_t0 = time.monotonic()
        _mark_base = snapshot()
        return
    _job_id = job_id
    _phases = {}
    external.clear()
    bq.update(queries=0, bytes_billed=0.0)
    _mark_t0 = time.monotonic()
    _mark_base = snapshot()


def mark(phase: str) -> dict:
    """Close out `phase`: everything spent since the previous mark.

    Idempotent. A gate that pauses re-executes its whole body on resume, so
    without this the second pass would overwrite a real phase with the ~0 it
    measured across the pause.
    """
    global _mark_base
    if phase in _phases:
        return _phases[phase]
    if not _mark_base:
        _mark_base = snapshot()
    now = snapshot()
    d = _delta(_mark_base, now)
    _mark_base = now
    _phases[phase] = d
    return d


def phases() -> dict:
    return copy.deepcopy(_phases)


def totals(ph: dict | None = None) -> dict:
    ph = _phases if ph is None else ph
    models: dict[str, dict] = {}
    for m in ph.values():
        for name, d in (m.get("llm") or {}).items():
            acc = models.setdefault(name, {"calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                                           "thought_tokens": 0, "errors_429": 0, "cost_usd": 0.0})
            for k in acc:
                acc[k] = round(acc[k] + d.get(k, 0), 4) if k == "cost_usd" else acc[k] + d.get(k, 0)
    ext: dict[str, int] = {}
    for m in ph.values():
        for k, v in (m.get("external") or {}).items():
            ext[k] = ext.get(k, 0) + v
    return {
        "seconds": round(sum(m.get("seconds", 0) for m in ph.values()), 1),
        "llm_calls": sum(m.get("llm_calls", 0) for m in ph.values()),
        "llm": models,
        "external_calls": sum(ext.values()),
        "external": ext,
        "bigquery": {"queries": sum((m.get("bigquery") or {}).get("queries", 0) for m in ph.values()),
                     "gib_billed": round(sum((m.get("bigquery") or {}).get("gib_billed", 0) for m in ph.values()), 3),
                     "cost_usd": round(sum((m.get("bigquery") or {}).get("cost_usd", 0) for m in ph.values()), 4)},
        "cost_usd": round(sum(m.get("cost_usd", 0) for m in ph.values()), 4),
    }


def report(ph: dict | None = None) -> dict:
    """The block that goes into results.json and the bottom of the report."""
    ph = phases() if ph is None else ph
    return {"job_id": _job_id, "prices_usd_per_mtok": {k: {"input": v[0], "output": v[1]} for k, v in PRICES.items()},
            "bigquery_usd_per_tib": BQ_USD_PER_TIB,
            "note": "Estimated from list prices; thought tokens billed as output. Not a bill.",
            "phases": ph, "totals": totals(ph)}
