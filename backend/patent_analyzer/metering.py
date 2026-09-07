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
# A price is a schedule, not a number. gemini-3.8-flash — the global model, and
# the one the deep read runs on — is on an introductory rate that DOUBLES on
# 2027-01-01, so a table holding one tuple per model is a table that is silently
# 2x wrong from New Year's Day. Each entry is (effective_from, input, output) in
# USD per 1,000,000 tokens, oldest first; `price_of` picks the row in force.
#
# The far-past first entry is deliberate: it means "as far back as this table
# claims to know", not a date anything happened.
PRICE_SCHEDULE: dict[str, list[tuple[str, float, float]]] = {
    # "through December 31, 2026 ... Global $0.75 / $3.75" then
    # "Starting January 1, 2027 ... Global $1.50 / $7.50"
    "gemini-3.8-flash": [("2000-01-01", 0.75, 3.75), ("2027-01-01", 1.50, 7.50)],
    # Same note on the page covers 3.7 and 3.6 Flash; listed so switching the
    # global model does not silently drop off the price table.
    "gemini-3.7-flash": [("2000-01-01", 0.75, 3.75), ("2027-01-01", 1.50, 7.50)],
    "gemini-3.6-flash": [("2000-01-01", 0.75, 3.75), ("2027-01-01", 1.50, 7.50)],
    "gemini-2.5-pro": [("2000-01-01", 1.25, 10.00)],        # <=200K tier; see _TIER_NOTE
    "gemini-3.1-flash-lite": [("2000-01-01", 0.25, 1.50)],
}


def price_of(model: str, on: str | None = None) -> tuple[float, float] | None:
    """The (input, output) rate in force for `model` on `on` (ISO date, default
    today UTC). None for a model this table does not price."""
    rows = PRICE_SCHEDULE.get(model)
    if not rows:
        return None
    from datetime import datetime, timezone
    day = on or datetime.now(timezone.utc).date().isoformat()
    current = None
    for start, pin, pout in rows:
        if start <= day:
            current = (pin, pout)
    return current or (rows[0][1], rows[0][2])


def upcoming_price_changes(after: str | None = None) -> list[dict]:
    """Every scheduled change still ahead of `after` (default today), so the
    ledger and the quota panel can warn before the bill does."""
    from datetime import datetime, timezone
    day = after or datetime.now(timezone.utc).date().isoformat()
    out = []
    for model, rows in PRICE_SCHEDULE.items():
        for i, (start, pin, pout) in enumerate(rows):
            if start > day and i:
                was_in, was_out = rows[i - 1][1], rows[i - 1][2]
                out.append({"model": model, "effective_from": start,
                            "input_usd_per_mtok": pin, "output_usd_per_mtok": pout,
                            "multiple": round(pout / was_out, 2) if was_out else None,
                            "from_input": was_in, "from_output": was_out})
    return sorted(out, key=lambda c: (c["effective_from"], c["model"]))


# Today's view of the schedule. Kept because the report, the evals and the quota
# panel all read a flat {model: (in, out)}; costing goes through price_of() so a
# process alive across a price change follows the schedule rather than this.
PRICES: dict[str, tuple[float, float]] = {m: price_of(m) for m in PRICE_SCHEDULE}   # type: ignore[misc]
# The date the table itself should be re-checked against the published page.
PRICE_REVIEW_DATE = min((c["effective_from"] for c in upcoming_price_changes()), default="2027-01-01")
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
embed: dict[str, dict] = {}            # model -> {"requests": n, "texts": n, "est_tokens": n}

# Append-only log of calls that did not go cleanly: a retry, an outright
# failure, a quota that ran out, or a fall back to a cheaper/lossier path. This
# is the third question the ledger has to answer ("which call failed or
# degraded"), and it is the one no counter can reconstruct after the fact.
# Sliced by index between two snapshots, so a phase gets exactly its own.
incidents: list[dict] = []

RETRY, FAILED, DEGRADED, EXHAUSTED = "retry", "failed", "degraded", "exhausted"


def incident(source: str, kind: str, detail: str = "") -> None:
    """`source` is a model id or a channel name; `kind` is one of the four
    constants above. Never raises: accounting must not break the pipeline."""
    try:
        incidents.append({"t": round(time.monotonic(), 1), "source": str(source),
                          "kind": str(kind), "detail": str(detail)[:200]})
    except Exception:
        pass

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
    # Also accumulate across jobs and instances, in MiB because the KV counter
    # is integer-only: the free 1 TiB/month is the one BigQuery number that
    # behaves like a quota, and /quota reports what is left of it.
    try:
        from .cache import kv
        from .runtime_state import month_key
        kv().incr("runtime", f"usage:bigquery_mib:{month_key()}", by=int(float(bytes_billed or 0) // 2 ** 20))
    except Exception:
        pass


def count_embed(model: str, texts: int, chars: int, billable_chars: int | None = None) -> None:
    """One embedding request. Vertex prices these per 1,000 "count" (input
    tokens) but the embeddings endpoint returns no token count, so tokens are
    ESTIMATED at 4 chars/token from the characters we sent — or from the
    provider's own billable_character_count when the response carries one."""
    m = embed.setdefault(model, {"requests": 0, "texts": 0, "chars": 0, "est_tokens": 0})
    c = int(billable_chars if billable_chars is not None else chars or 0)
    m["requests"] += 1
    m["texts"] += int(texts or 0)
    m["chars"] += c
    m["est_tokens"] += -(-c // 4)


def _llm_usage() -> dict:
    from app.llm import usage  # imported late: patent_analyzer must not need app/
    return copy.deepcopy(usage)


def snapshot() -> dict:
    return {"t": time.monotonic(), "llm": _llm_usage(),
            "external": dict(external), "bq": dict(bq),
            "embed": copy.deepcopy(embed), "incidents": len(incidents)}


def cost_usd(model: str, prompt_tokens: int, output_tokens: int, thought_tokens: int,
             on: str | None = None) -> float:
    """Thought tokens bill as output. `on` (ISO date, default today) selects the
    rate from the schedule — the tokens were spent on a day, and the price the
    model carried that day is the one that gets billed."""
    price = price_of(model, on)
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
        if price_of(model) is None:
            d["note"] = _UNPRICED_NOTE
        models[model] = d

    ext = {k: v - before["external"].get(k, 0) for k, v in after["external"].items()}
    ext = {k: v for k, v in ext.items() if v}

    bq_queries = int(after["bq"]["queries"] - before["bq"]["queries"])
    bq_bytes = after["bq"]["bytes_billed"] - before["bq"]["bytes_billed"]
    bq_cost = round(bq_bytes / 2 ** 40 * BQ_USD_PER_TIB, 4)

    emb: dict[str, dict] = {}
    for model, a in (after.get("embed") or {}).items():
        b = (before.get("embed") or {}).get(model, {})
        d = {k: a.get(k, 0) - b.get(k, 0) for k in ("requests", "texts", "chars", "est_tokens")}
        if d["requests"] <= 0:
            continue
        d["cost_usd"] = round(d["est_tokens"] / 1_000_000 * EMBED_USD_PER_MTOK, 4)
        emb[model] = d
    emb_cost = round(sum(m["cost_usd"] for m in emb.values()), 4)

    inc = incidents[before.get("incidents", 0):after.get("incidents", len(incidents))]

    llm_cost = round(sum(m["cost_usd"] for m in models.values()), 4)
    return {
        "seconds": round(after["t"] - before["t"], 1),
        "llm_calls": sum(m["calls"] for m in models.values()),
        "llm": models,
        "external_calls": sum(ext.values()),
        "external": ext,
        "bigquery": {"queries": bq_queries, "gib_billed": round(bq_bytes / 2 ** 30, 3), "cost_usd": bq_cost},
        "embedding": emb,
        "incidents": [dict(i) for i in inc],
        "failures": sum(1 for i in inc if i["kind"] in (FAILED, EXHAUSTED)),
        "degradations": sum(1 for i in inc if i["kind"] == DEGRADED),
        "retries": sum(1 for i in inc if i["kind"] == RETRY),
        "cost_usd": round(llm_cost + bq_cost + emb_cost, 4),
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
    embed.clear()
    incidents.clear()
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
    emb: dict[str, dict] = {}
    for m in ph.values():
        for name, d in (m.get("embedding") or {}).items():
            acc = emb.setdefault(name, {"requests": 0, "texts": 0, "chars": 0,
                                        "est_tokens": 0, "cost_usd": 0.0})
            for k in acc:
                acc[k] = round(acc[k] + d.get(k, 0), 4) if k == "cost_usd" else acc[k] + d.get(k, 0)
    return {
        "seconds": round(sum(m.get("seconds", 0) for m in ph.values()), 1),
        "llm_calls": sum(m.get("llm_calls", 0) for m in ph.values()),
        "llm": models,
        "external_calls": sum(ext.values()),
        "external": ext,
        "embedding": emb,
        "failures": sum(m.get("failures", 0) for m in ph.values()),
        "degradations": sum(m.get("degradations", 0) for m in ph.values()),
        "retries": sum(m.get("retries", 0) for m in ph.values()),
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


# ── the ledger ────────────────────────────────────────────────────────────────

def _incident_counts(m: dict) -> dict[str, dict]:
    """Incidents of one phase, grouped by the source that raised them."""
    out: dict[str, dict] = {}
    for i in m.get("incidents") or []:
        c = out.setdefault(i.get("source", "?"), {"retries": 0, "failures": 0, "degradations": 0})
        kind = i.get("kind")
        if kind == RETRY:
            c["retries"] += 1
        elif kind == DEGRADED:
            c["degradations"] += 1
        else:
            c["failures"] += 1
    return out


def _attach(row: dict, counts: dict[str, dict]) -> dict:
    """A row owns an incident when the incident's source is the row's resource,
    or its prefix — SerpAPI reports as "serpapi", the channel is
    "serpapi:google_patents"."""
    acc = {"retries": 0, "failures": 0, "degradations": 0}
    for src, c in counts.items():
        if row["name"] == src or row["name"].startswith(src + ":") or src.startswith(row["name"] + ":"):
            for k in acc:
                acc[k] += c[k]
    row.update(acc)
    return row


def _change_caveat(c: dict) -> str:
    """One scheduled price change, spelled out. A run costed on or after the
    date is costed at the new rate; one costed before it is not."""
    mult = f" ({c['multiple']}x output)" if c.get("multiple") else ""
    return (f"{c['model']} goes from ${c['from_input']}/${c['from_output']} to "
            f"${c['input_usd_per_mtok']}/${c['output_usd_per_mtok']} per 1M tokens on "
            f"{c['effective_from']}{mult} — costs before and after that date are not comparable.")


def ledger(ph: dict | None = None) -> dict:
    """Every line of spend on one job, flat enough to sort and total.

    Answers the three questions a finished job has to answer: what did it cost
    (`totals`), which step was the most expensive (`most_expensive`), and which
    call failed or degraded (`incidents`, and the counts on each row).

    One row = one (phase, resource). `kind` is model | embedding | bigquery |
    external; only the first three carry a dollar figure, external channels are
    free at the margin but are what actually runs out (see /quota).
    """
    from datetime import datetime, timezone
    ph = phases() if ph is None else ph
    rows: list[dict] = []
    all_incidents: list[dict] = []

    for phase, m in ph.items():
        counts = _incident_counts(m)
        for i in m.get("incidents") or []:
            all_incidents.append({"phase": phase, **{k: v for k, v in i.items() if k != "t"}})
        for model, d in (m.get("llm") or {}).items():
            rows.append(_attach({
                "phase": phase, "kind": "model", "name": model,
                "calls": d.get("calls", 0), "input_tokens": d.get("prompt_tokens", 0),
                "output_tokens": d.get("output_tokens", 0), "thought_tokens": d.get("thought_tokens", 0),
                "seconds": d.get("seconds", 0.0), "cost_usd": d.get("cost_usd", 0.0),
                "note": d.get("note", ""),
            }, counts))
        for model, d in (m.get("embedding") or {}).items():
            rows.append(_attach({
                "phase": phase, "kind": "embedding", "name": model,
                "calls": d.get("requests", 0), "texts": d.get("texts", 0),
                "input_tokens": d.get("est_tokens", 0), "output_tokens": 0, "thought_tokens": 0,
                "seconds": 0.0, "cost_usd": d.get("cost_usd", 0.0),
                "note": "tokens estimated at 4 chars/token — the embeddings API reports none",
            }, counts))
        bqd = m.get("bigquery") or {}
        if bqd.get("queries"):
            rows.append(_attach({
                "phase": phase, "kind": "bigquery", "name": "bigquery",
                "calls": bqd.get("queries", 0), "gib_billed": bqd.get("gib_billed", 0.0),
                "seconds": 0.0, "cost_usd": bqd.get("cost_usd", 0.0),
                "note": "the account's first 1 TiB each month is free and is not deducted here",
            }, counts))
        for channel, n in (m.get("external") or {}).items():
            rows.append(_attach({
                "phase": phase, "kind": "external", "name": channel,
                "calls": n, "seconds": 0.0, "cost_usd": 0.0,
                "note": "no marginal price; consumes a quota — see /quota",
            }, counts))

    t = totals(ph)
    priced_models = {r["name"] for r in rows if r["kind"] == "model"}
    by_phase = sorted(((p, m.get("cost_usd", 0.0)) for p, m in ph.items()), key=lambda x: -x[1])
    most = None
    if by_phase and by_phase[0][1] > 0:
        name, amount = by_phase[0]
        top = max((r for r in rows if r["phase"] == name), key=lambda r: r["cost_usd"], default=None)
        most = {"phase": name, "cost_usd": amount,
                "share_of_total": round(amount / t["cost_usd"], 3) if t["cost_usd"] else 0.0,
                "seconds": (ph[name] or {}).get("seconds", 0.0),
                "driver": (f'{top["name"]} (${top["cost_usd"]:.4f})' if top else "")}

    return {
        "job_id": _job_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {"cost_usd": t["cost_usd"], "seconds": t["seconds"], "llm_calls": t["llm_calls"],
                   "external_calls": t["external_calls"], "failures": t["failures"],
                   "degradations": t["degradations"], "retries": t["retries"]},
        "most_expensive": most,
        # No per-phase copy here: `cost.phases` in the same results.json already
        # carries it, and the ledger rides along on the job record, which is
        # re-serialised to GCS on every node update.
        "rows": sorted(rows, key=lambda r: -r["cost_usd"]),
        "incidents": all_incidents,
        "prices": {
            "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing"
                      " and https://cloud.google.com/bigquery/pricing, read 2026-09-18",
            "usd_per_mtok": {k: {"input": v[0], "output": v[1]} for k, v in PRICES.items()},
            "bigquery_usd_per_tib": BQ_USD_PER_TIB,
            "embedding_usd_per_mtok": EMBED_USD_PER_MTOK,
            "review_by": PRICE_REVIEW_DATE,
            "upcoming_changes": upcoming_price_changes(),
        },
        "caveats": [
            "Estimated from list prices, not a bill: no Vertex committed-use discount,"
            " no context-caching credit and no free tier is modelled.",
            _TIER_NOTE,
            "Prices are the Global-region column; llm.py defaults VERTEX_LOCATION=global."
            " A regional deployment pays ~10% more.",
            "A call whose response carried no usage_metadata contributes its call count"
            " but no tokens, so its cost is missing rather than wrong.",
        ] + [_change_caveat(c) for c in upcoming_price_changes()
             # only the models this run actually spent on: a scheduled change to
             # something nobody called is reference, not a caveat on this bill
             if not priced_models or c["model"] in priced_models],
    }
