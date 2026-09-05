"""One screen's worth of "what is left" across every external source.

Every channel already knew its own limits, in its own shape: SerpAPI kept a
per-key monthly counter, USPTO ODP a pair of weekly ones, Lens a rate gate and
a trial that expires on a date, Semantic Scholar nothing but a serial lock.
None of it was readable from outside the process that spent it, so the only way
to find out a key was dead was to watch a run die on it.

Every source here answers the same five questions — how much is left, out of
what, over which period, when does it reset, and is it gone — plus, where it
applies, when the whole plan expires. Anything a source cannot answer stays
null rather than being guessed: a panel that invents a number is worse than one
that admits it does not have it.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from .runtime_state import month_key, week_key

# The one BigQuery figure that behaves like a quota. Same page as the price:
# https://cloud.google.com/bigquery/pricing, read 2026-09-18 — "The first 1 TiB
# per month is free." Account-wide, so this is the whole project's allowance.
BQ_FREE_MIB_PER_MONTH = 1024 * 1024


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _next_month(now: datetime | None = None) -> datetime:
    now = now or _now()
    y, m = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return datetime(y, m, 1, tzinfo=timezone.utc)


def _next_week(now: datetime | None = None) -> datetime:
    """ISO weeks start on Monday, which is when week_key() rolls over."""
    now = now or _now()
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return start + timedelta(days=8 - now.isoweekday())   # Monday=1 -> a week out


def _row(source: str, name: str, *, used=None, cap=None, unit="requests",
         period="none", resets_at: datetime | None = None, limits=None,
         note="", error="", expires_on: str = "") -> dict:
    now = _now()
    remaining = None if (used is None or cap is None) else max(0, cap - used)
    row = {
        "source": source, "name": name,
        "used": used, "cap": cap, "remaining": remaining, "unit": unit,
        "period": period,
        "resets_at": resets_at.isoformat() if resets_at else None,
        "resets_in_hours": round((resets_at - now).total_seconds() / 3600, 1) if resets_at else None,
        "exhausted": bool(remaining == 0 and cap),
        "limits": limits or [],
        "note": note, "error": error,
        "expires_on": expires_on or None, "expires_in_days": None,
    }
    if expires_on:
        try:
            end = datetime.fromisoformat(expires_on).replace(tzinfo=timezone.utc)
            row["expires_in_days"] = (end.date() - now.date()).days
            # An expiry that has passed takes the source away as surely as a
            # spent counter does.
            row["exhausted"] = row["exhausted"] or row["expires_in_days"] < 0
        except ValueError:
            row["error"] = (row["error"] + f"; unparseable expiry {expires_on!r}").strip("; ")
    return row


def _serpapi() -> list[dict]:
    from .recall import serpapi as sp
    keys = sp.quota_status()
    if not keys:
        return [_row("serpapi", "SerpAPI", error="SERPAPI_KEYS not set")]
    return [_row("serpapi", f"SerpAPI key {k['key']}", used=k["used"], cap=k["cap"],
                 unit="searches", period="month", resets_at=_next_month(),
                 limits=[f"{k['cap']}/key/month (free tier)"],
                 note="SerpAPI bills a search that returns no results, so this counter "
                      "moves even on a query that found nothing")
            for k in keys]


def _uspto_odp() -> list[dict]:
    from .recall import uspto_odp as odp
    rows = [_row("uspto_odp", f"USPTO ODP {q['kind']}", used=q["used"], cap=q["cap"],
                 unit="requests", period="week", resets_at=_next_week(),
                 limits=[f"{odp.PER_MINUTE} req/min", "concurrency 1"],
                 note=f"ISO week {q['week']}; the weekly caps come from the key's registration")
            for q in odp.quota_status()]
    if not os.environ.get("USPTO_ODP_API_KEY"):
        for r in rows:
            r["error"] = "USPTO_ODP_API_KEY not set"
    return rows


def _pick(d: dict, *names):
    for n in names:
        if isinstance(d, dict) and d.get(n) is not None:
            return d[n]
    return None


def _walk_usage(payload) -> list[dict]:
    """Pull every limit/used triple out of a response whose shape is not
    documented. Returns [] rather than guessing when nothing matches."""
    out = []
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            cap = _pick(node, "limit", "max", "quota", "allowed")
            used = _pick(node, "used", "usage", "consumed", "count")
            left = _pick(node, "remaining", "left")
            if isinstance(cap, (int, float)) and (isinstance(used, (int, float)) or isinstance(left, (int, float))):
                if used is None:
                    used = cap - left
                out.append({"cap": int(cap), "used": int(used),
                            "resource": str(_pick(node, "resource", "type", "name") or "requests"),
                            "period": str(_pick(node, "period", "interval", "per") or "month").lower()})
            stack.extend(node.values())
    return out


async def _lens() -> list[dict]:
    from .recall import lens
    rows = []
    for q in lens.quota_status():
        ep = q["endpoint"]
        used, cap, note, err = q["used"], q["cap"], "", ""
        live, live_err = await lens.live_usage(ep)
        if live_err:
            err = live_err
            note = "counted locally — the provider's own usage endpoint did not answer"
        else:
            found = [u for u in _walk_usage(live) if u["period"].startswith("month")
                     and u["resource"].lower().startswith("request")]
            if found:
                used, cap = found[0]["used"], found[0]["cap"]
                note = "from the provider's /subscriptions/*/usage"
            else:
                note = ("the usage endpoint answered in a shape this code does not recognise; "
                        "the figure is this instance's own count")
        rows.append(_row("lens", f"Lens {ep} API", used=used, cap=cap,
                         unit="requests", period="month", resets_at=_next_month(),
                         limits=[f"{q['per_minute']} req/min",
                                 f"max {q['max_records_per_request']} records/request"],
                         note=note, error=err, expires_on=lens.TRIAL_ENDS))
    return rows


def _semantic_scholar() -> list[dict]:
    from .recall import semantic_scholar as s2
    keyed = bool(os.environ.get("SEMANTIC_SCHOLAR_KEY") or os.environ.get("S2_API_KEY"))
    return [_row("semantic_scholar", "Semantic Scholar",
                 unit="requests", period="none",
                 limits=[f"1 request per {s2.MIN_INTERVAL_KEYED if keyed else s2.MIN_INTERVAL_ANON:.1f}s, "
                         "across all endpoints"],
                 note="a rate, not an allowance: there is no monthly counter to run out of. "
                      + ("API key in use." if keyed else "No key — the anonymous limit applies."))]


def _bigquery() -> list[dict]:
    from .cache import kv
    used = int((kv().get("runtime", f"usage:bigquery_mib:{month_key()}") or {}).get("n", 0))
    from . import metering
    return [_row("bigquery", "BigQuery free tier", used=used, cap=BQ_FREE_MIB_PER_MONTH,
                 unit="MiB scanned", period="month", resets_at=_next_month(),
                 limits=[f"${metering.BQ_USD_PER_TIB}/TiB above the free tier"],
                 note="not a hard limit — past this the project is billed rather than blocked, "
                      "which is why it is the one BigQuery number worth watching")]


_FREE = [("openalex", "OpenAlex", "polite pool; no key, no counter"),
         ("arxiv", "arXiv", "1 request / 3s by convention; no counter"),
         ("google_patents", "Google Patents (direct)", "unmetered, but blocks on volume — "
          "a trip opens the shared circuit breaker and SerpAPI takes over")]


def _uncapped() -> list[dict]:
    return [_row(s, n, unit="requests", period="none", note=note) for s, n, note in _FREE]


async def snapshot() -> dict:
    """Every source, with each one's failure kept local: a panel that 500s
    because one provider is down tells you nothing about the other seven."""
    sources: list[dict] = []
    for label, fn in (("serpapi", _serpapi), ("uspto_odp", _uspto_odp),
                      ("semantic_scholar", _semantic_scholar), ("bigquery", _bigquery),
                      ("free", _uncapped)):
        try:
            sources.extend(fn())
        except Exception as exc:
            sources.append(_row(label, label, error=f"{type(exc).__name__}: {exc}"))
    try:
        sources.extend(await _lens())
    except Exception as exc:
        sources.append(_row("lens", "Lens", error=f"{type(exc).__name__}: {exc}"))

    exhausted = [s["name"] for s in sources if s["exhausted"]]
    expiring = [s["name"] for s in sources
                if s["expires_in_days"] is not None and 0 <= s["expires_in_days"] <= 14]
    return {
        "generated_at": _now().isoformat(),
        "month": month_key(), "week": week_key(),
        "sources": sources,
        "exhausted": exhausted,
        "expiring_soon": expiring,
        "note": "Counters are this deployment's own unless a row says otherwise; a source "
                "whose quota is shared with another tool will read low here.",
    }
