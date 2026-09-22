/** Prices and the per-job band, from the backend.
 *
 *  Nothing here is a number. A rate card written into the frontend is wrong the
 *  day a model is repriced or retired and nobody notices, because the page
 *  keeps rendering it with full confidence — and this page is read by people
 *  deciding whether to spend the money. So the figures come from
 *  `GET /api/quota` or they do not appear at all.
 *
 *  The endpoint is live (fb44725). It answers with `prices.models[]` —
 *  `{model, input_usd_per_mtok, output_usd_per_mtok, note}` — plus
 *  `prices.bigquery_usd_per_tib`, `prices.embedding_usd_per_mtok` and a
 *  `review_by` date that says when the card was last checked against Vertex's
 *  published prices. When it cannot be reached, `loadQuota()` resolves to
 *  `available: false` with the reason, and every caller renders em dashes plus
 *  that reason rather than a number nobody has checked.
 *
 *  The reader stays forgiving about key spelling because backend/quota.py and
 *  this file are edited by different people on different days, and a renamed
 *  key should cost one row, not the panel. */

import { esc } from "./ui";

export interface Rate {
  model: string;
  inPerM: number | null;   // USD per 1M input tokens
  outPerM: number | null;  // USD per 1M output tokens
  note?: string;
}

export interface OtherRate {
  item: string;
  price: string;
}

/** A rate with an end date is not the same fact as a rate, and it is exactly
 *  the fact a card kept in the frontend goes stale on without anyone noticing.
 *  Vertex's note above the Gemini 3 table: "offered with introductory pricing
 *  of $0.75 / $3.75 per 1M tokens input / output through December 31, 2026.
 *  Starting January 1, 2027, standard pricing of $1.5 / $7.5 per 1M tokens
 *  input / output will apply." Every number rendered from this comes off
 *  `prices.upcoming_changes`, never off that sentence. */
export interface PriceChange {
  model: string;
  from: string;             // effective_from, ISO date
  inPerM: number | null;
  outPerM: number | null;
  wasIn: number | null;
  wasOut: number | null;
  multiple: number | null;
}

export interface QuotaSource {
  name: string;
  remaining: number | null;
  limit: number | null;
  unit: string;
  exhausted: boolean;
}

export interface Quota {
  available: boolean;
  /** why there are no numbers, when there are none */
  reason?: string;
  /** when the backend last checked this card against the published prices */
  reviewBy?: string;
  priceNote?: string;
  rates: Rate[];
  other: OtherRate[];
  sources: QuotaSource[];
  /** scheduled rate changes still ahead of today, from the backend */
  upcoming: PriceChange[];
  /** this endpoint does not answer it; see GET /api/jobs/{id}/usage */
  jobCost: { low: number; high: number } | null;
}

export const fmtUSD = (v: number) => `$${v.toFixed(2)}`;

export const EMDASH = "—";

/** A rate cell: the number when there is one, an em dash when there is not. */
export function perM(v: number | null | undefined): string {
  return typeof v === "number" && isFinite(v) ? `$${v.toFixed(2)}` : EMDASH;
}

export function jobCostRange(q: Quota | null): string {
  const c = q?.jobCost;
  return c ? `${fmtUSD(c.low)}${EMDASH}${fmtUSD(c.high)}` : EMDASH;
}

function n(v: unknown): number | null {
  if (v === null || v === undefined || v === "") return null;
  const x = Number(v);
  return isFinite(x) ? x : null;
}

function pick(o: any, ...keys: string[]): any {
  for (const k of keys) if (o && o[k] !== undefined) return o[k];
  return undefined;
}

function parseQuota(d: any): Quota {
  const prices = pick(d, "prices", "rate_card", "rateCard") || {};
  const rates: Rate[] = (pick(prices, "models", "rates") || pick(d, "models", "rates") || []).map((r: any) => ({
    model: String(pick(r, "model", "name") ?? ""),
    inPerM: n(pick(r, "input_usd_per_mtok", "inPerM", "in_per_m", "input_per_m", "input")),
    outPerM: n(pick(r, "output_usd_per_mtok", "outPerM", "out_per_m", "output_per_m", "output")),
    note: pick(r, "note", "notes"),
  })).filter((r: Rate) => r.model);

  // Everything metered per something other than a token, spelled out from the
  // backend's numbers — formatting only, no arithmetic and no second opinion.
  const other: OtherRate[] = [];
  const bq = n(pick(prices, "bigquery_usd_per_tib", "bq_usd_per_tib"));
  if (bq !== null) other.push({ item: "BigQuery", price: `$${bq.toFixed(2)} / TiB scanned` });
  const emb = n(pick(prices, "embedding_usd_per_mtok", "embeddings_usd_per_mtok"));
  if (emb !== null) other.push({ item: "Embeddings", price: `$${emb.toFixed(3)} / 1M tokens` });
  for (const r of pick(d, "other", "other_rates", "services") || []) {
    const item = String(pick(r, "item", "name", "service") ?? "");
    if (item) other.push({ item, price: String(pick(r, "price", "cost", "rate") ?? "") });
  }

  // Not answered by this endpoint: a per-job figure is per job, and lives on
  // GET /api/jobs/{id}/usage. Left null rather than estimated here.
  const jc = pick(d, "jobCost", "job_cost", "per_job") || {};
  const low = n(pick(jc, "low", "min"));
  const high = n(pick(jc, "high", "max"));

  const sources: QuotaSource[] = (pick(d, "sources") || []).map((r: any) => ({
    name: String(pick(r, "name", "source") ?? ""),
    remaining: n(pick(r, "remaining", "left")),
    limit: n(pick(r, "limit", "cap")),
    unit: String(pick(r, "unit") ?? ""),
    exhausted: !!pick(r, "exhausted"),
  })).filter((r: QuotaSource) => r.name);

  const upcoming: PriceChange[] = (pick(prices, "upcoming_changes", "upcomingChanges") || []).map((c: any) => ({
    model: String(pick(c, "model", "name") ?? ""),
    from: String(pick(c, "effective_from", "from", "effectiveFrom") ?? ""),
    inPerM: n(pick(c, "input_usd_per_mtok", "inPerM")),
    outPerM: n(pick(c, "output_usd_per_mtok", "outPerM")),
    wasIn: n(pick(c, "from_input", "wasIn")),
    wasOut: n(pick(c, "from_output", "wasOut")),
    multiple: n(pick(c, "multiple")),
  })).filter((c: PriceChange) => c.model && c.from);

  return {
    available: rates.length > 0 || other.length > 0,
    reviewBy: pick(prices, "review_by", "reviewBy") || undefined,
    priceNote: pick(prices, "note") || undefined,
    rates,
    other,
    sources,
    upcoming,
    jobCost: low !== null && high !== null ? { low, high } : null,
  };
}

const EMPTY = (reason: string): Quota => ({ available: false, reason, rates: [], other: [], sources: [], upcoming: [], jobCost: null });

let cached: Promise<Quota> | null = null;

export function loadQuota(force = false): Promise<Quota> {
  if (!cached || force) {
    cached = (async () => {
      let r: Response;
      try {
        r = await fetch("/api/quota");
      } catch (e: any) {
        return EMPTY(`GET /api/quota could not be reached (${e?.message || e}).`);
      }
      if (r.status === 404 || r.status === 501) {
        return EMPTY("GET /api/quota is not served by this backend yet, so there are no prices to show.");
      }
      if (!r.ok) return EMPTY(`GET /api/quota answered ${r.status}, so there are no prices to show.`);
      try {
        const q = parseQuota(await r.json());
        return q.available ? q : EMPTY("GET /api/quota answered, but with no rates in it.");
      } catch {
        return EMPTY("GET /api/quota did not answer with JSON this page could read.");
      }
    })();
  }
  return cached;
}

/** The scheduled-change block, shared by every page that prints a rate card.
 *  One author, one wording — the same reason the verdict sentence is the
 *  backend's. Empty when the backend reports no change ahead, which is also
 *  what it reports when it could not be reached, and that case already says so
 *  through `quotaNote`. */
export function upcomingBlock(q: Quota | null): string {
  const cs = q?.upcoming || [];
  if (!cs.length) return "";
  return `<div class="notice notice-warn">
    <b>Scheduled price ${cs.length === 1 ? "change" : "changes"} — today's rate is not next year's.</b>
    <ul style="margin:.35rem 0 0;padding-left:1.1rem">
      ${cs.map((c) => `<li><code>${esc(c.model)}</code> — ${perM(c.wasIn)} / ${perM(c.wasOut)} per 1M today,
        <b>${perM(c.inPerM)} / ${perM(c.outPerM)} from ${esc(c.from)}</b>${
          c.multiple ? ` (output ×${c.multiple})` : ""}</li>`).join("")}
    </ul></div>`;
}

/** The one line that goes under a table with nothing in it. */
export function quotaNote(q: Quota): string {
  if (!q.available) return `${q.reason || "No prices available."} Nothing on this page is a hardcoded price.`;
  return [
    "Live from GET /api/quota.",
    q.priceNote || "",
    q.reviewBy ? `Backend rechecks these against the published prices by ${q.reviewBy}.` : "",
  ].filter(Boolean).join(" ");
}
