/** Prices and the per-job band, from the backend.
 *
 *  Nothing here is a number. A rate card written into the frontend is wrong the
 *  day a model is repriced or retired and nobody notices, because the page
 *  keeps rendering it with full confidence — and this page is read by people
 *  deciding whether to spend the money. So the figures come from
 *  `GET /api/quota` or they do not appear at all.
 *
 *  That endpoint is N1's and is not live yet. Until it is, `loadQuota()`
 *  resolves to a Quota with `available: false` and the reason, and every caller
 *  renders placeholders plus that reason. When it lands, this file is where it
 *  connects and nothing else has to change.
 *
 *  The shape below is deliberately forgiving — snake_case or camelCase, missing
 *  sections, numbers as strings — because it is being written against an
 *  endpoint that does not exist yet and guessing its exact schema would be the
 *  same mistake as guessing the prices. */

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

export interface Quota {
  available: boolean;
  /** why there are no numbers, when there are none */
  reason?: string;
  rates: Rate[];
  other: OtherRate[];
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
  const rates: Rate[] = (pick(d, "rates", "models", "rate_card", "rateCard") || []).map((r: any) => ({
    model: String(pick(r, "model", "name") ?? ""),
    inPerM: n(pick(r, "inPerM", "in_per_m", "input_per_m", "input", "in")),
    outPerM: n(pick(r, "outPerM", "out_per_m", "output_per_m", "output", "out")),
    note: pick(r, "note", "notes"),
  })).filter((r: Rate) => r.model);
  const other: OtherRate[] = (pick(d, "other", "other_rates", "otherRates", "services") || []).map((r: any) => ({
    item: String(pick(r, "item", "name", "service") ?? ""),
    price: String(pick(r, "price", "cost", "rate") ?? ""),
  })).filter((r: OtherRate) => r.item);
  const jc = pick(d, "jobCost", "job_cost", "per_job", "perJob");
  const low = n(pick(jc || {}, "low", "min", "p10"));
  const high = n(pick(jc || {}, "high", "max", "p90"));
  return {
    available: rates.length > 0 || other.length > 0 || (low !== null && high !== null),
    rates,
    other,
    jobCost: low !== null && high !== null ? { low, high } : null,
  };
}

const EMPTY = (reason: string): Quota => ({ available: false, reason, rates: [], other: [], jobCost: null });

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

/** The one line that goes under a table with nothing in it. */
export function quotaNote(q: Quota): string {
  return q.available
    ? "Live from GET /api/quota."
    : `${q.reason || "No prices available."} Nothing on this page is a hardcoded price.`;
}
