/** Rate card and the measured per-job band.
 *
 *  The backend meters tokens per model in app/llm.py (`usage`), but that meter is
 *  process-local and is not exposed on any endpoint, so the UI cannot show a
 *  billed figure. What it shows is: this rate card, the measured band from real
 *  runs, and the call counts it can actually count from the job's event stream. */

export interface Rate {
  model: string;
  inPerM: number;   // USD per 1M input tokens
  outPerM: number;  // USD per 1M output tokens
  note?: string;
}

export const RATES: Rate[] = [
  { model: "gemini-3.8-flash", inPerM: 0.75, outPerM: 3.75 },
  { model: "gemini-2.5-pro", inPerM: 1.25, outPerM: 10.0 },
  { model: "gemini-3.1-flash-lite", inPerM: 0.25, outPerM: 1.5, note: "thought tokens bill as output" },
];

export const OTHER_RATES = [
  { item: "BigQuery", price: "$6.25 / TiB scanned" },
  { item: "SerpAPI", price: "$0 (free tier)" },
];

/** Measured end-to-end cost of one job, from real runs. */
export const JOB_COST = { low: 2.97, high: 4.05 };

export const fmtUSD = (v: number) => `$${v.toFixed(2)}`;

export function jobCostRange(): string {
  return `${fmtUSD(JOB_COST.low)}–${fmtUSD(JOB_COST.high)}`;
}

export function costOf(model: string, inTokens: number, outTokens: number): number | null {
  const r = RATES.find((x) => x.model === model);
  if (!r) return null;
  return (inTokens / 1e6) * r.inPerM + (outTokens / 1e6) * r.outPerM;
}
