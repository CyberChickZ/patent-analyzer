/** Quota — how much of every external source is left, on one screen.
 *
 *  The page is deliberately blunt about what it does not know. A source with no
 *  counter shows its rate limit and a dash, not a full bar; a source whose
 *  provider would not answer says so in the row instead of showing the local
 *  count as if it were authoritative. The one number that is never guessed is
 *  "is this gone", because that is the one a run dies on. */

import { getQuota, type QuotaRow, type QuotaSnapshot } from "../api";
import { esc, errorBox, empty, num, fmtDate, fmtDay } from "../ui";

let timer: number | null = null;

export function disposeQuota(): void {
  if (timer !== null) { clearInterval(timer); timer = null; }
}

export async function renderQuota(host: HTMLElement): Promise<void> {
  host.innerHTML = `
    <div class="page-head">
      <div class="kicker">Operations</div>
      <h1>Quota</h1>
      <div class="lede">What is left on every external source this deployment spends, before a run finds out the hard way. Counters are shared by all jobs; a source with no allowance shows the rate it is held to instead.</div>
    </div>
    <div id="quota-body"><div class="small muted"><span class="spinner"></span> Reading the counters…</div></div>`;

  await paint();
  // Minute gates and weekly counters move on their own; a stale panel is the
  // one failure mode that defeats the point of having it.
  disposeQuota();
  timer = window.setInterval(() => void paint(), 60_000);
}

async function paint(): Promise<void> {
  const body = document.getElementById("quota-body");
  if (!body) { disposeQuota(); return; }
  let snap: QuotaSnapshot;
  try {
    snap = await getQuota();
  } catch (e: any) {
    body.innerHTML = errorBox(`Could not read the quota — ${e?.message || e}`);
    return;
  }
  if (!snap.sources.length) {
    body.innerHTML = empty("No sources configured", "The backend reported no external sources at all.");
    return;
  }
  body.innerHTML = headline(snap) + table(snap) + reservedTable(snap) + rateCard(snap) + footer(snap);
}

// ─── Reserved keys ───
//
// A SerpAPI key held back for demos and production jobs. The backend flags it
// (`reserved`) and names it; the panel has to make it look different from a key
// that is not there at all, because otherwise "250 searches left" reads as
// headroom an eval can spend, and the demo dies on the day it matters. So the
// reserved keys leave the main table entirely and are counted separately: what
// is above is what a run may spend, what is below is what it may not.

const isReserved = (r: QuotaRow): boolean => r.reserved === true;

/** The backend puts "(reserved)" in the name so a plain-text consumer sees it.
 *  This page has a badge, and saying it twice is what 5b35bc8 removed. */
const plainName = (r: QuotaRow): string => r.name.replace(/\s*\(reserved\)\s*$/i, "");

/** A price the backend did not send is a dash, like every other missing number
 *  on this page. `.toFixed()` on a null would have taken the whole card down. */
function usd(v: number | null | undefined, digits = 2): string {
  return typeof v === "number" && isFinite(v) ? `$${v.toFixed(digits)}` : "—";
}

/** The rate card comes down with the panel rather than living in the frontend:
 *  a price kept in two places is a price that gets quoted for the wrong model. */
function rateCard(s: QuotaSnapshot): string {
  const p = s.prices;
  if (!p || !p.models?.length) return "";
  const changed = new Set((p.upcoming_changes || []).map((c) => c.model));
  return `<section class="section" style="margin-top:1.4rem">
    <header><h2>Rate card</h2><span class="hint">what a call costs, from the list prices the backend costs runs with</span></header>
    ${upcoming(s)}
    <div class="tbl-wrap"><div class="tw"><table class="tbl">
      <thead><tr><th>Model</th><th class="right">Input / 1M</th><th class="right">Output / 1M</th><th>Note</th></tr></thead>
      <tbody>
        ${p.models.map((m) => `<tr>
          <td class="mono">${esc(m.model)}${changed.has(m.model)
            ? ` <span class="pill pill-paused pill-flat" title="this rate has an end date — see above">introductory</span>` : ""}</td>
          <td class="right num">${usd(m.input_usd_per_mtok)}</td>
          <td class="right num">${usd(m.output_usd_per_mtok)}</td>
          <td class="tiny muted">${esc(m.note || "")}</td></tr>`).join("")}
        <tr><td class="mono">text embeddings</td>
          <td class="right num">${usd(p.embedding_usd_per_mtok, 3)}</td>
          <td class="right muted">—</td>
          <td class="tiny muted">output is not charged</td></tr>
        <tr><td class="mono">BigQuery</td>
          <td class="right num" colspan="2">${usd(p.bigquery_usd_per_tib)} / TiB scanned</td>
          <td class="tiny muted">first 1 TiB each month is free</td></tr>
      </tbody>
    </table></div></div>
    <div class="tiny muted" style="margin-top:.4rem">${esc(p.note)} Source: ${esc(p.source)}. Review by ${esc(p.review_by)}.</div>
  </section>`;
}

/** Scheduled changes, which are a different fact from today's price and the one
 *  a card kept in the frontend always misses. Vertex's own note beside the
 *  Gemini 3 table reads "through December 31, 2026. Starting January 1, 2027,
 *  standard pricing of $1.5 / $7.5 per 1M tokens input / output will apply" —
 *  every figure below comes off `prices.upcoming_changes`, not out of that
 *  sentence, so it follows the backend when the schedule moves. */
function upcoming(s: QuotaSnapshot): string {
  const cs = s.prices?.upcoming_changes || [];
  if (!cs.length) return "";
  return `<div class="notice notice-warn" style="margin-bottom:.7rem">
    <b>Scheduled price ${cs.length === 1 ? "change" : "changes"}.</b> Today's rate is not the rate these runs
    will cost next year, and nothing on this page is a price the frontend holds its own copy of.
    <ul style="margin:.35rem 0 0;padding-left:1.1rem">
      ${cs.map((c) => `<li><code>${esc(c.model)}</code> — ${usd(c.from_input)} / ${usd(c.from_output)} per 1M today,
        <b>${usd(c.input_usd_per_mtok)} / ${usd(c.output_usd_per_mtok)} from ${esc(c.effective_from)}</b>${
          c.multiple ? ` (output ×${c.multiple})` : ""}</li>`).join("")}
    </ul></div>`;
}

function headline(s: QuotaSnapshot): string {
  const spendable = s.sources.filter((r) => !isReserved(r));
  // counted among the spendable ones: a reserved key's counter is not headroom,
  // so folding it into "9 metered" beside a count of 12 spendable sources was
  // two different populations in one tile.
  const counted = spendable.filter((r) => r.cap !== null);
  const held = s.sources.filter(isReserved);
  const nextReset = spendable
    .map((r) => r.resets_at)
    .filter(Boolean)
    .sort()[0];
  const warn = s.exhausted.length
    ? `<div class="notice notice-error" style="margin-bottom:.8rem">Spent: ${esc(s.exhausted.join(" · "))}</div>`
    : "";
  // The days are the fact, not the flag: a plan with nine left and one with one
  // left are the same row until the number is on the screen.
  const dated = s.sources.filter((r) => s.expiring_soon.includes(r.name));
  const soon = dated.length
    ? `<div class="notice notice-warn" style="margin-bottom:.8rem"><b>Expiring within two weeks.</b>
        ${dated.map((r) => `${esc(plainName(r))} — <b>${r.expires_in_days}d left</b>, ends ${esc(fmtDay(r.expires_on))}`).join(" · ")}</div>`
    : "";
  const heldLeft = held.reduce((n, r) => n + (r.remaining ?? 0), 0);
  return `${warn}${soon}
    <div class="budget" style="margin-bottom:1.1rem">
      <div><div class="k">Sources</div><div class="v">${spendable.length}<small> / ${counted.length} metered</small></div></div>
      <div><div class="k">Exhausted</div><div class="v" style="${s.exhausted.length ? "color:var(--danger)" : ""}">${s.exhausted.length}</div></div>
      <div><div class="k">Expiring soon</div><div class="v" style="${s.expiring_soon.length ? "color:var(--warn)" : ""}">${
        dated.length ? `${Math.min(...dated.map((r) => r.expires_in_days ?? 0))}<small>d, soonest of ${dated.length}</small>` : "0"}</div></div>
      <div><div class="k">Held back</div><div class="v">${held.length
        ? `${num(heldLeft)}<small> ${esc(held[0].unit)}, not for a run</small>`
        : `—<small> nothing reserved</small>`}</div></div>
      <div><div class="k">Next reset</div><div class="v" style="font-size:.86rem">${nextReset ? esc(fmtDay(nextReset)) : "—"}</div></div>
    </div>`;
}

function meter(r: QuotaRow): string {
  if (r.cap === null || r.used === null) return `<span class="muted tiny">no counter</span>`;
  const frac = Math.max(0, Math.min(1, r.used / r.cap));
  // A reserved key's bar is not headroom, so it does not get headroom's colour.
  const cls = isReserved(r) ? "m-held" : r.exhausted ? "m-danger" : frac > 0.8 ? "m-warn" : "";
  return `<span class="meter ${cls}" title="${num(r.used)} of ${num(r.cap)} ${esc(r.unit)} used"><span style="width:${(frac * 100).toFixed(1)}%"></span></span>`;
}

function left(r: QuotaRow): string {
  if (r.remaining === null) return `<span class="muted">—</span>`;
  const cls = r.exhausted ? "style=\"color:var(--danger);font-weight:600\"" : "";
  return `<span ${cls}>${num(r.remaining)}</span> <span class="tiny muted">${esc(r.unit)}</span>`;
}

/** The reset day, not the word "month".
 *
 *  Saying "month" invites the reader to assume the 1st, and for Lens that is
 *  wrong by a fortnight: its counter runs from the subscription's own
 *  anniversary — the provider's `/subscriptions/*​/usage` returns
 *  `resetDate: 2026-10-18` — and the backend carries that date through rather
 *  than taking the calendar month (N1, quota.py `_lens`). So the date is
 *  printed, and a monthly counter that does not roll over on the 1st says so
 *  instead of leaving the reader to spot it. Both facts come off `resets_at`;
 *  nothing here knows which source it is looking at. */
function resets(r: QuotaRow): string {
  if (!r.resets_at) return `<span class="muted">—</span>`;
  const h = r.resets_in_hours ?? 0;
  const when = h >= 48 ? `${Math.round(h / 24)}d` : `${Math.round(h)}h`;
  const day = new Date(r.resets_at).getUTCDate();
  const anniversary = r.period === "month" && day !== 1;
  return `<span title="${esc(r.resets_at)}">${esc(fmtDay(r.resets_at))} · in ${when}</span>
    <div class="tiny muted">${esc(r.period)}${anniversary
      ? ` · on the ${day}${ordinal(day)}, not the 1st — this plan's month runs from its own anniversary`
      : ""}</div>`;
}

function ordinal(d: number): string {
  if (d % 100 >= 11 && d % 100 <= 13) return "th";
  return ["th", "st", "nd", "rd"][d % 10] || "th";
}

/** A dated plan can be barely used and still gone, which no remaining-count
 *  shows. The days left are the number worth reading, so they are stated in
 *  full rather than tucked into a pill's tooltip. */
function expiry(r: QuotaRow): string {
  if (r.expires_in_days === null) return "";
  const d = r.expires_in_days;
  const cls = d < 0 ? "pill-failed" : d <= 14 ? "pill-paused" : "pill-tag";
  const label = d < 0 ? `expired ${-d}d ago` : `${d}d left`;
  return ` <span class="pill ${cls} pill-flat">${esc(label)}</span>
    <span class="tiny muted">plan ends ${esc(fmtDay(r.expires_on))}</span>`;
}

function row(r: QuotaRow): string {
  return `<tr class="${r.exhausted ? "is-spent" : ""}${isReserved(r) ? " is-reserved" : ""}">
    <td>
      <div><strong>${esc(plainName(r))}</strong>${isReserved(r)
        ? ` <span class="pill pill-tag pill-flat">held back</span>` : ""}${expiry(r)}</div>
      ${r.note ? `<div class="tiny muted">${esc(r.note)}</div>` : ""}
      ${r.error ? `<div class="tiny" style="color:var(--danger)">${esc(r.error)}</div>` : ""}
    </td>
    <td>${meter(r)}<div class="tiny muted nowrap">${r.cap === null ? "" : `${num(r.used)} / ${num(r.cap)}`}</div></td>
    <td class="right nowrap">${left(r)}</td>
    <td class="small">${resets(r)}</td>
    <td class="tiny muted">${r.limits.map((l) => esc(l)).join("<br>") || "—"}</td>
  </tr>`;
}

const HEAD = `<thead><tr>
  <th>Source</th><th style="width:7rem">Used</th><th class="right">Remaining</th>
  <th>Resets</th><th>Limits</th>
</tr></thead>`;

function table(s: QuotaSnapshot): string {
  const rows = s.sources.filter((r) => !isReserved(r));
  return `<div class="tbl-wrap"><div class="tw"><table class="tbl">
    ${HEAD}<tbody>${rows.map(row).join("")}</tbody>
  </table></div></div>`;
}

function reservedTable(s: QuotaSnapshot): string {
  const rows = s.sources.filter(isReserved);
  if (!rows.length) return "";
  const left = rows.reduce((n, r) => n + (r.remaining ?? 0), 0);
  return `<section class="section" style="margin-top:1.4rem">
    <header><h2>Held back</h2><span class="hint">counted, but not available to a run</span></header>
    <div class="notice notice-info" style="margin-bottom:.7rem">
      ${rows.length === 1 ? "This key is" : `These ${rows.length} keys are`} reserved for demos and production
      jobs; evaluation runs never rotate onto ${rows.length === 1 ? "it" : "them"}. Its
      ${num(left)} ${esc(rows[0].unit)} are deliberately not in the figures above — a key being kept back
      must not read the same as a key that is missing, or the panel says "one key left" either way.
    </div>
    <div class="tbl-wrap"><div class="tw"><table class="tbl">
      ${HEAD}<tbody>${rows.map(row).join("")}</tbody>
    </table></div></div>
  </section>`;
}

function footer(s: QuotaSnapshot): string {
  return `<div class="small muted" style="margin-top:.7rem">
    ${esc(s.note)} Month ${esc(s.month)}, ISO week ${esc(s.week)}; read ${esc(fmtDate(s.generated_at))}, refreshed every minute.
  </div>`;
}
