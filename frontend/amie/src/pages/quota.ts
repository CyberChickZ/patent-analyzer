/** Quota — how much of every external source is left, on one screen.
 *
 *  The page is deliberately blunt about what it does not know. A source with no
 *  counter shows its rate limit and a dash, not a full bar; a source whose
 *  provider would not answer says so in the row instead of showing the local
 *  count as if it were authoritative. The one number that is never guessed is
 *  "is this gone", because that is the one a run dies on. */

import { getQuota, type QuotaRow, type QuotaSnapshot } from "../api";
import { esc, errorBox, empty, num, fmtDate } from "../ui";

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
  body.innerHTML = headline(snap) + table(snap) + rateCard(snap) + footer(snap);
}

/** The rate card comes down with the panel rather than living in the frontend:
 *  a price kept in two places is a price that gets quoted for the wrong model. */
function rateCard(s: QuotaSnapshot): string {
  const p = s.prices;
  if (!p || !p.models?.length) return "";
  return `<section class="section" style="margin-top:1.4rem">
    <header><h2>Rate card</h2><span class="hint">what a call costs, from the list prices the backend costs runs with</span></header>
    <div class="tbl-wrap"><div class="tw"><table class="tbl">
      <thead><tr><th>Model</th><th class="right">Input / 1M</th><th class="right">Output / 1M</th><th>Note</th></tr></thead>
      <tbody>
        ${p.models.map((m) => `<tr>
          <td class="mono">${esc(m.model)}</td>
          <td class="right num">$${m.input_usd_per_mtok.toFixed(2)}</td>
          <td class="right num">$${m.output_usd_per_mtok.toFixed(2)}</td>
          <td class="tiny muted">${esc(m.note || "")}</td></tr>`).join("")}
        <tr><td class="mono">text embeddings</td>
          <td class="right num">$${p.embedding_usd_per_mtok.toFixed(3)}</td>
          <td class="right muted">—</td>
          <td class="tiny muted">output is not charged</td></tr>
        <tr><td class="mono">BigQuery</td>
          <td class="right num" colspan="2">$${p.bigquery_usd_per_tib.toFixed(2)} / TiB scanned</td>
          <td class="tiny muted">first 1 TiB each month is free</td></tr>
      </tbody>
    </table></div></div>
    <div class="tiny muted" style="margin-top:.4rem">${esc(p.note)} Source: ${esc(p.source)}. Review by ${esc(p.review_by)}.</div>
  </section>`;
}

function headline(s: QuotaSnapshot): string {
  const counted = s.sources.filter((r) => r.cap !== null);
  const nextReset = counted
    .map((r) => r.resets_at)
    .filter(Boolean)
    .sort()[0];
  const warn = s.exhausted.length
    ? `<div class="notice notice-error" style="margin-bottom:.8rem">Spent: ${esc(s.exhausted.join(" · "))}</div>`
    : "";
  const soon = s.expiring_soon.length
    ? `<div class="notice notice-warn" style="margin-bottom:.8rem">Expiring within two weeks: ${esc(s.expiring_soon.join(" · "))}</div>`
    : "";
  return `${warn}${soon}
    <div class="budget" style="margin-bottom:1.1rem">
      <div><div class="k">Sources</div><div class="v">${s.sources.length}<small> / ${counted.length} metered</small></div></div>
      <div><div class="k">Exhausted</div><div class="v" style="${s.exhausted.length ? "color:var(--danger)" : ""}">${s.exhausted.length}</div></div>
      <div><div class="k">Expiring soon</div><div class="v" style="${s.expiring_soon.length ? "color:var(--warn)" : ""}">${s.expiring_soon.length}</div></div>
      <div><div class="k">Next reset</div><div class="v" style="font-size:.86rem">${nextReset ? esc(fmtDate(nextReset)) : "—"}</div></div>
    </div>`;
}

function meter(r: QuotaRow): string {
  if (r.cap === null || r.used === null) return `<span class="muted tiny">no counter</span>`;
  const frac = Math.max(0, Math.min(1, r.used / r.cap));
  const cls = r.exhausted ? "m-danger" : frac > 0.8 ? "m-warn" : "";
  return `<span class="meter ${cls}" title="${num(r.used)} of ${num(r.cap)} ${esc(r.unit)} used"><span style="width:${(frac * 100).toFixed(1)}%"></span></span>`;
}

function left(r: QuotaRow): string {
  if (r.remaining === null) return `<span class="muted">—</span>`;
  const cls = r.exhausted ? "style=\"color:var(--danger);font-weight:600\"" : "";
  return `<span ${cls}>${num(r.remaining)}</span> <span class="tiny muted">${esc(r.unit)}</span>`;
}

function resets(r: QuotaRow): string {
  if (!r.resets_at) return `<span class="muted">—</span>`;
  const h = r.resets_in_hours ?? 0;
  const when = h >= 48 ? `${Math.round(h / 24)}d` : `${Math.round(h)}h`;
  return `<span title="${esc(r.resets_at)}">${esc(r.period)} · in ${when}</span>`;
}

function expiry(r: QuotaRow): string {
  if (r.expires_in_days === null) return "";
  const d = r.expires_in_days;
  const cls = d < 0 ? "pill-failed" : d <= 14 ? "pill-paused" : "pill-tag";
  const label = d < 0 ? `expired ${-d}d ago` : `${d}d left`;
  return ` <span class="pill ${cls} pill-flat" title="plan ends ${esc(r.expires_on || "")}">${esc(label)}</span>`;
}

function table(s: QuotaSnapshot): string {
  return `<div class="tbl-wrap"><div class="tw"><table class="tbl">
    <thead><tr>
      <th>Source</th><th style="width:7rem">Used</th><th class="right">Remaining</th>
      <th>Resets</th><th>Limits</th>
    </tr></thead>
    <tbody>${s.sources.map((r) => `<tr class="${r.exhausted ? "is-spent" : ""}">
      <td>
        <div><strong>${esc(r.name)}</strong>${expiry(r)}</div>
        ${r.note ? `<div class="tiny muted">${esc(r.note)}</div>` : ""}
        ${r.error ? `<div class="tiny" style="color:var(--danger)">${esc(r.error)}</div>` : ""}
      </td>
      <td>${meter(r)}<div class="tiny muted nowrap">${r.cap === null ? "" : `${num(r.used)} / ${num(r.cap)}`}</div></td>
      <td class="right nowrap">${left(r)}</td>
      <td class="small nowrap">${resets(r)}</td>
      <td class="tiny muted">${r.limits.map((l) => esc(l)).join("<br>") || "—"}</td>
    </tr>`).join("")}</tbody>
  </table></div></div>`;
}

function footer(s: QuotaSnapshot): string {
  return `<div class="small muted" style="margin-top:.7rem">
    ${esc(s.note)} Month ${esc(s.month)}, ISO week ${esc(s.week)}; read ${esc(fmtDate(s.generated_at))}, refreshed every minute.
  </div>`;
}
