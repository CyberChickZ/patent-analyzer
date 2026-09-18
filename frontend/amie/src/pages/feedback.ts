/** Feedback — what changed, and what somebody said about it, side by side.
 *
 *  Two columns because there are two different kinds of fact here. The left is
 *  a record: a prompt was saved, a reviewer edited an element at a gate. The
 *  right is an opinion: somebody read a card and typed something. Putting them
 *  in one list would make "the system noticed this" read as "somebody
 *  complained about this", and the useful question — did a complaint lead to a
 *  change? — would be invisible.
 *
 *  Every row links back to the exact place it came from, because feedback with
 *  no way back to what it was about is a to-do list nobody can action.
 */

import {
  listFeedback, patchFeedback, type FeedbackEntry, type FeedbackPage,
} from "../api";
import { esc, empty, errorBox, on } from "../ui";

let PAGE: FeedbackPage | null = null;
let FILTERS = { job: "", kind: "", status: "", prompt_name: "" };

export function disposeFeedback(): void {
  PAGE = null;
  FILTERS = { job: "", kind: "", status: "", prompt_name: "" };
}

export async function renderFeedback(host: HTMLElement): Promise<void> {
  host.innerHTML = `<div class="page-head"><div class="kicker">Operations</div>
    <h1>Feedback</h1>
    <div class="lede">Prompt and reviewer changes on the left, what people said on the right.
      Every row links back to the place it came from.</div></div>
    <div id="fb-body"><div class="small muted"><span class="spinner"></span> Loading…</div></div>`;
  await paint(host);
}

async function paint(host: HTMLElement): Promise<void> {
  const body = host.querySelector<HTMLElement>("#fb-body");
  if (!body) return;
  try {
    PAGE = await listFeedback({ ...FILTERS, limit: 100 });
  } catch (e: any) {
    body.innerHTML = errorBox(`Could not load the feedback — ${String(e?.message || e)}`);
    return;
  }
  const all = PAGE.entries;
  const changes = all.filter((e) => e.auto);
  const said = all.filter((e) => !e.auto);

  body.innerHTML = filters(all) + (all.length
    ? `<div class="fb-cols">
        ${column("Prompt and reviewer changes", changes, "Nothing has been changed yet.")}
        ${column("Feedback", said, "Nobody has left a comment or a rating yet.")}
       </div>
       <div class="tiny muted" style="margin-top:.6rem">${esc(PAGE.note || "")}</div>`
    : empty("No feedback yet",
        "Comments left on a result card, ratings on a report, saved prompt versions and "
        + "reviewer edits at a gate all show up here."));

  wire(host);
}

function filters(all: FeedbackEntry[]): string {
  const jobs = [...new Set(all.map((e) => e.job_id).filter(Boolean))];
  const prompts = [...new Set(all.map((e) => e.prompt_name).filter(Boolean))];
  const opt = (v: string, cur: string, label?: string) =>
    `<option value="${esc(v)}"${v === cur ? " selected" : ""}>${esc(label ?? (v || "all"))}</option>`;
  return `<div class="row fb-filters" style="gap:.6rem;margin-bottom:.8rem;flex-wrap:wrap">
    <select id="fb-job" >${opt("", FILTERS.job, "all jobs")}${jobs.map((j) => opt(j, FILTERS.job)).join("")}</select>
    <select id="fb-kind" >${["", "prompt_edit", "reviewer_edit", "comment", "rating"]
      .map((k) => opt(k, FILTERS.kind, k || "all kinds")).join("")}</select>
    <select id="fb-status" >${["", "open", "addressed"]
      .map((k) => opt(k, FILTERS.status, k || "any status")).join("")}</select>
    <select id="fb-prompt" >${opt("", FILTERS.prompt_name, "all prompts")}${prompts.map((p) => opt(p, FILTERS.prompt_name)).join("")}</select>
    <span class="small muted">${all.length} of ${PAGE?.total ?? 0}</span>
    <button class="icon-btn" id="fb-export" title="Copy this view as Markdown">Export</button>
  </div>`;
}

function column(title: string, rows: FeedbackEntry[], emptyText: string): string {
  return `<section class="section"><header><h2>${esc(title)}</h2>
      <span class="hint">${rows.length}</span></header>
    ${rows.length ? groupByDay(rows) : `<div class="small muted">${esc(emptyText)}</div>`}
  </section>`;
}

/** A YYYY-MM-DD read as a Date is midnight UTC, and printing that in a
 *  westward timezone shows the day before — which is how entries written this
 *  afternoon were filed under yesterday. Format the parts, not a timestamp. */
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function dayLabel(day: string): string {
  const [y, m, d] = day.split("-").map(Number);
  return m && d ? `${MONTHS[m - 1]} ${d}, ${y}` : day;
}

function groupByDay(rows: FeedbackEntry[]): string {
  const days: Record<string, FeedbackEntry[]> = {};
  for (const r of rows) (days[r.ts.slice(0, 10)] ||= []).push(r);
  return Object.entries(days)
    .map(([day, list]) => `<div class="fb-day"><div class="subhead">${esc(dayLabel(day))}</div>
      ${list.map(row).join("")}</div>`).join("");
}

/** Where this came from, as a link back. A prompt edit goes to the Prompts
 *  page; anything tied to a job goes to the tab and anchor it was left on. */
function backLink(e: FeedbackEntry): string {
  if (e.kind === "prompt_edit" && e.prompt_name) {
    return `<a href="#/prompts/${encodeURIComponent(e.prompt_name)}">${esc(e.prompt_name)}`
      + `${e.prompt_version != null ? ` v${e.prompt_version}` : ""}</a>`;
  }
  if (!e.job_id) return "";
  const tab = e.target?.tab || "verdict";
  const anchor = e.target?.anchor ? `#${encodeURIComponent(e.target.anchor)}` : "";
  return `<a href="#/results/${esc(e.job_id)}/${esc(tab)}${anchor}">${esc(e.job_title || e.job_id)}</a>`;
}

function row(e: FeedbackEntry): string {
  const thumb = e.kind === "rating" ? (Number(e.rating) >= 0 ? "👍 " : "👎 ") : "";
  const summary = (e.text || e.instruction || e.diff_summary || "").split("\n")[0].slice(0, 140);
  return `<details class="box fb-row" data-id="${esc(e.id)}">
    <summary>
      <span class="tiny muted nowrap">${esc(e.ts.slice(11, 16))}</span>
      <span class="pill pill-tag">${esc(e.kind.replace("_", " "))}</span>
      ${e.status === "addressed" ? `<span class="pill pill-flat">addressed</span>` : ""}
      <span class="small">${thumb}${esc(summary) || "<span class='muted'>(no text)</span>"}</span>
      <span class="tiny muted nowrap">${esc(e.by.split("@")[0])}</span>
    </summary>
    <div class="box-body">
      <div class="small">${backLink(e) || `<span class="muted">no job attached</span>`}</div>
      ${e.text ? `<div class="prose" style="white-space:pre-wrap">${esc(e.text)}</div>` : ""}
      ${e.instruction ? `<div class="subhead">Instruction</div><div class="prose">${esc(e.instruction)}</div>` : ""}
      ${e.diff_summary ? `<div class="subhead">Diff</div><pre class="fb-diff">${esc(e.diff_summary)}</pre>` : ""}
      ${e.addressed_by ? `<div class="small muted">addressed by ${esc(JSON.stringify(e.addressed_by))}</div>` : ""}
      <div class="row" style="margin-top:.4rem;gap:.4rem">
        <button class="icon-btn fb-toggle" data-id="${esc(e.id)}" data-status="${esc(e.status)}">
          ${e.status === "open" ? "Mark addressed" : "Reopen"}</button>
        ${e.status === "open" ? `<input class="fb-by" data-id="${esc(e.id)}" style="max-width:20rem"
           placeholder="addressed by prompt X vN (optional)">` : ""}
      </div>
    </div>
  </details>`;
}

function wire(host: HTMLElement): void {
  const sel = (id: string, key: keyof typeof FILTERS) => {
    const el = host.querySelector<HTMLSelectElement>(id);
    el?.addEventListener("change", () => { FILTERS[key] = el.value; void paint(host); });
  };
  sel("#fb-job", "job");
  sel("#fb-kind", "kind");
  sel("#fb-status", "status");
  sel("#fb-prompt", "prompt_name");

  on(host, "button.fb-toggle", async (el, ev) => {
    ev.preventDefault();
    const id = el.dataset.id!;
    const open = el.dataset.status === "open";
    const by = host.querySelector<HTMLInputElement>(`input.fb-by[data-id="${CSS.escape(id)}"]`)?.value.trim();
    const m = by ? /^(?:addressed by\s+)?(\S+)\s+v(\d+)$/i.exec(by) : null;
    try {
      await patchFeedback(id, {
        status: open ? "addressed" : "open",
        addressed_by: open && by ? (m ? { prompt_name: m[1], version: Number(m[2]) } : { commit: by }) : null,
      });
      await paint(host);
    } catch (e: any) {
      alert(`Could not update that entry — ${String(e?.message || e)}`);
    }
  });

  on(host, "#fb-export", async () => {
    const rows = PAGE?.entries || [];
    const md = ["# Feedback", "",
      ...rows.map((e) => `- **${e.ts.slice(0, 16)}** · ${e.kind} · ${e.by.split("@")[0]}`
        + `${e.job_id ? ` · job ${e.job_id}` : ""}${e.prompt_name ? ` · ${e.prompt_name} v${e.prompt_version}` : ""}`
        + ` · ${e.status}\n  ${(e.text || e.instruction || "").replace(/\n/g, "\n  ")}`)];
    try {
      await navigator.clipboard.writeText(md.join("\n"));
      alert(`Copied ${rows.length} entries as Markdown.`);
    } catch {
      alert("Could not reach the clipboard.");
    }
  });
}
