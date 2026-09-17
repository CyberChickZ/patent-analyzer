import { getStatus, getEvents, reportUrl, type JobEvent, type JobStatus } from "../api";
import { esc, clip, pill, fmtTime, errorBox, empty, openModal, md, promptBlocks, on, LOCALE } from "../ui";
import { STEPS, stepOfEvent, stepStates, PAUSE_LABEL, PAUSE_STEP, llmCallsByStep, dedupeEvents, isLegacyEvent } from "../phases";
import { mountReview, queriesTable } from "../hitl";
import { rememberJob } from "../main";

let timer: number | null = null;
let events: JobEvent[] = [];
let job: JobStatus | null = null;
let openStep: string | null = null;
let reviewFor = "";        // paused_at the panel is currently mounted for
let lastSig = "";
/** Server-side index asked for next — NOT events.length: the server's list is
 *  longer than ours whenever it re-sent something we dropped. */
let cursor = 0;
let seenKeys = new Set<string>();
let inFlight = false;

export function disposeRun(): void {
  if (timer) { clearInterval(timer); timer = null; }
  events = []; job = null; openStep = null; reviewFor = ""; lastSig = "";
  cursor = 0; seenKeys = new Set<string>(); inFlight = false;
}

/** `embedded` drops the page heading: the same timeline, live events and HITL
 *  review panel are mounted inside the Results page's Event log tab, where the
 *  page already has a heading and a job. One implementation, two places — the
 *  polling and the review panel are the hard parts and must not be forked. */
export function renderRun(host: HTMLElement, jobId: string, opts: { embedded?: boolean } = {}): void {
  disposeRun();
  if (!jobId) {
    host.innerHTML = `<div class="page-head"><div class="kicker">Step 2</div><h1>Run</h1></div>
      ${empty("No job selected", `Pick one from <a href="#/results">Results</a>, or start a new analysis.`)}`;
    return;
  }
  rememberJob(jobId);
  host.innerHTML = `
    ${opts.embedded ? "" : `<div class="page-head">
      <div class="kicker">Step 2</div>
      <h1 id="run-title">Job <span class="mono">${esc(jobId)}</span></h1>
      <div class="lede" id="run-lede">Connecting…</div>
    </div>`}
    <div id="run-error"></div>
    <section class="section"><header><h2>Pipeline</h2><span class="hint" id="run-hint"></span></header>
      <div id="timeline" class="timeline"></div></section>
    <div id="review-slot"></div>
    <section class="section"><header><h2>Live events</h2>
      <span class="hint">polled from <code>/events/{job}</code> every 2s</span></header>
      <div class="panel"><div class="panel-body live" id="live"><div class="small muted"><span class="spinner"></span> waiting…</div></div></div>
    </section>`;

  // A poll that outlives its 2s slot used to have the next one start from the
  // same `since`, so both appended the same tail. One at a time.
  const poll = async () => {
    if (inFlight) return;
    inFlight = true;
    try {
      const [s, e] = await Promise.all([getStatus(jobId), getEvents(jobId, cursor)]);
      job = s;
      cursor = typeof e.total === "number" ? e.total : cursor + (e.events?.length || 0);
      const fresh = dedupeEvents(seenKeys, (e.events || []).filter((x) => !isLegacyEvent(x)));
      if (fresh.length) events = events.concat(fresh);
      paint(jobId);
      if (s.status === "completed" || s.status === "error") {
        if (timer) { clearInterval(timer); timer = null; }
      }
    } catch (err: any) {
      const box = document.getElementById("run-error");
      if (box && !job) box.innerHTML = errorBox(`Cannot reach job ${jobId} — ${err?.message || err}`);
    } finally {
      inFlight = false;
    }
  };
  void poll();
  timer = window.setInterval(poll, 2000);
}

function paint(jobId: string): void {
  if (!job) return;
  const states = stepStates(job, events);
  const calls = llmCallsByStep(events);

  const lede = document.getElementById("run-lede");
  if (lede) {
    const bits = [
      pill(job.status, job.status === "waiting_for_hitl" ? "paused" : undefined),
      esc(job.filename || ""),
      job.queue_position != null ? `queued at position ${job.queue_position + 1} of ${job.queue_depth}` : "",
      job.status === "completed" ? `<a href="#/results/${esc(jobId)}">See the results →</a>` : "",
      job.status === "completed" ? `<a href="${reportUrl(jobId)}" target="_blank" rel="noopener">Full HTML report ↗</a>` : "",
    ].filter(Boolean);
    lede.innerHTML = bits.join(" · ");
  }
  const hint = document.getElementById("run-hint");
  if (hint) hint.textContent = `${events.length} events`;

  const errBox = document.getElementById("run-error")!;
  errBox.innerHTML = job.status === "error" ? errorBox(job.error || "The pipeline failed.") : "";

  // timeline
  const sig = `${job.status}|${job.paused_at}|${events.length}|${openStep}`;
  const tl = document.getElementById("timeline")!;
  if (sig !== lastSig) {
    lastSig = sig;
    tl.innerHTML = STEPS.map((s, i) => {
      const st = states[s.key];
      const evs = events.filter((e) => stepOfEvent(e) === s.key);
      const open = openStep === s.key;
      return `<div class="step is-${st}">
        <div class="step-head" data-step="${s.key}" role="button" tabindex="0">
          <span class="caret">${evs.length ? (open ? "▾" : "▸") : ""}</span>
          <span class="idx">${st === "completed" ? "✓" : i + 1}</span>
          <span class="name">${esc(s.label)}<span class="blurb">${esc(s.blurb)}</span></span>
          <span class="meta">
            ${calls[s.key] ? `<span class="pill pill-tag">${calls[s.key]} LLM</span>` : ""}
            ${evs.length ? `<span class="small muted nowrap">${evs.length} ev</span>` : ""}
            ${pill(st === "pending" ? "pending" : st)}
          </span>
        </div>
        ${open ? `<div class="step-body">${evs.length ? evs.map((e) => evRow(e, events.indexOf(e))).join("") : `<div class="small muted">No events.</div>`}</div>` : ""}
      </div>`;
    }).join("");
    on(tl, ".step-head", (el) => {
      const k = el.dataset.step!;
      openStep = openStep === k ? null : k;
      lastSig = "";
      paint(jobId);
    });
    wireEventClicks(tl);
  }

  // live feed — newest last, scrolled to the bottom
  const live = document.getElementById("live")!;
  if (live.dataset.n !== String(events.length)) {
    live.dataset.n = String(events.length);
    live.innerHTML = events.length
      ? events.slice(-200).map((e, i) => evRow(e, events.length - Math.min(200, events.length) + i, true)).join("")
      : `<div class="small muted">No events yet.</div>`;
    wireEventClicks(live);
    // follow the tail only while it is still producing
    if (job.status === "running" || job.status === "queued") live.scrollTop = live.scrollHeight;
  }

  // review panel
  const slot = document.getElementById("review-slot")!;
  if (job.status === "waiting_for_hitl") {
    const at = job.paused_at || "";
    if (reviewFor !== at) {
      reviewFor = at;
      // the gate labels and the step labels are the same vocabulary now, so
      // "paused after Draft / the gate sits after Draft" said it twice
      const step = STEPS.find((s) => s.key === (PAUSE_STEP[at] || ""));
      slot.innerHTML = `<section class="section"><header><h2>Review — paused after ${esc(PAUSE_LABEL[at] || at)}</h2>
        <span class="hint">${esc(step?.blurb || "")}</span></header>
        <div id="review-host"><div class="small muted"><span class="spinner"></span> loading the paused state…</div></div></section>`;
      void mountReview(document.getElementById("review-host")!, jobId, () => {
        reviewFor = "";
        if (!timer) timer = window.setInterval(() => void 0, 2000);
      });
    }
  } else if (reviewFor) {
    reviewFor = "";
    slot.innerHTML = "";
  }
}

const ICON: Record<string, string> = {
  start: "▶", info: "·", llm: "◆", llm_call: "◆", llm_response: "◇", search: "⌕",
  search_result: "✓", download: "↓", evaluating: "⚙", doc_json: "▤", done: "✓",
  error: "✕", warn: "⚠", round_done: "↻", channel_done: "▸", channel_limited: "❙❙",
  channel_crashed: "✕", prune_done: "⇩", react_step: "→", verified: "✓",
  self_check_pass: "✓", self_check_fail: "✕", draft_written: "✎", draft_plan: "✎",
  draft_flags: "⚠", draft_recheck: "↻", user_edit: "✎", no_invention: "∅",
};

function detailOf(e: JobEvent): boolean {
  const p = e.payload;
  return !!p && !!(p.system || p.user || p.response || p.full || p.traceback || p.query || p.round !== undefined || p.errors?.length || p.queries);
}

function evRow(e: JobEvent, idx: number, withPhase = false): string {
  const kindCls = e.kind === "error" || e.kind === "channel_crashed" || e.kind === "self_check_fail" ? "k-error"
    : e.kind === "warn" || e.kind === "channel_limited" ? "k-warn"
    : e.kind.startsWith("llm") ? "k-llm" : "";
  const p = e.payload || {};
  let pay = "";
  if (p.round !== undefined && p.n_queries !== undefined) {
    const n = (p.covered?.length || 0) + (p.uncovered?.length || 0);
    pay = `round ${p.round} · ${p.n_queries} queries · serpapi ${p.serpapi_calls ?? 0} · pool ${p.pool_size ?? "?"} · covered ${p.covered?.length || 0}/${n}`;
  } else if (p.preview) pay = clip(p.preview, 240);
  else if (p.query) pay = clip(p.query, 240);
  else if (p.channel && p.n !== undefined) pay = `${p.channel} → ${p.n}${p.seconds ? ` in ${Number(p.seconds).toFixed(1)}s` : ""}`;
  const clickable = detailOf(e);
  return `<div class="evt ${kindCls} ${clickable ? "clickable" : ""}" data-ev="${idx}">
    <span class="ic">${ICON[e.kind] || "·"}</span>
    <span class="t">${esc(fmtTime(e.ts))}</span>
    <span class="m">${withPhase ? `<span class="tiny muted mono">${esc(e.phase)}</span> ` : ""}${esc(e.message)}${clickable ? ` <span class="tiny muted">(details)</span>` : ""}</span>
    ${pay ? `<span class="pay">${esc(pay)}</span>` : ""}
  </div>`;
}

/** Rows carry their index into `events`, so a re-render never stales a handler. */
function wireEventClicks(root: ParentNode): void {
  on(root, ".evt.clickable", (el) => {
    const e = events[+el.dataset.ev!];
    if (e) showEvent(e);
  });
}

function showEvent(e: JobEvent): void {
  const p = e.payload || {};
  let h = `<div class="small muted" style="margin-bottom:.6rem">${esc(new Date(e.ts).toLocaleString(LOCALE))} · phase <code>${esc(e.phase)}</code> · kind <code>${esc(e.kind)}</code></div>`;
  if (p.system) h += `<div class="subhead">System prompt <span class="pill pill-tag">hardcoded</span></div><pre>${esc(p.system)}</pre>`;
  if (p.user) h += `<div class="subhead">User prompt</div>${promptBlocks(p.user)}`;
  if (p.response) h += `<div class="subhead">Response <span class="pill pill-tag">LLM output</span></div><div class="prose">${md(p.response)}</div>`;
  if (p.query) h += `<div class="subhead">Query</div><pre>${esc(p.query)}</pre>`;
  if (Array.isArray(p.queries)) h += `<div class="subhead">Queries this round</div>${queriesTable(p.queries.map((q: any) => ({ ...q, round: p.round })))}`;
  if (p.round !== undefined) h += `<div class="subhead">Uncovered after this round</div><pre>${esc((p.uncovered || []).join(", ") || "(none)")}</pre>`;
  if (p.errors?.length) h += `<div class="subhead">Channel errors</div><pre>${esc(JSON.stringify(p.errors, null, 1))}</pre>`;
  if (p.traceback) h += `<div class="subhead">Traceback</div><pre>${esc(p.traceback)}</pre>`;
  if (p.full) h += `<div class="subhead">Full payload</div><pre>${esc(typeof p.full === "string" ? p.full : JSON.stringify(p.full, null, 2))}</pre>`;
  openModal(`${e.kind}: ${clip(e.message, 90)}`, h);
}
