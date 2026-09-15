import { listJobs, submitJob, deleteJob, isNotSignedIn, type JobSummary } from "../api";
import { esc, pill, fmtDate, errorBox, empty, on } from "../ui";
import { PAUSE_ORDER, PAUSE_LABEL } from "../phases";
import { loadQuota, jobCostRange, perM, quotaNote, upcomingBlock, EMDASH, type Quota } from "../pricing";
import { go, rememberJob } from "../main";

const INPUT_MODES = [
  ["", "Auto-detect"],
  ["academic_paper", "Academic paper (published)"],
  ["manuscript", "Manuscript (draft, no related work)"],
  ["disclosure", "Invention disclosure (Core Idea / Novelty / How It Works)"],
  ["patent_draft", "Patent draft / claims"],
];

let source: "file" | "text" = "file";
let picked: File | null = null;

export function renderSubmit(host: HTMLElement): void {
  host.innerHTML = `
  <div class="page-head">
    <div class="kicker">Step 1</div>
    <h1>Submit a document</h1>
    <div class="lede">A paper, manuscript, disclosure or draft claims. The pipeline decomposes it, searches prior art, evaluates each reference against the elements and drafts claims.</div>
  </div>

  <section class="section">
    <header><h2>Source</h2><span class="hint">PDF, Markdown or plain text</span></header>
    <div class="panel"><div class="panel-body stack">
      <div class="seg" role="tablist">
        <button id="tab-file" class="${source === "file" ? "on" : ""}" role="tab">Upload a file</button>
        <button id="tab-text" class="${source === "text" ? "on" : ""}" role="tab">Paste text</button>
      </div>
      <div id="src-file" class="${source === "file" ? "" : "hidden"}">
        <div class="drop" id="drop">
          <div class="big" id="drop-title">Drop a file here, or click to choose</div>
          <div class="small muted">.pdf · .md · .txt — over 25 MB goes straight to GCS</div>
          <input type="file" id="file-in" accept=".pdf,.txt,.md">
        </div>
      </div>
      <div id="src-text" class="${source === "text" ? "" : "hidden"}">
        <label class="field" for="paste">Document text</label>
        <textarea id="paste" rows="10" placeholder="# Title&#10;&#10;Paste the manuscript, disclosure or claims here."></textarea>
      </div>
    </div></div>
  </section>

  <section class="section">
    <header><h2>Options</h2></header>
    <div class="panel"><div class="panel-body stack">
      <div>
        <label class="field" for="input-mode">Input type — tells the analysis where the invention lives</label>
        <select id="input-mode" style="max-width:26rem">
          ${INPUT_MODES.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("")}
        </select>
      </div>
      <div>
        <div class="field">Pause after — review and edit that phase's output before the next one runs</div>
        <div class="row">
          ${PAUSE_ORDER.map((p) => `<label class="check"><input type="checkbox" class="pause" value="${p}"><span>${esc(PAUSE_LABEL[p])}</span></label>`).join("")}
        </div>
      </div>
      <div>
        <label class="check"><input type="checkbox" id="email-on"><span>Email the report when it finishes</span></label>
        <input type="email" id="email" class="hidden" style="max-width:26rem;margin-top:.35rem" placeholder="recipient@example.com">
      </div>
    </div></div>
  </section>

  <section class="section">
    <header><h2>Budget</h2><span class="hint">prices come from <code>GET /api/quota</code></span></header>
    <div class="stack" id="budget">${budgetBody(null)}</div>
  </section>

  <div class="row" style="margin-bottom:1.6rem">
    <button class="btn btn-lg" id="start" disabled>Analyze</button>
    <span class="small muted" id="start-msg"></span>
  </div>
  <div id="submit-error"></div>

  <section class="section">
    <header><h2>Recent jobs</h2><button class="icon-btn" id="refresh-jobs">Refresh</button></header>
    <div id="jobs"><div class="small muted"><span class="spinner"></span> Loading…</div></div>
  </section>
  `;

  const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
  const drop = $("drop");
  const fileIn = $<HTMLInputElement>("file-in");
  const paste = $<HTMLTextAreaElement>("paste");
  const startBtn = $<HTMLButtonElement>("start");
  const errBox = $("submit-error");

  function setSource(s: "file" | "text") {
    source = s;
    $("tab-file").classList.toggle("on", s === "file");
    $("tab-text").classList.toggle("on", s === "text");
    $("src-file").classList.toggle("hidden", s !== "file");
    $("src-text").classList.toggle("hidden", s !== "text");
    syncStart();
  }
  $("tab-file").addEventListener("click", () => setSource("file"));
  $("tab-text").addEventListener("click", () => setSource("text"));

  function syncStart() {
    startBtn.disabled = source === "file" ? !picked : paste.value.trim().length < 20;
    $("start-msg").textContent = source === "text" && paste.value.trim().length < 20 && paste.value.length
      ? "Paste at least a paragraph." : "";
  }

  function take(f: File) {
    picked = f;
    $("drop-title").textContent = f.name;
    syncStart();
  }

  drop.addEventListener("click", () => fileIn.click());
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = (e as DragEvent).dataTransfer?.files?.[0];
    if (f) take(f);
  });
  fileIn.addEventListener("change", () => { if (fileIn.files?.[0]) take(fileIn.files[0]); });
  paste.addEventListener("input", syncStart);
  if (picked) $("drop-title").textContent = picked.name;
  syncStart();

  const emailOn = $<HTMLInputElement>("email-on");
  const email = $<HTMLInputElement>("email");
  try {
    const saved = localStorage.getItem("amie_email");
    if (saved) { email.value = saved; emailOn.checked = true; email.classList.remove("hidden"); }
  } catch { /* private mode */ }
  emailOn.addEventListener("change", () => email.classList.toggle("hidden", !emailOn.checked));

  const pauses = () => Array.from(document.querySelectorAll<HTMLInputElement>("input.pause:checked")).map((c) => c.value);

  function syncBudget() {
    const n = pauses().length;
    const el = document.getElementById("budget-rerun");
    if (!el) return;   // the block is replaced once the quota answers
    el.innerHTML = n
      ? `${n} gate${n > 1 ? "s" : ""}<small> can be rerun</small>`
      : `—<small> no pauses</small>`;
  }
  document.querySelectorAll<HTMLInputElement>("input.pause").forEach((c) => c.addEventListener("change", syncBudget));
  syncBudget();

  startBtn.addEventListener("click", async () => {
    errBox.innerHTML = "";
    const notify = emailOn.checked ? email.value.trim() : "";
    if (emailOn.checked && !notify) { email.focus(); return; }
    if (notify) { try { localStorage.setItem("amie_email", notify); } catch { /* private mode */ } }
    startBtn.disabled = true;
    startBtn.textContent = "Submitting…";
    try {
      const r = await submitJob({
        file: source === "file" ? picked : null,
        text: paste.value,
        filename: "pasted.md",
        inputMode: ($<HTMLSelectElement>("input-mode")).value,
        pauseAfter: pauses(),
        notifyEmail: notify,
      }, (m) => { startBtn.textContent = m; });
      rememberJob(r.job_id);
      go(`#/run/${r.job_id}`);
    } catch (e: any) {
      errBox.innerHTML = errorBox(e?.message || String(e));
      startBtn.disabled = false;
      startBtn.textContent = "Analyze";
    }
  });

  $("refresh-jobs").addEventListener("click", () => void loadJobs());
  void loadJobs();
  void loadQuota().then((q) => {
    const box = document.getElementById("budget");
    if (box) { box.innerHTML = budgetBody(q); syncBudget(); }
  });
}

/** The budget block, with or without prices. `null` is "still asking". */
function budgetBody(q: Quota | null): string {
  const rates = q?.rates || [];
  const other = q?.other || [];
  // one placeholder row so the table is a table, not a gap
  const rows = rates.length
    ? rates.map((r) => `<tr><td class="mono">${esc(r.model)}</td>
        <td class="right num">${perM(r.inPerM)}</td>
        <td class="right num">${perM(r.outPerM)}</td>
        <td class="small muted">${esc(r.note || "")}</td></tr>`).join("")
      + other.map((r) => `<tr><td class="mono">${esc(r.item)}</td><td colspan="3">${esc(r.price)}</td></tr>`).join("")
    : `<tr><td class="mono muted">${EMDASH}</td><td class="right num muted">${EMDASH}</td>
       <td class="right num muted">${EMDASH}</td><td class="small muted">${q ? "no rate card" : "asking the backend…"}</td></tr>`;
  return `
    <div class="budget">
      <div><div class="k">Estimated per job</div><div class="v" id="budget-job">${jobCostRange(q)}<small> ${q?.jobCost ? "" : "per-job spend is per job: <code>/api/jobs/{id}/usage</code>"}</small></div></div>
      <div><div class="k">Reruns</div><div class="v" id="budget-rerun">${EMDASH}<small> extra</small></div></div>
      <div><div class="k">External calls</div><div class="v">${esc(other[0]?.price || EMDASH)}<small> ${esc(other[0]?.item || "")}</small></div></div>
      <div><div class="k">Rate card</div><div class="v">${q?.reviewBy ? `checked to<small> ${esc(q.reviewBy)}</small>` : EMDASH}</div></div>
    </div>
    ${upcomingBlock(q)}
    <div class="tbl-wrap"><div class="tw"><table class="tbl">
      <thead><tr><th>Model</th><th class="right">Input / 1M</th><th class="right">Output / 1M</th><th>Note</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div></div>
    <div class="small muted">${q ? esc(quotaNote(q)) : "Asking the backend for the rate card…"}
      Pausing is free; each <b>Rerun this phase</b> re-charges that phase.</div>`;
}

async function loadJobs(): Promise<void> {
  const host = document.getElementById("jobs");
  if (!host) return;
  let jobs: JobSummary[];
  try {
    jobs = await listJobs();
  } catch (e: any) {
    host.innerHTML = isNotSignedIn(e)
      ? empty("请先登录", "登录后这里会列出你提交过的分析任务。")
      : errorBox(`Could not load jobs — ${e?.message || e}`);
    return;
  }
  if (!jobs.length) {
    host.innerHTML = empty("No jobs yet", "Submit a document above and it will show up here.");
    return;
  }
  jobs.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  host.innerHTML = `<div class="tbl-wrap"><div class="tw"><table class="tbl">
    <thead><tr><th>Job</th><th>Document</th><th>Status</th><th>Phase</th><th class="nowrap">Created</th><th></th></tr></thead>
    <tbody>${jobs.map((j) => `<tr>
      <td><a class="mono" href="#/run/${esc(j.id)}">${esc(j.id)}</a></td>
      <td>${esc(j.filename || "—")}</td>
      <td>${pill(j.status)}</td>
      <td class="small muted">${esc(j.phase || "—")}</td>
      <td class="small muted nowrap">${esc(fmtDate(j.created_at))}</td>
      <td class="right nowrap">
        ${j.status === "completed" ? `<a class="icon-btn" href="#/results/${esc(j.id)}">Results</a> ` : ""}
        <button class="icon-btn del" data-id="${esc(j.id)}" title="Delete job">✕</button></td>
    </tr>`).join("")}</tbody></table></div></div>`;

  on(host, "a[href^='#/run/'], a[href^='#/results/']", (el) => {
    rememberJob((el as HTMLAnchorElement).hash.split("/").pop()!);
  });
  on(host, "button.del", async (el, ev) => {
    ev.stopPropagation();
    const id = el.dataset.id!;
    if (!confirm(`Delete job ${id}? This removes its files from GCS.`)) return;
    await deleteJob(id);
    void loadJobs();
  });
}
