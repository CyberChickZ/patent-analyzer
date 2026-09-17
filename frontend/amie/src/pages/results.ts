import {
  getResults, getStatus, reportUrl, getFulltextGaps, uploadFulltext, dropFulltextUpload,
  rerunEvidence, getLedger, listJobs, deleteJob, isNotSignedIn,
  type JobEvent, type FulltextGaps, type FulltextGapRow, type JobSummary, type Ledger,
} from "../api";
import { esc, clip, pill, num, empty, errorBox, openModal, md, on, plainTitle, fmtDate } from "../ui";
import { queriesTable } from "../hitl";
import { dedupeEvents, isLegacyEvent } from "../phases";
import { loadQuota, perM, quotaNote, upcomingBlock, fmtUSD, EMDASH, type Quota } from "../pricing";
import { rememberJob } from "../main";
import { renderRun, disposeRun } from "./run";

let R: any = null;            // results.json
let LOADED = "";              // which job R holds, so switching tab does not refetch it
let EVENTS: JobEvent[] = [];
let showAllDocs = false;
/** Which collapsible blocks the reader has opened, so a repaint (Show every
 *  reference) does not close them again. */
let opened = new Set<string>();
let QUOTA: Quota | null = null;
let GAPS: FulltextGaps | null = null;
/** What this job actually spent, from GET /api/jobs/{id}/usage — the per-phase
 *  meter diffed while it ran, not a range worked out from a rate card. Null
 *  when the endpoint has nothing for this job, and the page says so rather
 *  than falling back to an estimate. */
let LEDGER: Ledger | null = null;
let LEDGER_WHY = "";
let showAllGaps = false;
let spy: IntersectionObserver | null = null;

/** Coverage as the adjudicator counts it — score >= 1 with a verified quote
 *  (patent_analyzer/adjudicate.element_covered). Reading it back off
 *  adjudication.per_doc_coverage keeps this page's matrix and its candidate
 *  badges consistent with the §102/§103 verdict printed above them; falling
 *  back to score >= 2 would say 1/5 where the verdict says 4/5. */
let COVERED: Map<string, Set<string>> = new Map();

function buildCoverage(): void {
  COVERED = new Map();
  for (const d of (R.adjudication || {}).per_doc_coverage || []) {
    COVERED.set(String(d.pub_num || d.title || ""), new Set<string>(d.covered || []));
  }
}

function isCovered(doc: any, criterion: string): boolean {
  const key = String(doc.pub_num || doc.title || "");
  const set = COVERED.get(key);
  if (set) return set.has(criterion);
  return (doc.checklist_results?.[criterion]?.score ?? 0) >= 2;   // no adjudication on this job
}

/** The router disposes on every navigation, and switching tab IS a navigation
 *  (the tab is in the hash). So dispose only drops what is per-paint; the
 *  results.json itself — tens of MB on a full run — is kept and thrown away in
 *  `resetResults` when the job id actually changes. */
export function disposeResults(): void {
  spy?.disconnect(); spy = null;
  disposeRun();          // the Event log tab mounts the run page; its poll has to stop
}

function resetResults(): void {
  R = null; LOADED = ""; EVENTS = []; showAllDocs = false; QUOTA = null; GAPS = null; showAllGaps = false;
  LEDGER = null; LEDGER_WHY = "";
  opened = new Set<string>();
  spy?.disconnect(); spy = null;
}

/** A full run's results page runs to ~15,000px, which is not a page anyone
 *  reads — it is a page people scroll past. Two things fix that: somewhere to
 *  jump from, and blocks that are closed until asked for. `key` identifies a
 *  block across repaints; `dflt` is what it does on first sight. */
function isOpen(key: string, dflt: boolean): boolean {
  return opened.has(key) ? true : dflt && !opened.has(`!${key}`);
}

function markOpen(key: string, open: boolean): void {
  opened.delete(key); opened.delete(`!${key}`);
  opened.add(open ? key : `!${key}`);
}

interface Sec { key: string; id: string; label: string; }

export async function renderResults(host: HTMLElement, arg: string): Promise<void> {
  const [jobId = "", tab = ""] = String(arg || "").split("/");
  disposeResults();
  if (jobId !== LOADED) resetResults();
  // Same job, already loaded: this is a tab switch. Repaint, do not refetch.
  if (jobId && R && jobId === LOADED) { paint(host, jobId, tab); return; }
  if (!jobId) { await renderJobList(host); return; }
  rememberJob(jobId);
  host.innerHTML = `<div class="page-head"><div class="kicker">Step 3</div>
    <h1>Results <span class="mono small muted">${esc(jobId)}</span></h1>
    <div class="lede"><span class="spinner"></span> Loading results.json — a full run's is tens of MB, so this can take a moment.</div></div>`;

  try {
    const [res, st, q, gaps, ledger] = await Promise.all([
      getResults(jobId),
      getStatus(jobId).catch(() => null),
      loadQuota(),
      // A job from before this endpoint existed, or a backend that is older
      // than the page, just means the section is not drawn.
      getFulltextGaps(jobId).catch(() => null),
      getLedger(jobId).catch((e: any) => { LEDGER_WHY = String(e?.message || e); return null; }),
    ]);
    QUOTA = q;
    GAPS = gaps;
    LEDGER = ledger?.ledger || null;
    R = res;
    // the record carries the backend's duplicates too; the call counts below
    // would otherwise be inflated by them
    EVENTS = dedupeEvents(new Set<string>(), ((st as any)?.events || []).filter((e: JobEvent) => !isLegacyEvent(e)));
  } catch (e: any) {
    const msg = String(e?.message || e);
    host.innerHTML = `<div class="page-head"><div class="kicker">Step 3</div><h1>Results <span class="mono small muted">${esc(jobId)}</span></h1></div>`
      + (!msg.startsWith("404")
        ? errorBox(`Could not load results — ${msg}`)
        : /job not found/i.test(msg)
          ? empty("No such job", `Nothing on the server has the id <code>${esc(jobId)}</code>. <a href="#/submit">Pick one from the job list →</a>`)
          : empty("No results yet", `This job has not produced a results.json. <a href="#/run/${esc(jobId)}">Watch it run →</a>`));
    return;
  }
  LOADED = jobId;
  paint(host, jobId, tab);
}

/** Every job, newest first, with the running ones floated to the top.
 *
 *  This is the page's front door. It used to be a dead end that said "No job
 *  selected", while the topbar's "In progress" quietly pointed at whatever job
 *  was last opened — which was usually one that had finished (Harry,
 *  2026-09-20). A list of jobs cannot go stale in that way.
 */
async function renderJobList(host: HTMLElement): Promise<void> {
  host.innerHTML = `<div class="page-head"><div class="kicker">Step 3</div><h1>Jobs</h1>
    <div class="lede"><span class="spinner"></span> Loading…</div></div>`;
  let jobs: JobSummary[];
  try {
    jobs = await listJobs();
  } catch (e: any) {
    host.innerHTML = `<div class="page-head"><div class="kicker">Step 3</div><h1>Jobs</h1></div>`
      + (isNotSignedIn(e)
        ? empty("Sign in to see your jobs", "Once you are signed in, every job you submitted shows up here.")
        : errorBox(`Could not load the job list — ${String(e?.message || e)}`));
    return;
  }
  const rank = (j: JobSummary) => (j.status === "running" || j.status === "queued" || j.status === "waiting_for_hitl" ? 0 : 1);
  jobs.sort((a, b) => rank(a) - rank(b) || (b.created_at || "").localeCompare(a.created_at || ""));

  host.innerHTML = `<div class="page-head"><div class="kicker">Step 3</div>
    <h1>Jobs <span class="small muted">${jobs.length}</span></h1>
    <div class="lede">Pick one to see its verdict, its evidence and its event log. <a href="#/submit">Start a new analysis →</a></div>
  </div>` + (jobs.length
    ? `<section class="section"><div class="tbl-wrap"><div class="tw"><table class="tbl">
        <thead><tr><th>Job</th><th>Status</th><th>Phase</th><th class="nowrap">Created</th><th></th></tr></thead>
        <tbody>${jobs.map((j) => `<tr>
          <td><a href="#/results/${esc(j.id)}">${esc(j.title || j.filename || j.id)}</a>
            <div class="mono tiny muted">${esc(j.id)}${j.title && j.filename ? ` · ${esc(clip(j.filename, 48))}` : ""}</div></td>
          <td>${pill(j.status, j.status === "waiting_for_hitl" ? "paused" : undefined)}</td>
          <td class="small muted">${esc(j.phase || "—")}${rank(j) === 0 ? ` <span class="spinner"></span>` : ""}</td>
          <td class="small muted nowrap">${esc(fmtDate(j.created_at))}</td>
          <td class="right nowrap"><button class="icon-btn del" data-id="${esc(j.id)}" title="Delete job">✕</button></td>
        </tr>`).join("")}</tbody></table></div></div></section>`
    : empty("No jobs yet", `Submit a document on <a href="#/submit">Submit</a> and it will show up here.`));

  on(host, "a[href^='#/results/']", (el) => rememberJob((el as HTMLAnchorElement).hash.split("/")[2] || ""));
  on(host, "button.del", async (el, ev) => {
    ev.stopPropagation();
    const id = el.dataset.id!;
    if (!confirm(`Delete job ${id}? This removes its files from GCS.`)) return;
    await deleteJob(id);
    await renderJobList(host);
  });
}

function paint(host: HTMLElement, jobId: string, tab = ""): void {
  buildCoverage();
  const p1 = R.phase1 || {};
  const cands: any[] = (R.extraction || {}).candidate_inventions || [];
  const checklist: any[] = (R.phase2 || {}).checklist || [];
  const sr: any[] = (R.evaluation || {}).scoring_report || [];
  const adj = R.adjudication || {};

  const secs = sections(R.user_edits || []);
  const active = secs.find((x) => x.key === tab) || secs[0];
  const build: Record<string, () => string> = {
    verdict: () => determinationSection(adj, sr),
    candidates: () => candidatesSection(cands, checklist, sr),
    search: () => searchSection(),
    fulltext: () => fulltextSection(),
    evidence: () => matrixSection(cands, checklist, sr),
    draft: () => draftSection(R.draft_claims || {}),
    edits: () => editsSection(R.user_edits || []),
    cost: () => costSection(sr),
    // filled after the page is in the DOM: the run page mounts itself
    events: () => `<section class="section" id="sec-events"><div id="run-mount"></div></section>`,
  };
  // One section at a time. A full run rendered all eight at once — ~15,000px
  // of page that nobody reads, where the nav was only a scroll-to (Harry,
  // 2026-09-20). The tab lives in the hash, so a refresh or a shared link
  // lands on the same one.
  const body = (build[active.key] || build.verdict)();

  host.innerHTML = `
  <div class="page-head">
    <div class="kicker"><a href="#/results">← All jobs</a> · ${esc(R.job_id || jobId)}</div>
    <h1>${esc(clip(plainTitle(R.source_title) || R.source_filename || "Results", 140))}</h1>
    <div class="lede">
      ${pill("completed")} ${esc(p1.status_determination || "—")} ·
      ${esc(p1.input_mode || "—")} · CPC ${esc(p1.cpc_subclass || "—")} ·
      <a href="${reportUrl(jobId)}" target="_blank" rel="noopener">Full HTML report ↗</a> ·
      <a href="/api/results/${esc(jobId)}" target="_blank" rel="noopener">results.json ↗</a>
    </div>
  </div>

  ${secNav(R.user_edits || [], jobId, active)}
  ${body}
  `;

  host.querySelectorAll<HTMLDetailsElement>("details[data-keep]").forEach((d) => {
    d.addEventListener("toggle", () => markOpen(d.dataset.keep!, d.open));
  });

  // The Event log tab is the run page, mounted here: same timeline, same live
  // poll, same HITL review panel. It polls only while this tab is open, and
  // disposeResults stops it.
  const mount = host.querySelector<HTMLElement>("#run-mount");
  if (mount) renderRun(mount, jobId, { embedded: true });
  else disposeRun();

  on(host, "[data-cell]", (el) => {
    const [pub, crit] = el.dataset.cell!.split("||");
    const doc = sr.find((d: any) => (d.pub_num || d.title) === pub);
    const cr = doc?.checklist_results?.[crit];
    if (!cr) return;
    const quotes: string[] = cr.evidence_quotes || [];
    openModal(`${doc.pub_num || doc.title} — score ${cr.score ?? 0}`, `
      <div class="small muted" style="margin-bottom:.5rem">${esc(doc.title || "")}</div>
      <div class="row">${isCovered(doc, crit)
        ? `<span class="pill pill-failed pill-flat">counted as covered by the adjudicator</span>`
        : `<span class="pill pill-tag">scored, but not counted as covered</span>`}
        <span class="small muted">score ${cr.score ?? 0}</span></div>
      <div class="subhead">Element</div><div class="prose">${esc(crit)}</div>
      <div class="subhead">Analysis</div><div class="prose">${md(cr.analysis || "—")}</div>
      <div class="subhead">Verbatim quotes (${quotes.length})</div>
      ${quotes.length ? quotes.map((q) => `<div class="quote">${esc(q)}</div>`).join("") : `<div class="small muted">None.</div>`}`);
  });
  on(host, "#toggle-docs", () => { showAllDocs = !showAllDocs; paint(host, jobId); });
  wireFulltext(host, jobId);
}

// ─── Section nav ───
//
// Sticky under the topbar, so the page has a spine. The entries match the
// section ids below one-for-one; Reviewer edits only exists when a reviewer
// edited something, so it is conditional here too.

function sections(edits: any[]): Sec[] {
  return [
    { key: "verdict", id: "sec-verdict", label: "Verdict" },
    { key: "candidates", id: "sec-candidates", label: "Candidates" },
    { key: "search", id: "sec-search", label: "Search" },
    ...(GAPS && GAPS.rows.length ? [{ key: "fulltext", id: "sec-fulltext", label: "Full text" }] : []),
    { key: "evidence", id: "sec-evidence", label: "Evidence" },
    { key: "draft", id: "sec-draft", label: "Draft" },
    ...(edits.length ? [{ key: "edits", id: "sec-edits", label: "Reviewer edits" }] : []),
    { key: "cost", id: "sec-cost", label: "Cost" },
    { key: "events", id: "sec-events", label: "Event log" },
  ];
}

function secNav(edits: any[], jobId: string, active: Sec): string {
  return `<nav class="secnav" role="tablist" aria-label="Sections of this report">
    ${sections(edits).map((x) => `<a href="#/results/${encodeURIComponent(jobId)}/${x.key}" data-sec="${x.key}"
       role="tab" aria-selected="${x.key === active.key}" class="${x.key === active.key ? "on" : ""}">${esc(x.label)}</a>`).join("")}
  </nav>`;
}



// ─── Determination ───

/** The headline sentence is `adj.label_text`, which is
 *  `patent_analyzer.report_sections.determination_label(adj)` — the backend's
 *  own string, printed as given.
 *
 *  It is not composed here from `label` and `basis`, and it must not be. That
 *  wording changed in 00e30b0 precisely because the old one read a benchmark's
 *  scoring convention (PANORAMA App. C.5.3 (a), primary reference >= 70%) as if
 *  it were the statutory standard, and cited MPEP 2143 for a percentage that
 *  appears nowhere in it. A second copy of that sentence living in the
 *  frontend would have had to be found and fixed separately, and the next time
 *  it changes it would drift again. One author, one string.
 *
 *  `label_text` now travels on the adjudication (c7955ce puts
 *  `determination_label(adj)` on the dict in graph/eval_subgraph.py), so it is
 *  printed verbatim. A §103 whose motivation comes from background knowledge
 *  rather than from a reference carries its own clause inside that sentence —
 *  "rationale asserted from ordinary skill, not evidenced in the references",
 *  obviousness.UNEVIDENCED_NOTE — and it stays where the backend put it. What
 *  this page adds is the rule trace underneath, so the reader can see *which*
 *  requirement was met without a quote instead of only being told that one
 *  was. On a job old enough to have no `label_text` the reader still gets
 *  `adj.reason`, the same author's full sentence. */
/** The verdict in a sentence, worked out from `label` here rather than read
 *  from `adjudication.label_text`.
 *
 *  `label_text` is written into results.json when the job runs
 *  (graph/eval_subgraph.py), so a wording fix in the backend would never reach
 *  a job that had already finished — this page would keep showing "§102" to
 *  the end of time. Computing it means every past job gets the new sentence.
 *  `label_text` stays as the fallback for a label this page does not know.
 *
 *  Kept word-for-word in step with patent_analyzer/report_sections._LABEL_TEXT
 *  so the page and the HTML report cannot say different things about the same
 *  verdict. */
function verdictHeadline(adj: any): string {
  const thr = Math.round(((adj.params || {}).single_partial_103 ?? 0.7) * 100);
  if (adj.label === "102") return "One document already shows everything (§102, anticipation)";
  if (adj.label === "103") {
    return adj.basis === "primary_partial"
      ? `Best single document shows most of it (≥${thr}% — screening flag)`
      : "Two or three documents together show everything (§103 screen — a flag for review, not a legal obviousness finding)";
  }
  if (adj.label === "ALLOW") return "No document or combination we read shows all the elements";
  return adj.label_text || "";
}

/** The two numbers the verdict rests on, in the report's words. */
function coverageLines(adj: any): string[] {
  const n = adj.n_elements || 0;
  if (!n) return [];
  const per: any[] = adj.per_doc_coverage || [];
  const bestN = per.length ? (per[0].n_covered ?? 0) : 0;
  const comboN = (adj.combo || {}).n_covered ?? 0;
  const k = (adj.params || {}).max_combo ?? 3;
  return [`Best single document: ${bestN} of ${n} elements.`,
          `Best combination (up to ${k} documents): ${comboN} of ${n}.`];
}

function determinationSection(adj: any, sr: any[]): string {
  if (!adj || !adj.label) {
    return `<section class="section" id="sec-verdict"><header><h2>Determination</h2></header>
      ${empty("No determination", "The rule adjudicator produced no verdict for this job.")}</section>`;
  }
  const chart = adj.claim_chart || {};
  const docs: any[] = chart.docs || [];
  const vclass = adj.label === "102" || adj.label === "103" ? `v-${adj.label}` : "v-allow";
  return `<section class="section" id="sec-verdict">
    <header><h2>Determination and its basis</h2><span class="hint">rule verdict from the charted evidence</span></header>
    <div class="panel"><div class="panel-body stack">
      <div class="verdict ${vclass}">
        <span class="label">§${esc(adj.label)}</span>
        <span class="pill pill-tag">${esc(adj.basis || "")}</span>
        <span class="pill ${adj.risk === "blocking" ? "pill-failed" : "pill-paused"}">${esc(adj.risk || "")}</span>
        <span class="small muted">${esc(adj.n_elements || 0)} elements · best single reference covers ${adj.best_coverage != null ? Math.round(adj.best_coverage * 100) + "%" : "—"}</span>
      </div>
      <div class="verdict-headline">${esc(verdictHeadline(adj))}</div>
      ${coverageLines(adj).map((l) => `<div class="small">${esc(l)}</div>`).join("")}
      <details class="box" data-keep="how"${isOpen("how", false) ? " open" : ""}>
        <summary>How this was decided (for attorneys)</summary>
        <div class="box-body prose">${esc(adj.reason || "")}</div></details>
      ${unevidencedBlock(adj)}
      ${adj.obviousness_explanation ? `<details class="box" open>
        <summary>Obviousness explanation <span class="pill pill-tag">model narrative, not the rule's output</span></summary>
        <div class="box-body prose">${md(adj.obviousness_explanation)}</div></details>` : ""}
      ${docs.length ? `<div class="tbl-wrap"><div class="tw"><table class="tbl">
        <thead><tr><th>Charted reference</th><th style="min-width:14rem">Title</th><th class="right">Elements covered</th><th class="right">Coverage</th></tr></thead>
        <tbody>${docs.map((d: any) => `<tr><td class="mono tiny">${esc(d.pub_num || "—")}${
          d.abstract_only ? ` <span class="pill pill-paused pill-flat">abstract only</span>` : ""}</td>
          <td>${esc(clip(d.title, 130))}</td><td class="right num">${d.n_covered ?? "—"}/${chart.n_elements ?? "—"}</td>
          <td class="right num">${d.coverage != null ? Math.round(d.coverage * 100) + "%" : "—"}</td></tr>`).join("")}</tbody>
      </table></div></div>${abstractOnlyCaveat(docs)}` : ""}
      ${(chart.uncovered || []).length ? `<div class="small"><b>Uncovered by every charted reference:</b> <span class="muted">${esc((chart.uncovered || []).join(" · "))}</span></div>` : ""}
      <div class="small muted">${sr.length} document${sr.length === 1 ? "" : "s"} evaluated in full.</div>
    </div></div>
  </section>`;
}

/** The clause the backend put inside `label_text`, opened out.
 *
 *  A §103 can rest on a motivation that comes from what a person of ordinary
 *  skill knows rather than from anything a reference says, and the rule counts
 *  that as met — MPEP 2143 does not require a quote for background knowledge.
 *  The sentence above already says so, in the backend's words. This says which
 *  requirement it was, because "asserted, not evidenced" is not actionable
 *  until you can see what was asserted. Nothing here re-words the verdict.
 *  `evidenced` defaults to true, so an old trace that predates the flag reads
 *  as evidenced rather than as a silent warning. */
function unevidencedBlock(adj: any): string {
  const bad: any[] = (adj.rule_trace || []).filter(
    (t: any) => t.status === "met" && t.evidenced === false);
  if (!bad.length) return "";
  return `<details class="box" data-keep="unevidenced"${isOpen("unevidenced", false) ? " open" : ""}>
    <summary>${bad.length} requirement${bad.length === 1 ? "" : "s"} met without a citation
      <span class="pill pill-paused pill-flat">asserted from ordinary skill</span></summary>
    <div class="box-body">
      <div class="small muted">Met on background knowledge, not on anything quoted from the references.
        That is allowed — MPEP 2143 does not require a quote for what a person of ordinary skill
        already knows — but it is the part of this determination with nothing behind it to check.</div>
      <div class="tw" style="margin-top:.4rem"><table class="tbl">
        <thead><tr><th>MPEP</th><th>Requirement</th><th>What the rule found</th></tr></thead>
        <tbody>${bad.map((t) => `<tr><td class="mono tiny nowrap">${esc(t.mpep || "—")}</td>
          <td>${esc(t.requirement || t.id || "")}</td>
          <td class="small">${esc(t.finding || "")}</td></tr>`).join("")}</tbody>
      </table></div>
    </div></details>`;
}

/** A blank cell under an abstract-only column does not mean the document fails
 *  to disclose the element. It means the abstract did not mention it — and an
 *  abstract states what a paper is about, not everything it teaches. Without
 *  this line the chart reads as evidence of absence, which it is not, and
 *  which is why adjudicate.py refuses to let an abstract carry a §102 at all. */
function abstractOnlyCaveat(docs: any[]): string {
  const n = docs.filter((d: any) => d.abstract_only).length;
  if (!n) return "";
  return `<div class="notice notice-warn" style="margin-top:.5rem">
    ${n} of these ${docs.length} column${docs.length === 1 ? "" : "s"} ${n === 1 ? "was" : "were"} read
    <b>as an abstract only</b>. A blank under such a column means the abstract did not mention that
    element — not that the document does not disclose it. Supply the full text below and re-run the
    evidence step before treating one of those blanks as a gap in the art.</div>`;
}

// ─── Candidates — one block each, checklist + verbatim quotes ───

function candidatesSection(cands: any[], checklist: any[], sr: any[]): string {
  if (!cands.length) {
    return `<section class="section" id="sec-candidates"><header><h2>Candidate inventions</h2></header>
      ${empty("No candidate inventions", (R.extraction || {}).no_invention_reason || "Decomposition found nothing to chart.")}</section>`;
  }
  return `<section class="section" id="sec-candidates">
    <header><h2>Candidate inventions (${cands.length})</h2><span class="hint">each element with the verbatim text it rests on</span></header>
    ${cands.map((c: any, i: number) => {
      const own = checklist.filter((x: any) => String(x.id || "").startsWith(`${c.id}.`));
      const els: any[] = c.elements || [];
      const best = bestDocFor(own, sr);
      const key = `cand:${c.id}`;
      return `<details class="cand" data-keep="${esc(key)}" ${isOpen(key, false) ? "open" : ""}>
        <summary class="cand-head">
          <div class="row">
            <span class="cid">${esc(c.id)}</span>
            <span class="pill pill-tag">${esc(c.level || "")}</span>
            ${c.primary_form ? `<span class="pill pill-tag">${esc(c.primary_form)}</span>` : ""}
            ${(c.cpc_pred || []).slice(0, 3).map((x: string) => `<span class="pill pill-tag">${esc(x)}</span>`).join("")}
            ${best ? `<span class="pill ${best.covered >= best.total ? "pill-failed" : "pill-paused"}" title="best single reference against this candidate">${best.covered}/${best.total} covered</span>` : ""}
            <span class="small muted nowrap">${els.length} element${els.length === 1 ? "" : "s"}</span>
          </div>
          <div class="concept">${esc(c.concept || "")}</div>
        </summary>
        <div class="cand-body">
          <div class="tbl-wrap"><div class="tw"><table class="tbl fixed stack-sm">
            <thead><tr><th style="width:12%">Element</th><th style="width:44%">Claim language</th><th style="width:44%">Verbatim basis in the document</th></tr></thead>
            <tbody>${els.map((e: any) => `<tr>
              <td class="mono tiny" data-l="Element">${esc(e.id)}${e.kind ? `<div class="muted">${esc(e.kind)}</div>` : ""}</td>
              <td data-l="Claim language">${esc(e.text)}</td>
              <td data-l="Verbatim basis">${e.evidence_quote
                ? `<div class="quote">${esc(e.evidence_quote)}</div>${locLine(e.evidence_loc)}`
                : `<span class="pill pill-failed">unsupported</span>`}</td></tr>`).join("")}
            </tbody></table></div></div>
          ${c.independent_claim_draft ? `<details class="box"><summary>Independent claim draft from decomposition</summary><div class="box-body">
            ${Object.entries(c.independent_claim_draft).map(([k, t]) => `<div class="subhead">${esc(k)}</div><div class="prose">${esc(String(t))}</div>`).join("")}
          </div></details>` : ""}
          ${(c.dependent_hints || []).length ? `<details class="box"><summary>Dependent hints (${c.dependent_hints.length})</summary><div class="box-body"><ul class="prose">${c.dependent_hints.map((h: any) => `<li>${esc(typeof h === "string" ? h : h.text || JSON.stringify(h))}</li>`).join("")}</ul></div></details>` : ""}
        </div>
      </details>`;
    }).join("")}
  </section>`;
}

function locLine(loc: any): string {
  if (!loc) return "";
  const bits = [loc.heading || loc.section, loc.para != null ? `¶${loc.para}` : "", loc.method].filter(Boolean);
  return `<div class="tiny muted">${esc(bits.join(" · "))}</div>`;
}

function bestDocFor(own: any[], sr: any[]): { covered: number; total: number } | null {
  if (!own.length || !sr.length) return null;
  let best = 0;
  for (const d of sr) {
    const n = own.filter((c) => isCovered(d, c.criterion)).length;
    if (n > best) best = n;
  }
  return { covered: best, total: own.length };
}

// ─── Search ───

function searchSection(): string {
  const s = (R.search || {}).summary || {};
  const rounds: any[] = s.loop_rounds || [];
  const qs = rounds.flatMap((r: any) => (r.queries || []).map((q: any) => ({ ...q, round: r.round })));
  const prune = s.prune || {};
  const quota: any[] = s.serpapi_quota || [];
  if (!rounds.length && !qs.length) {
    return `<section class="section" id="sec-search"><header><h2>Search</h2></header>${empty("No search recorded", "This job produced no recall rounds.")}</section>`;
  }
  return `<section class="section" id="sec-search">
    <header><h2>Prior art search</h2><span class="hint">one row per move, per round</span></header>
    <div class="stack">
      <div class="budget">
        <div><div class="k">Patents</div><div class="v">${num(s.total_patents)}</div></div>
        <div><div class="k">Papers</div><div class="v">${num(s.total_papers)}</div></div>
        <div><div class="k">Unique pool</div><div class="v">${num(s.total_unique)}</div></div>
        <div><div class="k">Downloaded</div><div class="v">${num(s.downloaded)}</div></div>
      </div>
      ${qs.length ? `<details class="box" data-keep="queries"${isOpen("queries", false) ? " open" : ""}>
        <summary>${qs.length} quer${qs.length === 1 ? "y" : "ies"} across ${rounds.length} round${rounds.length === 1 ? "" : "s"}</summary>
        <div class="box-body">${queriesTable(qs)}</div></details>` : ""}
      ${rounds.length ? `<div class="tbl-wrap"><div class="tw"><table class="tbl">
        <thead><tr><th>Round</th><th>Mode</th><th class="right">Queries</th><th class="right">SerpAPI</th><th class="right">Google Patents</th><th class="right">Seeds</th><th class="right">Cited expansion</th><th class="right">Pool</th><th>Uncovered after</th></tr></thead>
        <tbody>${rounds.map((r: any) => `<tr>
          <td class="num">${r.round}</td><td>${esc(r.mode || "")}</td>
          <td class="right num">${num(r.n_queries)}</td><td class="right num">${num(r.serpapi_calls)}</td>
          <td class="right num">${num(r.gp_calls)}${r.gp_blocked ? " (blocked)" : ""}</td>
          <td class="right num">${num(r.seeds)}</td><td class="right num">+${num(r.expanded, "0")}</td>
          <td class="right num">${num(r.pool_size)}</td>
          <td class="mono tiny">${esc((r.uncovered || []).join(" ") || "—")}</td></tr>`).join("")}</tbody>
      </table></div></div>` : ""}
      ${Object.keys(prune).length ? `<div class="subhead">Semantic ranking</div>
        <div class="budget">
          <div><div class="k">In</div><div class="v">${num(prune.stage1_in)}</div></div>
          <div><div class="k">After cosine</div><div class="v">${num(prune.stage1_out)}<small> cut ${prune.stage1_cut_cos ? Number(prune.stage1_cut_cos).toFixed(3) : "—"}</small></div></div>
          <div><div class="k">Worth reading</div><div class="v">${num(prune.stage2_worth)}<small> ${num(prune.stage2_calls, "0")} calls</small></div></div>
          <div><div class="k">Kept</div><div class="v">${num(prune.stage2_out)}<small> ${prune.seconds ? Number(prune.seconds).toFixed(0) + "s" : ""}</small></div></div>
        </div>` : ""}
      ${quota.length ? `<details class="box"><summary>SerpAPI keys (${quota.length})</summary><div class="box-body"><div class="tw"><table class="tbl">
        <thead><tr><th>Key</th><th class="right">Used</th><th class="right">Cap</th></tr></thead>
        <tbody>${quota.map((k: any) => `<tr><td class="mono tiny">${esc(k.key)}</td><td class="right num">${num(k.used)}</td><td class="right num">${num(k.cap)}</td></tr>`).join("")}</tbody>
      </table></div></div></details>` : ""}
    </div>
  </section>`;
}

// ─── Full text we could not reach ───
//
// The references the deep read never actually read, what was tried for each,
// and a slot for the reviewer's own copy. Nothing on this page fetches a paper:
// OSU Libraries' Responsible Use policy forbids programmatic downloading of
// licensed content, and the backend carries the quote (patent_analyzer/
// fulltext.py). The remedy is a person with a browser, and then one button.

const TIER_LABEL: Record<string, string> = {
  bigquery_claims: "Our claims table",
  arxiv: "arXiv",
  oa: "Open access",
  pdf_download: "PDF download",
};

const OUTCOME_PILL: Record<string, string> = {
  ok: "pill-completed", failed: "pill-failed", missed: "pill-pending",
  skipped: "pill-tag", unknown: "pill-tag",
};

function readPill(row: FulltextGapRow): string {
  if (row.read_state === "full_text") return `<span class="pill pill-completed">read in full</span>`;
  if (row.read_state === "abstract_only") return `<span class="pill pill-paused">abstract only</span>`;
  return `<span class="pill pill-failed">nothing read</span>`;
}

// ── Resolved ≠ fetched ───────────────────────────────────────────────────────
//
// Two columns, because they are two facts and the run looks much better when
// they are printed as one. `fulltext_tier` is how far the resolution chain got;
// `fulltext_download` is whether a PDF came back. On the N8 gold set 17
// references resolved 6 open-access URLs and produced 0 readable PDFs — one
// column would have shown six successes.
//
// The tier list is closed: arXiv, open access, abstract only, plus the
// reviewer's own upload. There is no paywalled/EZproxy tier and there will not
// be one; fulltext.py carries the policy and the reason.

const TIER_PILL: Record<string, [string, string]> = {
  arxiv: ["pill-completed", "arXiv"],
  oa: ["pill-completed", "open access"],
  abstract_only: ["pill-failed", "no OA copy"],
  user_upload: ["pill-completed", "your upload"],
};

/** Did a PDF actually arrive? `ok`/`cached` yes; everything else is a reason it
 *  did not, and "not_needed" is the one that is not a loss — the claims text
 *  was already in hand. */
const DOWNLOAD_PILL: Record<string, [string, string]> = {
  ok: ["pill-completed", "PDF fetched"],
  cached: ["pill-completed", "PDF from cache"],
  failed: ["pill-failed", "fetch failed"],
  no_url: ["pill-queued", "no URL to fetch"],
  skipped_budget: ["pill-paused", "budget ran out"],
  not_needed: ["pill-tag", "not needed"],
};

function tierCell(row: FulltextGapRow): string {
  const t = TIER_PILL[row.fulltext_tier];
  if (!t) {
    return `<span class="tiny muted">not recorded</span>
      <div class="tiny muted">this job ran before the tiers were stamped per reference</div>`;
  }
  return `<span class="pill pill-flat ${t[0]}">${esc(t[1])}</span>`;
}

function downloadCell(row: FulltextGapRow): string {
  const d = DOWNLOAD_PILL[row.fulltext_download];
  if (!d) {
    // No stamp: the trail still knows, because _attempts infers it from the URL
    // and the source. Read it back rather than printing a second "unknown".
    const a = row.attempts.find((x) => x.tier === "pdf_download");
    if (!a || a.outcome === "unknown") return `<span class="tiny muted">not recorded</span>`;
    const cls = a.outcome === "ok" ? "pill-completed" : a.outcome === "failed" ? "pill-failed" : "pill-queued";
    return `<span class="pill pill-flat ${cls}">${esc(a.outcome === "ok" ? "PDF fetched" : a.outcome)}</span>
      <div class="tiny muted">${esc(clip(a.detail, 60))} <i>(inferred)</i></div>`;
  }
  return `<span class="pill pill-flat ${d[0]}">${esc(d[1])}</span>`;
}

/** Tiers the run has no record for. "Unknown" is a property of the job, not of
 *  the reference — repeating the same sentence down twenty rows says nothing
 *  about any of them — so those entries leave the table and the tiers are named
 *  once above it. It is still stated: the alternative is a trail that looks
 *  complete while a whole tier is missing from it. */
function unknownTiers(rows: FulltextGapRow[]): Set<string> {
  const out = new Set<string>();
  for (const r of rows) {
    for (const a of r.attempts) if (a.outcome === "unknown") out.add(a.tier);
  }
  return out;
}

function trail(row: FulltextGapRow, hidden: Set<string>): string {
  const shown = row.attempts.filter((a) => a.outcome !== "unknown");
  if (!shown.length) return `<span class="tiny muted">nothing recorded</span>`;
  return `<ul class="trail">${shown.map((a) => `<li>
    <span class="pill pill-flat ${OUTCOME_PILL[a.outcome] || "pill-tag"}">${esc(a.outcome)}</span>
    <b>${esc(TIER_LABEL[a.tier] || a.tier)}</b> ${esc(a.detail)}</li>`).join("")}</ul>`;
}

const ARXIV_ID = /^(\d{4}\.\d{4,5})(v\d+)?$/;

/** Where the reader goes to save the PDF by hand. Without a link the row is
 *  just a complaint, so a bare arXiv id — which is what the arXiv channel puts
 *  in `pub_num` — is turned back into its abstract page. */
function refLinks(row: FulltextGapRow): string {
  const bits: string[] = [];
  if (row.doi) bits.push(`<a href="https://doi.org/${encodeURIComponent(row.doi)}" target="_blank" rel="noopener">doi.org ↗</a>`);
  const aid = ARXIV_ID.exec(row.pub_num || "");
  if (aid) bits.push(`<a href="https://arxiv.org/abs/${esc(aid[1])}" target="_blank" rel="noopener">arXiv ↗</a>`);
  if (row.landing_page && !row.landing_page.includes("doi.org")) {
    bits.push(`<a href="${esc(row.landing_page)}" target="_blank" rel="noopener">publisher page ↗</a>`);
  }
  return bits.join(" · ") || `<span class="muted">no link on the record</span>`;
}

function uploadCell(row: FulltextGapRow, busy: boolean): string {
  const up = row.upload;
  if (up) {
    return `<div class="stack" style="gap:.25rem">
      <div class="small">📄 ${esc(clip(up.filename, 28))} <span class="muted tiny">${(up.bytes / 1048576).toFixed(1)} MB</span></div>
      ${up.reread
        ? `<span class="pill pill-completed pill-flat">read into the report</span>`
        : `<div class="row" style="gap:.35rem"><span class="pill pill-queued pill-flat">waiting for the re-run</span>
             <button class="icon-btn" data-drop="${esc(row.ref_id)}">Remove</button></div>`}
    </div>`;
  }
  return `<div class="stack" style="gap:.3rem">
    <input type="file" accept="application/pdf,.pdf" class="file-in" data-file="${esc(row.ref_id)}" ${busy ? "disabled" : ""}>
    <button class="btn btn-sm" data-up="${esc(row.ref_id)}" ${busy ? "disabled" : ""}>Upload PDF</button>
  </div>`;
}

/** The whole list is 19 rows on a full run, and the page has just been cut from
 *  15,000px to 5,700px to stop people scrolling past it. The rows are sorted by
 *  score, so the first few are the ones whose full text could still change the
 *  determination; the rest are one click away. */
const MAX_GAP_ROWS = 6;

function fulltextSection(): string {
  if (!GAPS || !GAPS.rows.length) return "";
  const g = GAPS, s = g.summary;
  const busy = g.status === "running" || g.status === "queued";
  const hist = g.rerun_history || [];
  const hidden = unknownTiers(g.rows);
  const rows = showAllGaps ? g.rows : g.rows.slice(0, MAX_GAP_ROWS);
  const hiddenNote = [...hidden].map((t) => TIER_LABEL[t] || t).join(", ");
  // The two counts the whole section exists to keep apart: how many references
  // the chain found a full-text link for, and how many of those actually
  // arrived as a readable PDF. The gap between them is the thing to see.
  const resolved = g.rows.filter((r) => r.fulltext_tier === "arxiv" || r.fulltext_tier === "oa").length;
  const fetched = g.rows.filter((r) => (r.fulltext_tier === "arxiv" || r.fulltext_tier === "oa")
    && (r.fulltext_download === "ok" || r.fulltext_download === "cached")).length;
  return `<section class="section" id="sec-fulltext">
    <header><h2>Full text we could not reach</h2>
      <span class="hint">an abstract cannot establish that an element is or is not disclosed</span></header>
    <div class="stack">
      <div class="budget">
        <div><div class="k">Evaluated</div><div class="v">${num(s.evaluated)}</div></div>
        <div><div class="k">Read in full</div><div class="v">${num(s.full_text)}</div></div>
        <div><div class="k">Abstract only</div><div class="v">${num(s.abstract_only)}</div></div>
        <div><div class="k">Nothing read</div><div class="v">${num(s.nothing)}</div></div>
        <div><div class="k">Links resolved</div><div class="v">${num(resolved)}<small> of ${num(g.rows.length)} listed</small></div></div>
        <div><div class="k">PDFs obtained</div><div class="v" style="${resolved && !fetched ? "color:var(--danger)" : ""}">${num(fetched)}<small> of those ${num(resolved)}</small></div></div>
        <div><div class="k">Uploaded</div><div class="v">${num(s.uploaded)}<small>${s.pending_reread ? ` ${s.pending_reread} pending` : ""}</small></div></div>
      </div>
      <div class="notice notice-warn">${esc(g.policy_note)}
        <a href="${esc(g.policy_url)}" target="_blank" rel="noopener">The policy ↗</a></div>
      ${hiddenNote ? `<div class="small muted">Some references on this job carry no record of ${esc(hiddenNote)} —
        they ran before that was written down, so those tiers are left out of their trails below
        rather than shown as failures.</div>` : ""}
      <div class="small muted">Two separate columns below, on purpose. <b>Link resolved</b> is how far the
        open-access chain got — arXiv, an OA copy, or no OA copy at all; those three are the whole list,
        and there is no paywalled tier because there will not be one. <b>PDF obtained</b> is whether the
        file was actually fetched and read. A resolved link is not a read paper: on the reference set this
        was measured against, 6 open-access URLs produced 0 readable PDFs.</div>
      <div class="tw"><table class="tbl stacked">
        <thead><tr><th>Reference</th><th>What was read</th><th class="nowrap">Link resolved</th>
          <th class="nowrap">PDF obtained</th><th>What was tried</th><th class="nowrap">Your copy</th></tr></thead>
        <tbody>${rows.map((row) => `<tr>
          <td data-l="Reference">
            <div>${esc(clip(row.title || row.ref_id, 110))}</div>
            <div class="tiny muted mono">${esc(row.pub_num || row.doi || "—")}</div>
            <div class="tiny">${refLinks(row)}</div>
          </td>
          <td class="nowrap" data-l="What was read">${readPill(row)}
            <div class="tiny muted">${esc(clip(row.read_reason, 70))}</div>
            <div class="tiny muted">score ${row.similarity_score.toFixed(2)}</div></td>
          <td class="nowrap" data-l="Link resolved">${tierCell(row)}</td>
          <td class="nowrap" data-l="PDF obtained">${downloadCell(row)}</td>
          <td data-l="What was tried">${trail(row, hidden)}</td>
          <td data-l="Your copy">${uploadCell(row, busy)}</td></tr>`).join("")}</tbody>
      </table></div>
      ${g.rows.length > MAX_GAP_ROWS ? `<div><button class="icon-btn" id="toggle-gaps">${showAllGaps
        ? `Show only the ${MAX_GAP_ROWS} highest-scoring`
        : `Show every reference (${g.rows.length})`}</button></div>` : ""}
      <div class="row">
        <button class="btn" id="rerun-evidence" ${s.pending_reread && !busy ? "" : "disabled"}>
          Re-run the evidence step${s.pending_reread ? ` (${s.pending_reread})` : ""}</button>
        <span class="small muted">${busy
          ? "This job is running — the page will show the result when it finishes."
          : s.pending_reread
            ? "Reads only the uploaded PDFs, then recomputes the determination and rewrites the report."
            : "Upload a PDF above to enable this."}</span>
      </div>
      <div id="ft-msg"></div>
      ${hist.length ? `<details class="box" data-keep="ft-hist"${isOpen("ft-hist", false) ? " open" : ""}>
        <summary>Earlier re-runs (${hist.length})</summary><div class="box-body"><div class="tw"><table class="tbl">
        <thead><tr><th>When</th><th class="right">Read</th><th class="right">Failed</th><th>Determination</th></tr></thead>
        <tbody>${hist.map((h) => `<tr><td class="tiny mono">${esc(h.at.slice(0, 19))}</td>
          <td class="right num">${h.read.length}</td><td class="right num">${h.failed.length}</td>
          <td class="tiny">${h.changed
            ? `${esc(h.label_before || "—")} → <b>${esc(h.label_after || "—")}</b> · ${esc(clip(h.determination_after, 120))}`
            : `unchanged (${esc(h.label_after || h.label_before || "—")})`}</td></tr>`).join("")}</tbody>
      </table></div></div></details>` : ""}
    </div>
  </section>`;
}

function wireFulltext(host: HTMLElement, jobId: string): void {
  const msg = host.querySelector<HTMLElement>("#ft-msg");
  const say = (html: string) => { if (msg) msg.innerHTML = html; };

  const refresh = async () => {
    GAPS = await getFulltextGaps(jobId).catch(() => GAPS);
    paint(host, jobId);
  };

  on(host, "[data-up]", async (el) => {
    const ref = el.dataset.up!;
    const input = host.querySelector<HTMLInputElement>(`input[data-file="${CSS.escape(ref)}"]`);
    const file = input?.files?.[0];
    if (!file) return say(`<div class="notice notice-warn">Choose a PDF first.</div>`);
    (el as HTMLButtonElement).disabled = true;
    say(`<div class="notice notice-info"><span class="spinner"></span> Uploading ${esc(file.name)}…</div>`);
    try {
      await uploadFulltext(jobId, ref, file);
      await refresh();
    } catch (e: any) {
      (el as HTMLButtonElement).disabled = false;
      say(errorBox(`Upload failed — ${String(e?.message || e)}`));
    }
  });

  on(host, "[data-drop]", async (el) => {
    try {
      await dropFulltextUpload(jobId, el.dataset.drop!);
      await refresh();
    } catch (e: any) {
      say(errorBox(`Could not remove that upload — ${String(e?.message || e)}`));
    }
  });

  on(host, "#toggle-gaps", () => { showAllGaps = !showAllGaps; paint(host, jobId); });

  on(host, "#rerun-evidence", async (el) => {
    (el as HTMLButtonElement).disabled = true;
    say(`<div class="notice notice-info"><span class="spinner"></span> Queued. The uploaded PDFs are being read; the determination and the report follow.</div>`);
    try {
      const r = await rerunEvidence(jobId);
      say(`<div class="notice notice-info">Queued ${r.refs.length} reference(s).
        <a href="#/run/${esc(jobId)}">Watch it run →</a></div>`);
    } catch (e: any) {
      (el as HTMLButtonElement).disabled = false;
      say(errorBox(`Could not start the re-run — ${String(e?.message || e)}`));
    }
  });
}

// ─── Evidence matrix, one per candidate ───

const MAX_DOCS = 8;

/** Was this reference read as an abstract? Same test as the backend's
 *  `adjudicate.abstract_only` — the adjudication's own answer where there is
 *  one, otherwise the row's `source` / `fulltext_tier`. It decides whether a
 *  blank in this grid is "not disclosed" or "not mentioned in the abstract",
 *  which are very different claims to be making on a reader's behalf. */
function isAbstractOnly(doc: any): boolean {
  const key = String(doc.pub_num || doc.title || "");
  const per = ((R.adjudication || {}).per_doc_coverage || [])
    .find((d: any) => String(d.pub_num || d.title || "") === key);
  if (per && typeof per.abstract_only === "boolean") return per.abstract_only;
  return String(doc.source || "").startsWith("abstract") || doc.fulltext_tier === "abstract_only";
}

function matrixSection(cands: any[], checklist: any[], sr: any[]): string {
  if (!sr.length || !checklist.length) {
    return `<section class="section" id="sec-evidence"><header><h2>Evidence matrix</h2></header>
      ${empty("Nothing to chart", "No document was evaluated against the checklist.")}</section>`;
  }
  const groups = cands.length
    ? cands.map((c: any) => ({ id: c.id, concept: c.concept, rows: checklist.filter((x: any) => String(x.id || "").startsWith(`${c.id}.`)) }))
    : [{ id: "all", concept: "", rows: checklist }];

  const blocks = groups.filter((g) => g.rows.length).map((g) => {
    const scored = sr
      .map((d: any) => ({
        doc: d,
        hits: g.rows.reduce((n: number, c: any) => n + (isCovered(d, c.criterion) ? 1 : 0), 0),
        any: g.rows.reduce((n: number, c: any) => n + ((d.checklist_results?.[c.criterion]?.score ?? 0) > 0 ? 1 : 0), 0),
      }))
      .filter((x) => x.any > 0)
      .sort((a, b) => b.hits - a.hits || b.any - a.any);
    const docs = (showAllDocs ? scored : scored.slice(0, MAX_DOCS)).map((x) => x.doc);
    const key = `matrix:${g.id}`;
    if (!docs.length) {
      return `<details class="cand" data-keep="${esc(key)}" ${isOpen(key, false) ? "open" : ""}>
        <summary class="cand-head"><span class="cid">${esc(g.id)}</span>
        <div class="concept">${esc(clip(g.concept, 150))}</div>
        <div class="small muted">no reference touches this candidate</div></summary>
        <div class="cand-body">${empty("No reference touches this candidate", "Every evaluated document scored 0 on all of its elements.")}</div></details>`;
    }
    return `<details class="cand" data-keep="${esc(key)}" ${isOpen(key, false) ? "open" : ""}>
      <summary class="cand-head"><span class="cid">${esc(g.id)}</span>
        <div class="concept">${esc(clip(g.concept, 150))}</div>
        <div class="small muted">${g.rows.length} elements × ${docs.length} of ${scored.length} references that touch it${
          (() => { const hit = scored.filter((x) => x.hits > 0).length; return hit ? ` · ${hit} cover at least one element` : ""; })()}</div></summary>
      <div class="cand-body"><div class="tbl-wrap"><div class="tw"><table class="tbl matrix fixed">
        <thead><tr><th class="el-col" style="width:44%">Element</th>
          ${docs.map((d: any, i: number) => `<th class="doc-col" title="${esc(d.pub_num || "")} — ${esc(d.title || "")}${
            isAbstractOnly(d) ? " (read as an abstract only)" : ""}">D${i + 1}${
            isAbstractOnly(d) ? `<div class="tiny" style="font-weight:400;color:var(--warn)">abs</div>` : ""}</th>`).join("")}</tr></thead>
        <tbody>${g.rows.map((c: any) => `<tr>
          <td class="el-col"><div class="mono tiny muted">${esc(c.id)}</div>${esc(clip(c.criterion, 220))}</td>
          ${docs.map((d: any) => {
            const cr = d.checklist_results?.[c.criterion];
            const sc = cr?.score ?? 0;
            const cls = isCovered(d, c.criterion) ? "cell-covered" : sc > 0 ? "cell-scored" : "cell-absent";
            const key = `${esc(d.pub_num || d.title)}||${esc(c.criterion)}`;
            return `<td class="cell-score ${cls}" ${cr ? `data-cell="${key}" title="score ${sc} — click for the quote and the analysis"` : ""}>${sc || "·"}</td>`;
          }).join("")}</tr>`).join("")}</tbody>
      </table></div></div>
      ${docs.some(isAbstractOnly) ? `<div class="notice notice-warn" style="margin-top:.4rem">
        Columns marked <b>abs</b> were read as an abstract only. A blank in one of those columns means the
        abstract did not mention that element, not that the document does not disclose it.</div>` : ""}
      <div class="tw" style="margin-top:.4rem"><table class="tbl doc-key">
        <tbody>${docs.map((d: any, i: number) => `<tr><td>D${i + 1}</td>
          <td class="mono tiny">${esc(d.pub_num || "—")}</td>
          <td>${esc(clip(d.title, 150))}${isAbstractOnly(d)
            ? ` <span class="pill pill-paused pill-flat">abstract only</span>` : ""}</td>
          <td class="right num">${d.similarity_score ?? ""}</td></tr>`).join("")}</tbody></table></div>
      </div>
    </details>`;
  }).join("");

  return `<section class="section" id="sec-evidence">
    <header><h2>Evidence matrix</h2>
      <span class="hint">one grid per candidate, closed until opened — the cell is the evaluator's score (2 full · 1 partial · · none); shaded means the adjudicator counted it as covered — score ≥ 1 with a verified quote. Click a cell for the quote.</span>
      ${sr.length > MAX_DOCS ? `<button class="icon-btn" id="toggle-docs">${showAllDocs ? "Show the top references only" : "Show every reference that touches a candidate"}</button>` : ""}
    </header>
    ${blocks}
  </section>`;
}

// ─── Draft claims ───

function draftSection(dc: any): string {
  const claims: any[] = dc.claims || [];
  if (!claims.length) {
    return `<section class="section" id="sec-draft"><header><h2>Draft claims</h2></header>
      ${empty("No draft claims", `Strategy: ${esc(dc.strategy || "none")}.`)}</section>`;
  }
  const flags: any[] = ((dc.definiteness || {}).flags || []).filter((f: any) => !f.fixed);
  return `<section class="section" id="sec-draft">
    <header><h2>Draft claims (${claims.length})</h2>
      <span class="hint">strategy <code>${esc(dc.strategy || "")}</code>${dc.candidate_id ? ` · from ${esc(dc.candidate_id)}` : ""}</span>
      ${flags.length ? `<span class="pill pill-failed">${flags.length} open 112(b)</span>` : ""}
    </header>
    <div class="panel"><div class="panel-body">
      ${claims.map((c: any) => {
        const key = `claim:${c.no}`;
        return `<details class="cand" data-keep="${esc(key)}" ${isOpen(key, false) ? "open" : ""}>
        <summary class="cand-head"><div class="row">
          <span class="cid">Claim ${esc(c.no)}</span>
          <span class="pill pill-tag">${esc(c.form || "")}</span>
          ${c.depends_on != null ? `<span class="small muted">depends on ${esc(c.depends_on)}</span>` : ""}
          <span class="small muted nowrap">${(c.limitations || []).length} limitation${(c.limitations || []).length === 1 ? "" : "s"}</span>
        </div><div class="concept">${esc(clip(c.preamble || "", 160))}</div></summary>
        <div class="cand-body">
        <div class="prose">${esc(c.preamble || "")}</div>
        <ul class="prose">${(c.limitations || []).map((l: any) => {
          const cov = l.coverage || {};
          const star = cov.verified && !(cov.covered_by || []).length ? ` <b title="no charted reference covers this limitation">★</b>` : "";
          return `<li><span class="mono tiny muted">${esc(l.lid || "")}</span> ${esc(l.text)}${star}</li>`;
        }).join("")}</ul>
        </div></details>`;
      }).join("")}
      ${dc.avoidance?.reason ? `<div class="small muted">Avoidance: ${esc(dc.avoidance.reason)}</div>` : ""}
    </div></div>
  </section>`;
}

// ─── Reviewer edits ───

function editsSection(edits: any[]): string {
  if (!edits.length) return "";
  return `<section class="section" id="sec-edits">
    <header><h2>Reviewer edits (${edits.length})</h2><span class="hint">changes made at the phase gates</span></header>
    <div class="tbl-wrap"><div class="tw"><table class="tbl">
      <thead><tr><th>Phase</th><th>Kind</th><th>Id</th><th>Op</th><th style="min-width:14rem">Before</th><th style="min-width:14rem">After</th></tr></thead>
      <tbody>${edits.map((e: any) => `<tr>
        <td>${esc(e.phase)}</td><td>${esc(e.kind)}</td><td class="mono tiny">${esc(e.id || e.pub_num || "")}</td>
        <td>${esc(e.op)}</td>
        <td class="small muted">${esc(clip(typeof e.before === "object" ? JSON.stringify(e.before) : e.before, 200))}</td>
        <td class="small">${esc(clip(typeof e.after === "object" ? JSON.stringify(e.after) : e.after, 200))}</td></tr>`).join("")}</tbody>
    </table></div></div>
  </section>`;
}

// ─── Cost and call counts ───

const emptyQuota = (): Quota => ({ available: false, rates: [], other: [], sources: [], upcoming: [], jobCost: null });


/** The measured figure, not a band off the rate card.
 *
 *  This tile used to print `jobCostRange(QUOTA)`, which was always an em dash:
 *  GET /api/quota does not answer a per-job question, and the endpoint that
 *  does — GET /api/jobs/{id}/usage, backend f85ef88 — had no route through the
 *  Express proxy, so nothing could reach it. The route exists now. What comes
 *  back is `metering.ledger()` over the per-phase meter the run kept, so it is
 *  what was spent rather than what a list price suggests it might have been.
 *  When the endpoint has nothing for this job the tile says which, and why, and
 *  does not substitute an estimate. */
/** An evidence re-run rewrites the job record's per-phase metrics, so on a job
 *  that has had one the ledger holds the re-run's phases and nothing else.
 *  Printing that total as "what this job cost" would understate a full run by
 *  two orders of magnitude (96eada38 reads $0.01 against 1ba48316's $1.50 for
 *  the same shape of run), so the page says which phases it is adding up. */
function partialLedger(): boolean {
  return !!LEDGER && !LEDGER.rows.some((r) => r.phase === "idca");
}

function measuredCost(): string {
  const t = LEDGER?.totals;
  if (typeof t?.cost_usd === "number") {
    const phases = new Set(LEDGER!.rows.map((r) => r.phase)).size;
    return partialLedger()
      ? `${fmtUSD(t.cost_usd)}<small> recorded · ${phases} phase${phases === 1 ? "" : "s"} only</small>`
      : `${fmtUSD(t.cost_usd)}<small> measured · ${num(t.llm_calls, "0")} LLM calls</small>`;
  }
  const why = /^404/.test(LEDGER_WHY)
    ? "no usage recorded for this job"
    : LEDGER_WHY ? "the usage endpoint did not answer" : "not read";
  return `${EMDASH}<small> ${esc(why)}</small>`;
}

/** Where the money went, per phase, with the calls that failed or degraded —
 *  the three questions no counter reconstructs after the fact. */
function ledgerBlock(): string {
  const L = LEDGER;
  if (!L) {
    return `<div class="small muted">Per-job spend comes from <code>GET /api/jobs/{id}/usage</code>;
      ${esc(LEDGER_WHY ? `it answered ${clip(LEDGER_WHY, 120)}` : "it was not read")}, so no dollar figure is shown for this run.</div>`;
  }
  const me = L.most_expensive;
  return `<details class="box" data-keep="ledger"${isOpen("ledger", false) ? " open" : ""}>
    <summary>Where it went — ${L.rows.length} ledger row${L.rows.length === 1 ? "" : "s"}${
      me ? ` · ${esc(me.phase)} is ${Math.round(me.share_of_total * 100)}% of it` : ""}</summary>
    <div class="box-body">
      ${partialLedger() ? `<div class="notice notice-warn" style="margin-bottom:.5rem">
        The job record holds ${esc([...new Set(L.rows.map((r) => r.phase))].join(", "))} and no earlier phase.
        An evidence re-run replaces the per-phase metrics, so the search and evaluation this report rests on
        are not in the total above — it is what the record still has, not what the run cost.</div>` : ""}
      <div class="tw"><table class="tbl">
        <thead><tr><th>Phase</th><th>What</th><th class="right">Calls</th><th class="right">Tokens in / out</th>
          <th class="right">Seconds</th><th class="right">Cost</th></tr></thead>
        <tbody>${L.rows.map((r) => `<tr>
          <td>${esc(r.phase)}</td>
          <td class="mono tiny">${esc(r.name)}<div class="muted">${esc(r.kind)}</div></td>
          <td class="right num">${num(r.calls, "0")}</td>
          <td class="right num tiny">${num(r.input_tokens, "—")} / ${num(r.output_tokens, "—")}</td>
          <td class="right num">${r.seconds.toFixed(1)}</td>
          <td class="right num">${fmtUSD(r.cost_usd)}</td></tr>`).join("")}</tbody>
        <tfoot><tr><td colspan="4"><b>Total</b></td>
          <td class="right num"><b>${L.totals.seconds.toFixed(1)}</b></td>
          <td class="right num"><b>${fmtUSD(L.totals.cost_usd)}</b></td></tr></tfoot>
      </table></div>
      ${L.incidents.length ? `<div class="subhead">Calls that retried, failed or degraded (${L.incidents.length})</div>
        <div class="tw"><table class="tbl"><tbody>${L.incidents.map((i) => `<tr>
          <td class="tiny">${esc(i.phase)}</td><td class="mono tiny">${esc(i.source)}</td>
          <td><span class="pill pill-flat ${i.kind === "failed" ? "pill-failed" : "pill-paused"}">${esc(i.kind)}</span></td>
          <td class="tiny muted">${esc(clip(i.detail, 160))}</td></tr>`).join("")}</tbody></table></div>`
        : `<div class="small muted">No call retried, failed or degraded.</div>`}
      ${L.caveats.map((c) => `<div class="tiny muted" style="margin-top:.35rem">${esc(c)}</div>`).join("")}
    </div></details>`;
}

function costSection(sr: any[]): string {
  const s = (R.search || {}).summary || {};
  const rounds: any[] = s.loop_rounds || [];
  const prune = s.prune || {};
  const dc = R.draft_claims || {};
  const llmEvents = EVENTS.filter((e) => e.kind === "llm" || e.kind === "llm_call").length;
  const serp = rounds.reduce((n, r) => n + (r.serpapi_calls || 0), 0);
  const gp = rounds.reduce((n, r) => n + (r.gp_calls || 0), 0);
  const quotaUsed = (s.serpapi_quota || []).reduce((n: number, k: any) => n + (k.used || 0), 0);
  const pv = R.prompt_versions || {};

  const rows: [string, string, string][] = [
    ["Documents read in full", num((R.eval_stats || {}).evaluated ?? sr.length), "one Gemini call per document, against the whole checklist"],
    ["Ranking calls", num(prune.stage2_calls, "0"), "worth-reading screen over the pruned pool"],
    ["Search rounds", num(rounds.length, "0"), `${num(rounds.reduce((n: number, r: any) => n + (r.n_queries || 0), 0), "0")} queries`],
    ["SerpAPI calls", num(serp, "0"), `${num(quotaUsed, "0")} counted against the free-tier keys`],
    ["Google Patents calls", num(gp, "0"), ""],
    ["Draft-claim LLM calls", num(dc.llm_calls, "0"), "reported by the draft node"],
    ["LLM events on the timeline", num(llmEvents, "0"), "calls the event stream announced; the pipeline makes more than it announces"],
  ];

  return `<section class="section" id="sec-cost">
    <header><h2>Cost and call counts</h2>
      <span class="hint">counts are real and come from this job's own record; the dollar figure is measured, from <code>GET /api/jobs/{id}/usage</code>, and the rate card below from <code>GET /api/quota</code></span></header>
    <div class="stack">
      <div class="budget">
        <div><div class="k">This job cost</div><div class="v">${measuredCost()}</div></div>
        <div><div class="k">Documents evaluated</div><div class="v">${num((R.eval_stats || {}).evaluated ?? sr.length)}</div></div>
        <div><div class="k">Quotes verified</div><div class="v">${num(((R.eval_stats || {}).quote_stats || {}).verified)}<small> of ${num(((R.eval_stats || {}).quote_stats || {}).quotes)}</small></div></div>
        <div><div class="k">BigQuery</div><div class="v">${esc(QUOTA?.other.find((o) => /bigquery/i.test(o.item))?.price || EMDASH)}<small> ${QUOTA?.available ? "from /api/quota" : "no rate card"}</small></div></div>
      </div>
      ${ledgerBlock()}
      <div class="tbl-wrap"><div class="tw"><table class="tbl">
        <thead><tr><th>What was called</th><th class="right">Count</th><th>Note</th></tr></thead>
        <tbody>${rows.map(([k, v, n]) => `<tr><td>${esc(k)}</td><td class="right num">${v}</td><td class="small muted">${esc(n)}</td></tr>`).join("")}</tbody>
      </table></div></div>
      <details class="box"><summary>Rate card and prompt versions</summary><div class="box-body">
        <div class="tw"><table class="tbl">
          <thead><tr><th>Model</th><th class="right">In / 1M</th><th class="right">Out / 1M</th></tr></thead>
          <tbody>${(QUOTA?.rates || []).length
            ? (QUOTA?.rates || []).map((r) => `<tr><td class="mono">${esc(r.model)}</td><td class="right num">${perM(r.inPerM)}</td><td class="right num">${perM(r.outPerM)}</td></tr>`).join("")
            : `<tr><td class="mono muted">${EMDASH}</td><td class="right num muted">${EMDASH}</td><td class="right num muted">${EMDASH}</td></tr>`}</tbody>
        </table></div>
        <div class="small muted">${esc(quotaNote(QUOTA || emptyQuota()))}</div>
        ${upcomingBlock(QUOTA)}
        ${Object.keys(pv).length ? `<div class="subhead">Prompt versions used</div><div class="tw"><table class="tbl">
          <thead><tr><th>Prompt</th><th class="right">Version</th></tr></thead>
          <tbody>${Object.entries(pv).map(([k, v]) => `<tr><td class="mono tiny">${esc(k)}</td><td class="right num">${esc(v)}</td></tr>`).join("")}</tbody>
        </table></div>` : ""}
      </div></details>
    </div>
  </section>`;
}
