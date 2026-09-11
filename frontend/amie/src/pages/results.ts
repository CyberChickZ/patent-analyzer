import {
  getResults, getStatus, reportUrl, getFulltextGaps, uploadFulltext, dropFulltextUpload,
  rerunEvidence, type JobEvent, type FulltextGaps, type FulltextGapRow,
} from "../api";
import { esc, clip, pill, num, empty, errorBox, openModal, md, on, plainTitle } from "../ui";
import { queriesTable } from "../hitl";
import { dedupeEvents, isLegacyEvent } from "../phases";
import { loadQuota, jobCostRange, perM, quotaNote, EMDASH, type Quota } from "../pricing";
import { rememberJob } from "../main";

let R: any = null;            // results.json
let EVENTS: JobEvent[] = [];
let showAllDocs = false;
/** Which collapsible blocks the reader has opened, so a repaint (Show every
 *  reference) does not close them again. */
let opened = new Set<string>();
let QUOTA: Quota | null = null;
let GAPS: FulltextGaps | null = null;
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

export function disposeResults(): void {
  R = null; EVENTS = []; showAllDocs = false; QUOTA = null; GAPS = null; showAllGaps = false;
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

interface Sec { id: string; label: string; }

export async function renderResults(host: HTMLElement, jobId: string): Promise<void> {
  disposeResults();
  if (!jobId) {
    host.innerHTML = `<div class="page-head"><div class="kicker">Step 3</div><h1>Results</h1></div>
      ${empty("No job selected", `Pick a completed job from <a href="#/submit">Submit</a>.`)}`;
    return;
  }
  rememberJob(jobId);
  host.innerHTML = `<div class="page-head"><div class="kicker">Step 3</div>
    <h1>Results <span class="mono small muted">${esc(jobId)}</span></h1>
    <div class="lede"><span class="spinner"></span> Loading results.json — a full run's is tens of MB, so this can take a moment.</div></div>`;

  try {
    const [res, st, q, gaps] = await Promise.all([
      getResults(jobId),
      getStatus(jobId).catch(() => null),
      loadQuota(),
      // A job from before this endpoint existed, or a backend that is older
      // than the page, just means the section is not drawn.
      getFulltextGaps(jobId).catch(() => null),
    ]);
    QUOTA = q;
    GAPS = gaps;
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
  paint(host, jobId);
}

function paint(host: HTMLElement, jobId: string): void {
  buildCoverage();
  const p1 = R.phase1 || {};
  const cands: any[] = (R.extraction || {}).candidate_inventions || [];
  const checklist: any[] = (R.phase2 || {}).checklist || [];
  const sr: any[] = (R.evaluation || {}).scoring_report || [];
  const adj = R.adjudication || {};

  host.innerHTML = `
  <div class="page-head">
    <div class="kicker">Step 3 · ${esc(R.job_id || jobId)}</div>
    <h1>${esc(clip(plainTitle(R.source_title) || R.source_filename || "Results", 140))}</h1>
    <div class="lede">
      ${pill("completed")} ${esc(p1.status_determination || "—")} ·
      ${esc(p1.input_mode || "—")} · CPC ${esc(p1.cpc_subclass || "—")} ·
      <a href="${reportUrl(jobId)}" target="_blank" rel="noopener">Full HTML report ↗</a> ·
      <a href="/api/results/${esc(jobId)}" target="_blank" rel="noopener">results.json ↗</a> ·
      <a href="#/run/${esc(jobId)}">Event log →</a>
    </div>
  </div>

  ${secNav(R.user_edits || [])}
  ${determinationSection(adj, sr)}
  ${candidatesSection(cands, checklist, sr)}
  ${searchSection()}
  ${fulltextSection()}
  ${matrixSection(cands, checklist, sr)}
  ${draftSection(R.draft_claims || {})}
  ${editsSection(R.user_edits || [])}
  ${costSection(sr)}
  `;

  wireSecNav(host);
  host.querySelectorAll<HTMLDetailsElement>("details[data-keep]").forEach((d) => {
    d.addEventListener("toggle", () => markOpen(d.dataset.keep!, d.open));
  });

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
    { id: "sec-verdict", label: "Verdict" },
    { id: "sec-candidates", label: "Candidates" },
    { id: "sec-search", label: "Search" },
    ...(GAPS && GAPS.rows.length ? [{ id: "sec-fulltext", label: "Full text" }] : []),
    { id: "sec-evidence", label: "Evidence" },
    { id: "sec-draft", label: "Draft" },
    ...(edits.length ? [{ id: "sec-edits", label: "Reviewer edits" }] : []),
    { id: "sec-cost", label: "Cost" },
  ];
}

function secNav(edits: any[]): string {
  return `<nav class="secnav" aria-label="Sections of this report">
    ${sections(edits).map((x) => `<a href="#${x.id}" data-sec="${x.id}">${esc(x.label)}</a>`).join("")}
  </nav>`;
}

function wireSecNav(host: HTMLElement): void {
  const links = new Map<string, HTMLElement>();
  host.querySelectorAll<HTMLElement>(".secnav a[data-sec]").forEach((a) => {
    links.set(a.dataset.sec!, a);
    // the hash is the router's, so the anchor scrolls by hand
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      document.getElementById(a.dataset.sec!)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
  spy?.disconnect();
  // Mark the section whose heading last crossed the top of the reading area.
  spy = new IntersectionObserver((entries) => {
    for (const en of entries) {
      if (!en.isIntersecting) continue;
      links.forEach((el, id) => el.classList.toggle("on", id === en.target.id));
    }
  }, { rootMargin: "-25% 0px -70% 0px", threshold: 0 });
  links.forEach((_, id) => {
    const el = document.getElementById(id);
    if (el) spy!.observe(el);
  });
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
 *  `label_text` is not on any endpoint yet: results.json's `adjudication` (see
 *  graph/eval_subgraph.py, which stores what `adjudicate()` returns) carries
 *  label / basis / risk / reason but not the rendered label, which today only
 *  report_sections and report_generator call. Until the backend adds one line
 *  putting `determination_label(adj)` on the dict, this renders nothing extra
 *  and the reader gets `adj.reason`, which is the same author's full sentence
 *  and already carries the corrected wording. */
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
      ${adj.label_text ? `<div class="verdict-headline">${esc(adj.label_text)}</div>` : ""}
      <div class="prose">${esc(adj.reason || "")}</div>
      ${adj.obviousness_explanation ? `<details class="box" open>
        <summary>Obviousness explanation <span class="pill pill-tag">model narrative, not the rule's output</span></summary>
        <div class="box-body prose">${md(adj.obviousness_explanation)}</div></details>` : ""}
      ${docs.length ? `<div class="tbl-wrap"><div class="tw"><table class="tbl">
        <thead><tr><th>Charted reference</th><th style="min-width:14rem">Title</th><th class="right">Elements covered</th><th class="right">Coverage</th></tr></thead>
        <tbody>${docs.map((d: any) => `<tr><td class="mono tiny">${esc(d.pub_num || "—")}</td>
          <td>${esc(clip(d.title, 130))}</td><td class="right num">${d.n_covered ?? "—"}/${chart.n_elements ?? "—"}</td>
          <td class="right num">${d.coverage != null ? Math.round(d.coverage * 100) + "%" : "—"}</td></tr>`).join("")}</tbody>
      </table></div></div>` : ""}
      ${(chart.uncovered || []).length ? `<div class="small"><b>Uncovered by every charted reference:</b> <span class="muted">${esc((chart.uncovered || []).join(" · "))}</span></div>` : ""}
      <div class="small muted">${sr.length} document${sr.length === 1 ? "" : "s"} evaluated in full.</div>
    </div></div>
  </section>`;
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
      return `<details class="cand" data-keep="${esc(key)}" ${isOpen(key, i === 0) ? "open" : ""}>
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
      ${queriesTable(qs)}
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
  return `<section class="section" id="sec-fulltext">
    <header><h2>Full text we could not reach</h2>
      <span class="hint">an abstract cannot establish that an element is or is not disclosed</span></header>
    <div class="stack">
      <div class="budget">
        <div><div class="k">Evaluated</div><div class="v">${num(s.evaluated)}</div></div>
        <div><div class="k">Read in full</div><div class="v">${num(s.full_text)}</div></div>
        <div><div class="k">Abstract only</div><div class="v">${num(s.abstract_only)}</div></div>
        <div><div class="k">Nothing read</div><div class="v">${num(s.nothing)}</div></div>
        <div><div class="k">Uploaded</div><div class="v">${num(s.uploaded)}<small>${s.pending_reread ? ` ${s.pending_reread} pending` : ""}</small></div></div>
      </div>
      <div class="notice notice-warn">${esc(g.policy_note)}
        <a href="${esc(g.policy_url)}" target="_blank" rel="noopener">The policy ↗</a></div>
      ${hiddenNote ? `<div class="small muted">No per-reference record of ${esc(hiddenNote)} on this job —
        it ran before that was written down, so those tiers are left out of the trails below
        rather than shown as failures.</div>` : ""}
      <div class="tw"><table class="tbl stacked">
        <thead><tr><th>Reference</th><th>What was read</th><th>What was tried</th><th class="nowrap">Your copy</th></tr></thead>
        <tbody>${rows.map((row) => `<tr>
          <td data-l="Reference">
            <div>${esc(clip(row.title || row.ref_id, 110))}</div>
            <div class="tiny muted mono">${esc(row.pub_num || row.doi || "—")}</div>
            <div class="tiny">${refLinks(row)}</div>
          </td>
          <td class="nowrap" data-l="What was read">${readPill(row)}
            <div class="tiny muted">${esc(clip(row.read_reason, 70))}</div>
            <div class="tiny muted">score ${row.similarity_score.toFixed(2)}</div></td>
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
          ${docs.map((d: any, i: number) => `<th class="doc-col" title="${esc(d.pub_num || "")} — ${esc(d.title || "")}">D${i + 1}</th>`).join("")}</tr></thead>
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
      <div class="tw" style="margin-top:.4rem"><table class="tbl doc-key">
        <tbody>${docs.map((d: any, i: number) => `<tr><td>D${i + 1}</td>
          <td class="mono tiny">${esc(d.pub_num || "—")}</td>
          <td>${esc(clip(d.title, 150))}</td>
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
      ${claims.map((c: any) => `<div style="margin-bottom:.9rem">
        <div class="subhead">Claim ${esc(c.no)} <span class="pill pill-tag">${esc(c.form || "")}</span>
          ${c.depends_on != null ? `<span class="small muted">depends on ${esc(c.depends_on)}</span>` : ""}</div>
        <div class="prose">${esc(c.preamble || "")}</div>
        <ul class="prose">${(c.limitations || []).map((l: any) => {
          const cov = l.coverage || {};
          const star = cov.verified && !(cov.covered_by || []).length ? ` <b title="no charted reference covers this limitation">★</b>` : "";
          return `<li><span class="mono tiny muted">${esc(l.lid || "")}</span> ${esc(l.text)}${star}</li>`;
        }).join("")}</ul>
      </div>`).join("")}
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

const emptyQuota = (): Quota => ({ available: false, rates: [], other: [], sources: [], jobCost: null });


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
      <span class="hint">counts are real and come from this job's own record; anything in dollars comes from <code>GET /api/quota</code> or is left blank</span></header>
    <div class="stack">
      <div class="budget">
        <div><div class="k">Estimated for this job</div><div class="v">${jobCostRange(QUOTA)}</div></div>
        <div><div class="k">Documents evaluated</div><div class="v">${num((R.eval_stats || {}).evaluated ?? sr.length)}</div></div>
        <div><div class="k">Quotes verified</div><div class="v">${num(((R.eval_stats || {}).quote_stats || {}).verified)}<small> of ${num(((R.eval_stats || {}).quote_stats || {}).quotes)}</small></div></div>
        <div><div class="k">BigQuery</div><div class="v">${esc(QUOTA?.other.find((o) => /bigquery/i.test(o.item))?.price || EMDASH)}<small> ${QUOTA?.available ? "from /api/quota" : "no rate card"}</small></div></div>
      </div>
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
        ${Object.keys(pv).length ? `<div class="subhead">Prompt versions used</div><div class="tw"><table class="tbl">
          <thead><tr><th>Prompt</th><th class="right">Version</th></tr></thead>
          <tbody>${Object.entries(pv).map(([k, v]) => `<tr><td class="mono tiny">${esc(k)}</td><td class="right num">${esc(v)}</td></tr>`).join("")}</tbody>
        </table></div>` : ""}
      </div></details>
    </div>
  </section>`;
}
