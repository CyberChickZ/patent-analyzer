import { getResults, getStatus, reportUrl, type JobEvent } from "../api";
import { esc, clip, pill, num, empty, errorBox, openModal, md, on, plainTitle } from "../ui";
import { queriesTable } from "../hitl";
import { dedupeEvents, isLegacyEvent } from "../phases";
import { RATES, jobCostRange } from "../pricing";
import { rememberJob } from "../main";

let R: any = null;            // results.json
let EVENTS: JobEvent[] = [];
let showAllDocs = false;

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
  R = null; EVENTS = []; showAllDocs = false;
}

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
    const [res, st] = await Promise.all([
      getResults(jobId),
      getStatus(jobId).catch(() => null),
    ]);
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

  ${determinationSection(adj, sr)}
  ${candidatesSection(cands, checklist, sr)}
  ${searchSection()}
  ${matrixSection(cands, checklist, sr)}
  ${draftSection(R.draft_claims || {})}
  ${editsSection(R.user_edits || [])}
  ${costSection(sr)}
  `;

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
}

// ─── Determination ───

function determinationSection(adj: any, sr: any[]): string {
  if (!adj || !adj.label) {
    return `<section class="section"><header><h2>Determination</h2></header>
      ${empty("No determination", "The rule adjudicator produced no verdict for this job.")}</section>`;
  }
  const chart = adj.claim_chart || {};
  const docs: any[] = chart.docs || [];
  const vclass = adj.label === "102" || adj.label === "103" ? `v-${adj.label}` : "v-allow";
  return `<section class="section">
    <header><h2>Determination and its basis</h2><span class="hint">rule verdict from the charted evidence</span></header>
    <div class="panel"><div class="panel-body stack">
      <div class="verdict ${vclass}">
        <span class="label">§${esc(adj.label)}</span>
        <span class="pill pill-tag">${esc(adj.basis || "")}</span>
        <span class="pill ${adj.risk === "blocking" ? "pill-failed" : "pill-paused"}">${esc(adj.risk || "")}</span>
        <span class="small muted">${esc(adj.n_elements || 0)} elements · best single reference covers ${adj.best_coverage != null ? Math.round(adj.best_coverage * 100) + "%" : "—"}</span>
      </div>
      <div class="prose">${esc(adj.reason || "")}</div>
      ${adj.obviousness_explanation ? `<details class="box" open><summary>§103 reasoning</summary><div class="box-body prose">${md(adj.obviousness_explanation)}</div></details>` : ""}
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
    return `<section class="section"><header><h2>Candidate inventions</h2></header>
      ${empty("No candidate inventions", (R.extraction || {}).no_invention_reason || "Decomposition found nothing to chart.")}</section>`;
  }
  return `<section class="section">
    <header><h2>Candidate inventions (${cands.length})</h2><span class="hint">each element with the verbatim text it rests on</span></header>
    ${cands.map((c: any) => {
      const own = checklist.filter((x: any) => String(x.id || "").startsWith(`${c.id}.`));
      const els: any[] = c.elements || [];
      const best = bestDocFor(own, sr);
      return `<div class="cand">
        <div class="cand-head">
          <div class="row">
            <span class="cid">${esc(c.id)}</span>
            <span class="pill pill-tag">${esc(c.level || "")}</span>
            ${c.primary_form ? `<span class="pill pill-tag">${esc(c.primary_form)}</span>` : ""}
            ${(c.cpc_pred || []).slice(0, 3).map((x: string) => `<span class="pill pill-tag">${esc(x)}</span>`).join("")}
            ${best ? `<span class="pill ${best.covered >= best.total ? "pill-failed" : "pill-paused"}" title="best single reference against this candidate">${best.covered}/${best.total} covered</span>` : ""}
          </div>
          <div class="concept">${esc(c.concept || "")}</div>
        </div>
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
      </div>`;
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
    return `<section class="section"><header><h2>Search</h2></header>${empty("No search recorded", "This job produced no recall rounds.")}</section>`;
  }
  return `<section class="section">
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

// ─── Evidence matrix, one per candidate ───

const MAX_DOCS = 8;

function matrixSection(cands: any[], checklist: any[], sr: any[]): string {
  if (!sr.length || !checklist.length) {
    return `<section class="section"><header><h2>Evidence matrix</h2></header>
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
    if (!docs.length) {
      return `<div class="cand"><div class="cand-head"><span class="cid">${esc(g.id)}</span>
        <div class="concept">${esc(clip(g.concept, 150))}</div></div>
        <div class="cand-body">${empty("No reference touches this candidate", "Every evaluated document scored 0 on all of its elements.")}</div></div>`;
    }
    return `<div class="cand">
      <div class="cand-head"><span class="cid">${esc(g.id)}</span>
        <div class="concept">${esc(clip(g.concept, 150))}</div>
        <div class="small muted">${g.rows.length} elements × ${docs.length} of ${scored.length} references that touch it</div></div>
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
    </div>`;
  }).join("");

  return `<section class="section">
    <header><h2>Evidence matrix</h2>
      <span class="hint">the cell is the evaluator's score (2 full · 1 partial · · none); shaded means the adjudicator counted it as covered — score ≥ 1 with a verified quote. Click a cell for the quote.</span>
      ${sr.length > MAX_DOCS ? `<button class="icon-btn" id="toggle-docs">${showAllDocs ? "Show the top references only" : "Show every reference that touches a candidate"}</button>` : ""}
    </header>
    ${blocks}
  </section>`;
}

// ─── Draft claims ───

function draftSection(dc: any): string {
  const claims: any[] = dc.claims || [];
  if (!claims.length) {
    return `<section class="section"><header><h2>Draft claims</h2></header>
      ${empty("No draft claims", `Strategy: ${esc(dc.strategy || "none")}.`)}</section>`;
  }
  const flags: any[] = ((dc.definiteness || {}).flags || []).filter((f: any) => !f.fixed);
  return `<section class="section">
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
  return `<section class="section">
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

  return `<section class="section">
    <header><h2>Cost and call counts</h2>
      <span class="hint">counts are real; the dollar figure is an estimate — app/llm.py meters tokens per model but no endpoint exposes them</span></header>
    <div class="stack">
      <div class="budget">
        <div><div class="k">Estimated for this job</div><div class="v">${jobCostRange()}</div></div>
        <div><div class="k">Documents evaluated</div><div class="v">${num((R.eval_stats || {}).evaluated ?? sr.length)}</div></div>
        <div><div class="k">Quotes verified</div><div class="v">${num(((R.eval_stats || {}).quote_stats || {}).verified)}<small> of ${num(((R.eval_stats || {}).quote_stats || {}).quotes)}</small></div></div>
        <div><div class="k">SerpAPI spend</div><div class="v">$0<small> free tier</small></div></div>
      </div>
      <div class="tbl-wrap"><div class="tw"><table class="tbl">
        <thead><tr><th>What was called</th><th class="right">Count</th><th>Note</th></tr></thead>
        <tbody>${rows.map(([k, v, n]) => `<tr><td>${esc(k)}</td><td class="right num">${v}</td><td class="small muted">${esc(n)}</td></tr>`).join("")}</tbody>
      </table></div></div>
      <details class="box"><summary>Rate card and prompt versions</summary><div class="box-body">
        <div class="tw"><table class="tbl">
          <thead><tr><th>Model</th><th class="right">In / 1M</th><th class="right">Out / 1M</th></tr></thead>
          <tbody>${RATES.map((r) => `<tr><td class="mono">${esc(r.model)}</td><td class="right num">$${r.inPerM.toFixed(2)}</td><td class="right num">$${r.outPerM.toFixed(2)}</td></tr>`).join("")}</tbody>
        </table></div>
        ${Object.keys(pv).length ? `<div class="subhead">Prompt versions used</div><div class="tw"><table class="tbl">
          <thead><tr><th>Prompt</th><th class="right">Version</th></tr></thead>
          <tbody>${Object.entries(pv).map(([k, v]) => `<tr><td class="mono tiny">${esc(k)}</td><td class="right num">${esc(v)}</td></tr>`).join("")}</tbody>
        </table></div>` : ""}
      </div></details>
    </div>
  </section>`;
}
