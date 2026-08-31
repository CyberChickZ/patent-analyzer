/** The review panel shown while a job is paused at a phase gate.
 *
 *  The behaviour here is the one that was verified end to end: GET the paused
 *  phase's editable values, let the reviewer change them, PATCH whole-value
 *  replacements of the editable keys, then POST resume with continue or
 *  rerun_phase. Only the presentation changed. */

import { getPhaseState, patchPhaseState, resumeJob, getPrompt, putPrompt, type PhaseState } from "./api";
import { esc, clip, pill, errorBox } from "./ui";
import { PAUSE_LABEL, PHASE_PROMPTS } from "./phases";

export type Collect = () => Record<string, unknown>;

export async function mountReview(host: HTMLElement, jobId: string, onResumed: () => void): Promise<void> {
  let st: PhaseState;
  try {
    st = await getPhaseState(jobId);
  } catch (e: any) {
    host.innerHTML = errorBox(`Could not load the paused state — ${e?.message || e}`);
    return;
  }
  const phase = st.paused_at || "extract";
  const v = st.values || {};
  const ctx = st.context || {};

  const body =
    phase === "idca" ? idcaBody(v, ctx)
    : phase === "extract" ? extractBody(v)
    : phase === "search" ? searchBody(v, ctx)
    : phase === "evaluate" ? evaluateBody(v, ctx)
    : phase === "draft" ? draftBody(v.draft_claims || {}, ctx.adjudication || {})
    : `<div class="small muted">Nothing editable at this gate.</div>`;

  const names = PHASE_PROMPTS[phase] || [];
  host.innerHTML = `
  <div class="review" id="review">
    <div class="review-head">
      <span class="pill pill-paused">paused</span>
      <span class="title">Review — ${esc(PAUSE_LABEL[phase] || phase)}</span>
      <span class="small muted" style="flex:1">What you leave here is what the next phase receives; every change is recorded in the report.</span>
    </div>
    <div class="review-body">
      <div id="review-body">${body}</div>
      <div id="prompt-boxes">${names.length ? `<div class="small muted" style="margin-top:.9rem"><span class="spinner"></span> loading prompts…</div>` : ""}</div>
    </div>
    <div class="review-actions">
      <button class="btn" id="btn-continue">Continue</button>
      ${names.length ? `<button class="btn btn-ghost" id="btn-rerun">Rerun this phase</button>` : ""}
      <span class="small muted" id="review-msg"></span>
    </div>
  </div>`;

  const div = host.querySelector<HTMLElement>("#review")!;
  const msg = div.querySelector<HTMLElement>("#review-msg")!;

  if (names.length) void mountPrompts(div.querySelector<HTMLElement>("#prompt-boxes")!, names, msg);

  const collect: Collect = () => collectEdits(div, phase, v);

  div.querySelector<HTMLButtonElement>("#btn-rerun")?.addEventListener("click", async (ev) => {
    const b = ev.currentTarget as HTMLButtonElement;
    b.disabled = true; b.textContent = "Rerunning…";
    try {
      await resumeJob(jobId, "rerun_phase");
      div.innerHTML = `<div class="review-body"><div class="notice notice-info">
        ${pill("running", "rerunning")} Rerunning ${esc(PAUSE_LABEL[phase] || phase)} with the current prompt versions — it will pause here again.</div></div>`;
      onResumed();
    } catch (e: any) {
      b.disabled = false; b.textContent = "Rerun this phase";
      msg.textContent = `Failed: ${e?.message || e}`;
    }
  });

  div.querySelector<HTMLButtonElement>("#btn-continue")!.addEventListener("click", async (ev) => {
    const b = ev.currentTarget as HTMLButtonElement;
    b.disabled = true; b.textContent = "Resuming…";
    try {
      const edits = collect();
      if (Object.keys(edits).length) await patchPhaseState(jobId, edits);
      await resumeJob(jobId, "continue");
      div.innerHTML = `<div class="review-body"><div class="notice notice-info">
        ${pill("running", "resuming")} Continuing after ${esc(PAUSE_LABEL[phase] || phase)}${Object.keys(edits).length ? ` with your edits to ${esc(Object.keys(edits).join(", "))}` : ""}.</div></div>`;
      onResumed();
    } catch (e: any) {
      b.disabled = false; b.textContent = "Continue";
      msg.textContent = `Failed: ${e?.message || e}`;
    }
  });
}

// ─── Bodies ───

function idcaBody(v: any, ctx: any): string {
  return `<div class="subhead">Detection</div>
  <div class="tbl-wrap"><div class="tw"><table class="tbl">
    <tbody>
      <tr><th style="width:11rem">Title</th><td>${esc(ctx.source_title || "—")}</td></tr>
      <tr><th>Determination</th><td><select class="ed" data-key="status_determination" style="max-width:14rem">
        ${["Present", "Implied", "Absent"].map((o) => `<option ${o === v.status_determination ? "selected" : ""}>${o}</option>`).join("")}
      </select></td></tr>
      <tr><th>Input mode</th><td><select class="ed" data-key="input_mode" style="max-width:20rem">
        ${["academic_paper", "manuscript", "disclosure", "patent_draft", "invention"].map((o) => `<option ${o === v.input_mode ? "selected" : ""}>${o}</option>`).join("")}
      </select></td></tr>
      <tr><th>CPC subclass</th><td><input type="text" class="ed" data-key="cpc_subclass" style="max-width:10rem" value="${esc(v.cpc_subclass || "")}"></td></tr>
      <tr><th>Doc type</th><td class="small muted">${esc(ctx.doc_type || "—")} · ${esc(ctx.category || "—")}</td></tr>
    </tbody></table></div></div>
  <div class="subhead">Invention summary <span class="small muted">— editable; this is what decomposition reads</span></div>
  <textarea class="ed" data-key="summary" rows="12">${esc(v.summary || "")}</textarea>`;
}

function extractBody(v: any): string {
  const cands: any[] = (v.extraction && v.extraction.candidate_inventions) || [];
  if (!cands.length) {
    const cl: any[] = v.checklist || [];
    if (!cl.length) return `<div class="empty"><div class="empty-title">No candidate inventions</div><div>Decomposition found nothing to chart.</div></div>`;
    return `<div class="subhead">Checklist (${cl.length})</div>${checklistTable(cl)}`;
  }
  return cands.map((c: any, ci: number) => `
    <div class="cand">
      <div class="cand-head">
        <div class="row">
          <span class="cid">${esc(c.id)}</span>
          <span class="pill pill-tag">${esc(c.level || "")}</span>
          ${c.primary_form ? `<span class="pill pill-tag">${esc(c.primary_form)}</span>` : ""}
          ${(c.cpc_pred || []).slice(0, 3).map((x: string) => `<span class="pill pill-tag">${esc(x)}</span>`).join("")}
          <label class="check" style="margin-left:auto"><input type="checkbox" class="keep-cand" data-ci="${ci}" checked><span>keep</span></label>
        </div>
        <div class="concept">${esc(c.concept || "")}</div>
      </div>
      <div class="cand-body">
        <div class="tbl-wrap"><div class="tw"><table class="tbl">
          <thead><tr><th style="width:5rem">Element</th><th style="min-width:16rem">Text — editable</th><th style="min-width:14rem">Evidence in the document</th><th style="width:3.5rem">Keep</th></tr></thead>
          <tbody>${(c.elements || []).map((e: any, ei: number) => `<tr>
            <td class="mono tiny">${esc(e.id)}${e.kind ? `<div class="muted">${esc(e.kind)}</div>` : ""}</td>
            <td><textarea class="el" data-ci="${ci}" data-ei="${ei}" rows="2">${esc(e.text)}</textarea></td>
            <td>${e.evidence_quote ? `<div class="quote">${esc(clip(e.evidence_quote, 260))}</div>` : `<span class="small muted">—</span>`}
                ${e.unsupported ? `<span class="pill pill-failed">unsupported</span>` : ""}</td>
            <td><input type="checkbox" class="keep-el" data-ci="${ci}" data-ei="${ei}" checked></td></tr>`).join("")}
          </tbody></table></div></div>
      </div>
    </div>`).join("") + (v.checklist?.length ? `<details class="box"><summary>Checklist (${v.checklist.length}) — what the evaluation scores against</summary><div class="box-body">${checklistTable(v.checklist)}</div></details>` : "");
}

function checklistTable(cl: any[]): string {
  return `<div class="tbl-wrap"><div class="tw"><table class="tbl">
    <thead><tr><th style="width:6rem">Id</th><th>Criterion</th><th class="right" style="width:5rem">Weight</th></tr></thead>
    <tbody>${cl.map((c: any, i: number) => `<tr><td class="mono tiny">${esc(c.id || `c${i + 1}`)}</td>
      <td>${esc(c.criterion)}</td><td class="right num">${c.weight ?? "—"}</td></tr>`).join("")}</tbody>
  </table></div></div>`;
}

export function queriesTable(qs: any[]): string {
  if (!qs.length) return `<div class="small muted">No queries recorded.</div>`;
  return `<div class="tbl-wrap"><div class="tw"><table class="tbl">
    <thead><tr><th>#</th><th>Round</th><th>Move</th><th>Channel</th><th>Elements</th>
      <th class="right">Total</th><th class="right">Hits</th><th class="right">New</th><th style="min-width:18rem">Query</th></tr></thead>
    <tbody>${qs.map((q: any, i: number) => {
      const skipped = String(q.channel || "").startsWith("skipped");
      return `<tr${skipped ? ' class="muted"' : ""}>
        <td class="num">${q.n ?? i + 1}</td>
        <td class="num">${q.round ?? "—"}</td>
        <td><span class="pill pill-tag">${esc(q.kind || q.mode || "?")}</span></td>
        <td class="small">${esc(q.channel || "—")}</td>
        <td class="mono tiny">${esc((q.target_elements || q.elements || []).join(" "))}</td>
        <td class="right num">${q.total ?? "—"}</td>
        <td class="right num">${q.hits ?? "—"}</td>
        <td class="right num">${q.new ?? "—"}</td>
        <td class="q">${esc(clip(q.query, 300))}
          ${q.decision ? `<details class="box"><summary>why</summary><div class="box-body small">${esc(q.observation || "")}<br><b>→</b> ${esc(q.decision)}</div></details>` : ""}</td>
      </tr>`;
    }).join("")}</tbody></table></div></div>`;
}

function searchBody(v: any, ctx: any): string {
  const ranked: any[] = v.ranked_candidates || [];
  const stats = ctx.search_stats || {};
  const qs: any[] = ctx.queries || [];
  const prune = stats.prune || {};
  return `
  <div class="subhead">Recall — ${qs.length} quer${qs.length === 1 ? "y" : "ies"}, one row per move</div>
  ${queriesTable(qs)}
  ${Object.keys(prune).length ? `<div class="subhead">Semantic ranking</div>
    <div class="budget">
      <div><div class="k">Pool</div><div class="v">${prune.pool ?? "—"}</div></div>
      <div><div class="k">After cosine</div><div class="v">${prune.stage1_out ?? "—"}<small> cut ${prune.stage1_cut_cos ? Number(prune.stage1_cut_cos).toFixed(3) : "—"}</small></div></div>
      <div><div class="k">Worth reading</div><div class="v">${prune.stage2_worth ?? "—"}<small> ${prune.stage2_calls ?? 0} calls</small></div></div>
      <div><div class="k">Kept</div><div class="v">${prune.stage2_out ?? "—"}</div></div>
    </div>` : ""}
  <div class="subhead">Going to evaluation (${ranked.length}) <span class="small muted">— untick to drop</span></div>
  ${ranked.length ? `<div class="tbl-wrap"><div class="tw"><table class="tbl">
    <thead><tr><th>#</th><th>Publication</th><th style="min-width:16rem">Title</th><th>Type</th><th>Sources</th><th style="width:3.5rem">Keep</th></tr></thead>
    <tbody>${ranked.map((d: any, i: number) => `<tr>
      <td class="num">${i + 1}</td>
      <td class="mono tiny">${esc(d.pub_num || "—")}</td>
      <td>${esc(clip(d.title, 160))}</td>
      <td class="small muted">${esc(d.match_type || "")}</td>
      <td class="small muted">${esc((d.sources || []).join(", "))}</td>
      <td><input type="checkbox" class="keep-doc" data-i="${i}" checked></td></tr>`).join("")}
    </tbody></table></div></div>` : `<div class="empty"><div class="empty-title">Nothing survived the ranking</div><div>The pipeline will skip evaluation and go straight to the report.</div></div>`}`;
}

function evaluateBody(v: any, ctx: any): string {
  const sr: any[] = v.scoring_report || [];
  const qs = (ctx.eval_stats || {}).quote_stats || {};
  if (!sr.length) return `<div class="empty"><div class="empty-title">Nothing evaluated</div><div>No document made it through to evaluation.</div></div>`;
  return `
  ${Object.keys(qs).length ? `<div class="budget" style="margin-bottom:.7rem">
    <div><div class="k">Documents read</div><div class="v">${qs.docs_verified ?? "—"}</div></div>
    <div><div class="k">Quotes</div><div class="v">${qs.quotes ?? "—"}</div></div>
    <div><div class="k">Verified</div><div class="v">${qs.verified ?? "—"}</div></div>
    <div><div class="k">Downgraded</div><div class="v">${qs.downgraded ?? "—"}</div></div>
  </div>` : ""}
  <div class="subhead">Evaluated documents (${sr.length}) <span class="small muted">— untick to drop from the report</span></div>
  <div class="tbl-wrap"><div class="tw"><table class="tbl">
    <thead><tr><th>#</th><th>Publication</th><th style="min-width:15rem">Title</th><th class="right">Score</th><th class="right">Elements hit</th><th style="width:3.5rem">Keep</th></tr></thead>
    <tbody>${sr.map((d: any, i: number) => {
      const cr = d.checklist_results || {};
      const keys = Object.keys(cr);
      const hit = keys.filter((k) => (cr[k]?.score ?? (cr[k]?.match ? 2 : 0)) > 0).length;
      return `<tr>
        <td class="num">${i + 1}</td>
        <td class="mono tiny">${esc(d.pub_num || "—")}</td>
        <td>${esc(clip(d.title, 150))}</td>
        <td class="right num">${d.similarity_score ?? d.score ?? "—"}</td>
        <td class="right num">${hit}/${keys.length}</td>
        <td><input type="checkbox" class="keep-doc" data-i="${i}" checked></td></tr>`;
    }).join("")}</tbody></table></div></div>`;
}

/** Every limitation is editable and carries its basis (element id + verbatim quote);
 *  112(b) flags the node could not fix show as warnings; the adjudication that chose
 *  the drafting strategy is read-only context. */
function draftBody(dc: any, adj: any): string {
  const claims: any[] = dc.claims || [];
  if (!claims.length) {
    return `<div class="empty"><div class="empty-title">No draft claims</div><div>Strategy: ${esc(dc.strategy || "none")}.</div></div>`;
  }
  const av = dc.avoidance || {};
  const flags: any[] = ((dc.definiteness || {}).flags || []).filter((f: any) => !f.fixed);
  const flagBy: Record<string, string[]> = {};
  for (const f of flags) (flagBy[f.lid || ""] ||= []).push(`${f.category || f.kind || "112(b)"} (${f.rule || ""}) “${f.span || ""}”: ${f.note || ""}`);

  const head = `<div class="row" style="margin-bottom:.5rem">
    <span class="pill pill-tag">strategy ${esc(dc.strategy || "")}</span>
    ${adj?.label ? `<span class="pill pill-tag">adjudication ${esc(adj.label)}</span>` : ""}
    ${flags.length ? `<span class="pill pill-failed">${flags.length} open 112(b)</span>` : ""}
    ${av.reason ? `<span class="small muted">Avoidance: ${esc(av.reason)}</span>` : ""}
  </div>
  <div class="small muted" style="margin-bottom:.6rem">Attorney review. Each limitation shows the element and the verbatim disclosure text it rests on; ★ marks a limitation no charted reference covers.</div>`;

  return head + claims.map((c: any, ci: number) => {
    const dep = c.depends_on != null ? ` <span class="small muted">depends on claim ${esc(c.depends_on)}</span>` : "";
    const lims = (c.limitations || []).map((l: any, li: number) => {
      const basis: any[] = Array.isArray(l.basis) ? l.basis : l.basis ? [l.basis] : [];
      const fl = flagBy[l.lid || ""] || [];
      const cov = l.coverage || {};
      const star = cov.verified && !(cov.covered_by || []).length ? ` <b title="no charted reference covers this limitation">★</b>` : "";
      const basisHtml = basis.map((b: any) => `<div class="tiny mono">${esc(b.element_id || "")}</div>${b.evidence_quote ? `<div class="quote">${esc(clip(b.evidence_quote, 200))}</div>` : ""}`).join("")
        || `<span class="small muted">${esc(l.origin || "—")}</span>`;
      return `<tr${fl.length ? ' class="flagged"' : ""}>
        <td class="mono tiny">${esc(l.lid || "")}</td>
        <td><textarea class="lim" data-ci="${ci}" data-li="${li}" rows="3">${esc(l.text)}</textarea>
          ${fl.length ? `<div class="flag">⚠ ${esc(fl.join("; "))}</div>` : ""}</td>
        <td>${basisHtml}${star}</td></tr>`;
    }).join("");
    return `<div class="cand">
      <div class="cand-head"><div class="row">
        <span class="cid">Claim ${esc(c.no)}</span>
        <span class="pill pill-tag">${esc(c.form || "")}</span>${dep}
        <label class="check" style="margin-left:auto"><input type="checkbox" class="keep-claim" data-ci="${ci}" checked><span>keep</span></label>
      </div></div>
      <div class="cand-body">
        <label class="field">Preamble</label>
        <textarea class="preamble" data-ci="${ci}" rows="2">${esc(c.preamble)}</textarea>
        <div class="tbl-wrap" style="margin-top:.5rem"><div class="tw"><table class="tbl">
          <thead><tr><th style="width:4.5rem">#</th><th style="min-width:18rem">Limitation — editable</th><th style="min-width:14rem">Basis in the disclosure</th></tr></thead>
          <tbody>${lims}</tbody></table></div></div>
      </div></div>`;
  }).join("");
}

// ─── Edit collection — whole-value replacement of the gate's editable keys ───

function collectEdits(div: HTMLElement, phase: string, v: any): Record<string, unknown> {
  const edits: Record<string, unknown> = {};

  if (phase === "idca") {
    div.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(".ed[data-key]").forEach((el) => {
      const key = el.dataset.key!;
      const val = el.value.trim();
      if (val !== String(v[key] ?? "").trim()) edits[key] = val;
    });
    return edits;
  }

  if (phase === "extract" && v.extraction) {
    const ext = JSON.parse(JSON.stringify(v.extraction));
    const cands: any[] = ext.candidate_inventions || [];
    let changed = false;
    div.querySelectorAll<HTMLTextAreaElement>("textarea.el").forEach((ta) => {
      const c = cands[+ta.dataset.ci!];
      const e = c && c.elements[+ta.dataset.ei!];
      if (e && ta.value.trim() !== e.text) { e.text = ta.value.trim(); changed = true; }
    });
    const dropEl = new Set<string>();
    div.querySelectorAll<HTMLInputElement>("input.keep-el").forEach((cb) => { if (!cb.checked) dropEl.add(`${cb.dataset.ci}:${cb.dataset.ei}`); });
    const dropCand = new Set<number>();
    div.querySelectorAll<HTMLInputElement>("input.keep-cand").forEach((cb) => { if (!cb.checked) dropCand.add(+cb.dataset.ci!); });
    if (dropEl.size || dropCand.size) {
      changed = true;
      ext.candidate_inventions = cands
        .map((c: any, ci: number) => ({ ...c, elements: (c.elements || []).filter((_: any, ei: number) => !dropEl.has(`${ci}:${ei}`)) }))
        .filter((_: any, ci: number) => !dropCand.has(ci));
    }
    if (changed) edits.extraction = ext;
    return edits;
  }

  if (phase === "draft" && v.draft_claims) {
    const dc = JSON.parse(JSON.stringify(v.draft_claims));
    const claims: any[] = dc.claims || [];
    let changed = false;
    div.querySelectorAll<HTMLTextAreaElement>("textarea.lim").forEach((ta) => {
      const c = claims[+ta.dataset.ci!];
      const l = c && (c.limitations || [])[+ta.dataset.li!];
      if (l && ta.value.trim() !== l.text) { l.text = ta.value.trim(); changed = true; }
    });
    div.querySelectorAll<HTMLTextAreaElement>("textarea.preamble").forEach((ta) => {
      const c = claims[+ta.dataset.ci!];
      if (c && ta.value.trim() !== c.preamble) { c.preamble = ta.value.trim(); changed = true; }
    });
    const dropClaim = new Set<number>();
    div.querySelectorAll<HTMLInputElement>("input.keep-claim").forEach((cb) => { if (!cb.checked) dropClaim.add(+cb.dataset.ci!); });
    if (dropClaim.size) { changed = true; dc.claims = claims.filter((_: any, ci: number) => !dropClaim.has(ci)); }
    if (changed) edits.draft_claims = dc;
    return edits;
  }

  if (phase === "search" || phase === "evaluate") {
    const key = phase === "search" ? "ranked_candidates" : "scoring_report";
    const rows: any[] = v[key] || [];
    const keep = new Set<number>();
    div.querySelectorAll<HTMLInputElement>("input.keep-doc").forEach((cb) => { if (cb.checked) keep.add(+cb.dataset.i!); });
    if (keep.size !== rows.length) edits[key] = rows.filter((_: any, i: number) => keep.has(i));
  }
  return edits;
}

// ─── Prompt editors at the gate ───

async function mountPrompts(host: HTMLElement, names: string[], msg: HTMLElement): Promise<void> {
  const parts: string[] = [];
  for (const name of names) {
    try {
      const p = await getPrompt(name);
      const cur = p.current ? (p.versions.find((x) => x.v === p.current) || ({} as any)).text : p.default;
      parts.push(`<details class="box">
        <summary>Prompt <code>${esc(name)}</code> — version ${p.current}, ${p.versions.length} saved</summary>
        <div class="box-body">
          <textarea class="prompt-text" data-name="${esc(name)}" rows="14">${esc(cur || "")}</textarea>
          <div class="row" style="margin-top:.4rem">
            <button class="btn btn-sm save-prompt" data-name="${esc(name)}">Save as new version</button>
            <span class="small muted">Then <b>Rerun this phase</b> to see the new output before continuing.</span>
          </div>
        </div></details>`);
    } catch { /* prompt registry unavailable for this name */ }
  }
  host.innerHTML = `<div class="subhead">Prompts used by this phase</div>${parts.join("") || `<div class="small muted">None registered.</div>`}`;
  host.querySelectorAll<HTMLButtonElement>("button.save-prompt").forEach((b) => b.addEventListener("click", async () => {
    const name = b.dataset.name!;
    const ta = host.querySelector<HTMLTextAreaElement>(`textarea.prompt-text[data-name="${name}"]`)!;
    b.disabled = true;
    try {
      const d = await putPrompt(name, ta.value);
      msg.textContent = `Saved ${name} as version ${d.version} (now current).`;
    } catch (e: any) {
      msg.textContent = `Save failed: ${e?.message || e}`;
    } finally { b.disabled = false; }
  }));
}
