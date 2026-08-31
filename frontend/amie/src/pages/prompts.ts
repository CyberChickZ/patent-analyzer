import { listPrompts, getPrompt, putPrompt, setPromptCurrent, type PromptSummary, type PromptDetail } from "../api";
import { esc, empty, errorBox, fmtDate, on } from "../ui";

let list: PromptSummary[] = [];
let selected = "";
let detail: PromptDetail | null = null;
let viewing = -1;   // version being shown; -1 = current (or default when none saved)

export function disposePrompts(): void {
  detail = null; viewing = -1;
}

export async function renderPrompts(host: HTMLElement, arg: string): Promise<void> {
  host.innerHTML = `
    <div class="page-head">
      <div class="kicker">Registry</div>
      <h1>Prompts</h1>
      <div class="lede">Every prompt the pipeline uses, its saved versions, and which one is current. Saving makes a new version; the pipeline picks up the current one on the next phase it runs.</div>
    </div>
    <div id="prompt-error"></div>
    <section class="section">
      <div id="prompt-list"><div class="small muted"><span class="spinner"></span> Loading…</div></div>
    </section>
    <section class="section" id="prompt-detail"></section>`;

  try {
    list = await listPrompts();
  } catch (e: any) {
    document.getElementById("prompt-error")!.innerHTML = errorBox(`Could not load the registry — ${e?.message || e}`);
    document.getElementById("prompt-list")!.innerHTML = "";
    return;
  }
  paintList();
  if (arg) void select(arg);
  else if (selected) void select(selected);
}

function paintList(): void {
  const host = document.getElementById("prompt-list");
  if (!host) return;
  if (!list.length) {
    host.innerHTML = empty("No prompts registered", "The backend registers its templates on import; none came back.");
    return;
  }
  const byGroup: Record<string, PromptSummary[]> = {};
  for (const p of list) (byGroup[p.name.split(".")[0]] ||= []).push(p);
  host.innerHTML = `<div class="tbl-wrap"><div class="tw"><table class="tbl rows-clickable">
    <thead><tr><th>Phase</th><th>Prompt</th><th class="right">Current</th><th class="right">Versions</th><th></th></tr></thead>
    <tbody>${Object.entries(byGroup).map(([g, ps]) => ps.map((p, i) => `<tr data-name="${esc(p.name)}" class="${p.name === selected ? "on" : ""}">
      ${i === 0 ? `<td rowspan="${ps.length}" class="mono tiny">${esc(g)}</td>` : ""}
      <td class="mono">${esc(p.name.split(".").slice(1).join("."))}</td>
      <td class="right num">${p.current ? `v${p.current}` : `<span class="muted">default</span>`}</td>
      <td class="right num">${p.n_versions}</td>
      <td class="right"><button class="icon-btn open" data-name="${esc(p.name)}">${p.name === selected ? "Editing" : "Open"}</button></td>
    </tr>`).join("")).join("")}</tbody>
  </table></div></div>`;
  on(host, "tr[data-name], button.open", (el) => void select(el.dataset.name!));
}

async function select(name: string): Promise<void> {
  selected = name;
  viewing = -1;
  const host = document.getElementById("prompt-detail");
  if (!host) return;
  host.innerHTML = `<div class="small muted"><span class="spinner"></span> Loading ${esc(name)}…</div>`;
  try {
    detail = await getPrompt(name);
  } catch (e: any) {
    host.innerHTML = errorBox(`${name} — ${e?.message || e}`);
    return;
  }
  paintList();
  paintDetail();
  host.scrollIntoView({ behavior: "smooth", block: "start" });
}

function textOf(d: PromptDetail, v: number): string {
  if (v <= 0) return d.default;
  return (d.versions.find((x) => x.v === v) || ({} as any)).text || d.default;
}

function paintDetail(): void {
  const host = document.getElementById("prompt-detail");
  if (!host || !detail) return;
  const d = detail;
  const shown = viewing >= 0 ? viewing : d.current;
  host.innerHTML = `
    <header><h2 class="mono">${esc(d.name)}</h2>
      <span class="hint">${d.versions.length} saved version${d.versions.length === 1 ? "" : "s"} · current ${d.current ? `v${d.current}` : "default (built in)"}</span></header>
    <div class="panel"><div class="panel-body stack">
      <div class="tbl-wrap"><div class="tw"><table class="tbl">
        <thead><tr><th>Version</th><th>Saved by</th><th>When</th><th class="right">Chars</th><th></th></tr></thead>
        <tbody>
          <tr class="${shown === 0 ? "on" : ""}"><td>default</td><td class="muted">built in</td><td class="muted">—</td>
            <td class="right num">${d.default.length}</td>
            <td class="right"><button class="icon-btn view" data-v="0">${shown === 0 ? "Showing" : "View"}</button></td></tr>
          ${d.versions.map((v) => `<tr>
            <td>v${v.v} ${v.v === d.current ? `<span class="pill pill-completed">current</span>` : ""}</td>
            <td>${esc(v.by || "—")}</td><td class="small muted">${esc(fmtDate(v.ts))}</td>
            <td class="right num">${(v.text || "").length}</td>
            <td class="right nowrap">
              <button class="icon-btn view" data-v="${v.v}">${shown === v.v ? "Showing" : "View"}</button>
              ${v.v === d.current ? "" : `<button class="icon-btn mkcur" data-v="${v.v}">Set current</button>`}
            </td></tr>`).join("")}
        </tbody></table></div></div>

      <div>
        <label class="field" for="prompt-text">Editing ${shown ? `v${shown}` : "the built-in default"} — saving creates a new version and makes it current</label>
        <textarea id="prompt-text" rows="20">${esc(textOf(d, shown))}</textarea>
      </div>
      <div class="row">
        <button class="btn" id="save-prompt">Save as new version</button>
        <button class="btn btn-ghost" id="revert-prompt">Revert to the built-in default</button>
        <span class="small muted" id="prompt-msg"></span>
      </div>
    </div></div>`;

  on(host, "button.view", (el) => { viewing = +el.dataset.v!; paintDetail(); });
  on(host, "button.mkcur", async (el) => {
    const v = +el.dataset.v!;
    try {
      await setPromptCurrent(d.name, v);
      await select(d.name);
      msg(`v${v} is now current.`);
    } catch (e: any) { msg(`Failed: ${e?.message || e}`); }
  });
  document.getElementById("revert-prompt")!.addEventListener("click", () => {
    (document.getElementById("prompt-text") as HTMLTextAreaElement).value = d.default;
    msg("Loaded the built-in default into the editor — save it to make it a version.");
  });
  document.getElementById("save-prompt")!.addEventListener("click", async (ev) => {
    const b = ev.currentTarget as HTMLButtonElement;
    const ta = document.getElementById("prompt-text") as HTMLTextAreaElement;
    if (!ta.value.trim()) { msg("Empty prompts are rejected by the backend."); return; }
    b.disabled = true;
    try {
      const r = await putPrompt(d.name, ta.value);
      list = await listPrompts();
      await select(d.name);
      msg(`Saved as v${r.version} (now current).`);
    } catch (e: any) {
      msg(`Save failed: ${e?.message || e}`);
    } finally { b.disabled = false; }
  });
}

function msg(text: string): void {
  const el = document.getElementById("prompt-msg");
  if (el) el.textContent = text;
}
