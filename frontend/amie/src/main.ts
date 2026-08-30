import "./style.css";
import { loadConfig, isDevMode } from "./api";
import { login, logout, onAuth, isDeveloper } from "./auth";
import * as theme from "./theme";
import { closeModal, esc } from "./ui";
import { renderSubmit } from "./pages/submit";
import { renderRun, disposeRun } from "./pages/run";
import { renderResults, disposeResults } from "./pages/results";
import { renderPrompts, disposePrompts } from "./pages/prompts";

export interface Page {
  render(host: HTMLElement, arg: string): void | Promise<void>;
  dispose?(): void;
}

const root = document.getElementById("app")!;

root.innerHTML = `
<div class="shell">
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand"><span class="mark">▨</span> AMIE <span class="sub">patent analyzer</span></div>
      <nav class="nav" id="nav">
        <a href="#/submit" data-route="submit">Submit</a>
        <a href="#/run" data-route="run">In progress</a>
        <a href="#/results" data-route="results">Results</a>
        <a href="#/prompts" data-route="prompts">Prompts</a>
      </nav>
      <button class="icon-btn" id="theme-btn" title="Light / dark / follow system"></button>
      <span id="auth-slot"></span>
    </div>
  </header>
  <main class="main" id="view"></main>
</div>
<div class="overlay" id="overlay"></div>
`;

const view = document.getElementById("view")!;
const nav = document.getElementById("nav")!;
const themeBtn = document.getElementById("theme-btn")!;
const authSlot = document.getElementById("auth-slot")!;

themeBtn.textContent = theme.glyph();
themeBtn.addEventListener("click", () => { themeBtn.textContent = theme.glyph(theme.cycle()); });

document.getElementById("overlay")!.addEventListener("click", (e) => {
  if ((e.target as HTMLElement).id === "overlay") closeModal();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

// ─── Routing ───

/** The job the run / results tabs default to, so those tabs are never a dead end. */
let lastJob: string = (() => {
  try { return localStorage.getItem("amie_last_job") || ""; } catch { return ""; }
})();

export function rememberJob(id: string): void {
  lastJob = id;
  try { localStorage.setItem("amie_last_job", id); } catch { /* private mode */ }
  markNav();
}

export function go(hash: string): void {
  if (location.hash === hash) route();
  else location.hash = hash;
}

const PAGES: Record<string, Page> = {
  submit: { render: renderSubmit },
  run: { render: renderRun, dispose: disposeRun },
  results: { render: renderResults, dispose: disposeResults },
  prompts: { render: renderPrompts, dispose: disposePrompts },
};

let currentPage: Page | null = null;
let currentName = "";

function parse(): { name: string; arg: string } {
  const h = location.hash.replace(/^#\/?/, "");
  const [name = "submit", arg = ""] = h.split("/");
  return { name: PAGES[name] ? name : "submit", arg: decodeURIComponent(arg) };
}

function markNav(): void {
  const { name } = parse();
  nav.querySelectorAll<HTMLAnchorElement>("a[data-route]").forEach((a) => {
    const r = a.dataset.route!;
    a.classList.toggle("on", r === name);
    if ((r === "run" || r === "results") && lastJob) a.href = `#/${r}/${lastJob}`;
    else a.href = `#/${r}`;
  });
}

function route(): void {
  const { name, arg } = parse();
  currentPage?.dispose?.();
  closeModal();
  currentName = name;
  currentPage = PAGES[name];
  markNav();
  view.innerHTML = "";
  void currentPage.render(view, arg || (name === "run" || name === "results" ? lastJob : ""));
}

window.addEventListener("hashchange", route);

// ─── Auth gate ───
//
// BACKEND_ENV=dev (the express proxy reports it at /api/config) runs with the
// backend's AUTH_DISABLED, so there is nothing to sign in to.

async function boot(): Promise<void> {
  await loadConfig();
  if (isDevMode()) {
    authSlot.innerHTML = `<span class="pill pill-tag" title="BACKEND_ENV=dev — auth disabled">dev</span>`;
    route();
    return;
  }
  authSlot.innerHTML = `<span class="small muted">…</span>`;
  onAuth(async (user) => {
    authSlot.innerHTML = "";
    if (!user) {
      const b = document.createElement("button");
      b.className = "btn btn-sm";
      b.textContent = "Sign in";
      b.addEventListener("click", async () => {
        try { await login(); } catch (e: any) { alert(e?.message || "Login failed"); }
      });
      authSlot.appendChild(b);
      view.innerHTML = `<div class="empty" style="margin-top:3rem">
        <div class="empty-title">Sign in to continue</div>
        <div>An @oregonstate.edu Google account is required.</div></div>`;
      return;
    }
    const dev = await isDeveloper();
    authSlot.innerHTML = `<span class="small muted nowrap">${esc(user.email || "")}${dev ? " · dev" : ""}</span>`;
    const out = document.createElement("button");
    out.className = "icon-btn";
    out.textContent = "Sign out";
    out.addEventListener("click", () => logout());
    authSlot.appendChild(out);
    route();
  });
}

void boot();
