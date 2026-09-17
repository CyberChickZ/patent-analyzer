/** Rendering helpers shared by every page. */

/** The UI is written in English, so its dates and numbers are formatted in
 *  English too. Passing [] takes the browser's locale instead, which is how a
 *  Chinese date turned up on the Quota page of an otherwise English page
 *  (Harry, 2026-09-20) — the strings were never translated, the formatter was
 *  just following whoever was looking at it. */
export const LOCALE = "en-US";

export function esc(s: unknown): string {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!
  ));
}

export function clip(s: unknown, n: number): string {
  const t = String(s ?? "");
  return t.length > n ? t.slice(0, n - 1) + "…" : t;
}

/** Backend statuses → the five pill states the UI speaks. */
export const PILL: Record<string, string> = {
  queued: "queued",
  pending: "pending",
  running: "running",
  waiting_for_hitl: "paused",
  paused: "paused",
  completed: "completed",
  error: "failed",
  failed: "failed",
};

export function pill(status: string, label?: string): string {
  const k = PILL[status] || "pending";
  return `<span class="pill pill-${k}">${esc(label ?? k)}</span>`;
}

export function tag(text: string): string {
  return `<span class="pill pill-tag">${esc(text)}</span>`;
}

export function fmtTime(ts: string): string {
  const d = new Date(ts);
  // 24h so the column stays one line at every width
  return isNaN(d.getTime()) ? "" : d.toLocaleTimeString(LOCALE, { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** Titles come straight off the document, markdown heading marks and all. */
export function plainTitle(s: unknown): string {
  return String(s ?? "").replace(/^\s*#+\s*/, "").replace(/\s*#+\s*$/, "").trim();
}

export function fmtDate(ts?: string): string {
  if (!ts) return "—";
  const d = new Date(ts);
  return isNaN(d.getTime()) ? "—" : d.toLocaleString(LOCALE, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/** A calendar day, no clock, in UTC.
 *
 *  Used where the day itself is the fact — a plan's expiry, or the day of the
 *  month a counter rolls over on. UTC because that is the day the provider
 *  means: a bare `"2026-10-02"` parses as UTC midnight, and rendering it in
 *  a US timezone prints the 1st, which is a different answer to "when does
 *  this expire" than the one the backend gave. */
export function fmtDay(ts?: string | null): string {
  if (!ts) return "—";
  const d = new Date(ts);
  return isNaN(d.getTime())
    ? "—"
    : d.toLocaleDateString(LOCALE, { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}

export function num(n: unknown, fallback = "—"): string {
  if (n === null || n === undefined || n === "") return fallback;
  const v = Number(n);
  return isNaN(v) ? String(n) : v.toLocaleString(LOCALE);
}

export function pct(n: unknown, digits = 0): string {
  const v = Number(n);
  return isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;
}

export function empty(title: string, body: string): string {
  return `<div class="empty"><div class="empty-title">${esc(title)}</div><div>${body}</div></div>`;
}

export function errorBox(msg: string): string {
  return `<div class="notice notice-error">${esc(msg)}</div>`;
}

/** Light markdown for LLM prose: bold/italic/code/headings/lists. */
export function md(text: string): string {
  let h = esc(text);
  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
  h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  h = h.replace(/^### (.+)$/gm, "<h4>$1</h4>");
  h = h.replace(/^## (.+)$/gm, "<h3>$1</h3>");
  h = h.replace(/^# (.+)$/gm, "<h3>$1</h3>");
  h = h.replace(/^\s*[-*] (.+)$/gm, "<li>$1</li>");
  h = h.replace(/^\s*\d+\. (.+)$/gm, "<li>$1</li>");
  h = h.replace(/(<li>[\s\S]*?<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);
  return h.split(/\n\n+/).map((p) => (p.trim().startsWith("<") ? p : `<p>${p.replace(/\n/g, "<br>")}</p>`)).join("");
}

/** A prompt body split on the ════ SECTION ════ markers the backend emits. */
export function promptBlocks(text: string): string {
  const parts = text.split(/════ ([^═]+) ════/g);
  if (parts.length < 3) return `<pre>${esc(text)}</pre>`;
  let html = parts[0].trim() ? `<pre>${esc(parts[0].trim())}</pre>` : "";
  for (let i = 1; i < parts.length; i += 2) {
    const label = parts[i].trim();
    const isTemplate = /template/i.test(label);
    html += `<div class="pblock ${isTemplate ? "pb-template" : "pb-data"}">
      <div class="pb-label">${esc(label)} · ${isTemplate ? "hardcoded" : "data"}</div>
      <pre>${esc((parts[i + 1] || "").trim())}</pre></div>`;
  }
  return html;
}

// ─── Modal ───

export function openModal(title: string, bodyHtml: string): void {
  const overlay = document.getElementById("overlay")!;
  overlay.innerHTML = `<div class="modal" role="dialog" aria-modal="true" aria-label="${esc(title)}">
    <div class="modal-head"><h3 style="flex:1">${esc(title)}</h3>
      <button class="icon-btn" id="modal-close" aria-label="Close">✕</button></div>
    <div class="modal-body">${bodyHtml}</div></div>`;
  overlay.classList.add("open");
  document.getElementById("modal-close")!.addEventListener("click", closeModal);
}

export function closeModal(): void {
  const overlay = document.getElementById("overlay")!;
  overlay.classList.remove("open");
  overlay.innerHTML = "";
}

/** Delegated click: one listener, matched by CSS selector. */
export function on(root: ParentNode, selector: string, handler: (el: HTMLElement, ev: Event) => void): void {
  root.querySelectorAll<HTMLElement>(selector).forEach((el) => {
    el.addEventListener("click", (ev) => handler(el, ev));
  });
}
