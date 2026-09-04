import { getToken } from "./auth";

export interface JobSummary {
  id: string;
  status: string;
  phase: string;
  filename: string;
  created_at?: string;
}

export interface JobStatus {
  id: string;
  status: string;
  phase: string;
  filename: string;
  phases: Record<string, any>;
  paused_at?: string;
  pause_after?: string[];
  error?: string;
  created_at?: string;
  queue_position?: number;
  queue_depth?: number;
  hitl_pending?: any;
}

export interface JobEvent {
  ts: string;
  phase: string;
  kind: string;
  message: string;
  payload?: any;
}

export interface PhaseState {
  job_id: string;
  status: string;
  phase: string;
  paused_at: string;
  pause_after: string[];
  editable: string[];
  values: Record<string, any>;
  context: Record<string, any>;
  pending_edits: string[];
  user_edits: any[];
  prompt_versions: Record<string, number>;
  phase_checkpoints: string[];
}

export interface PromptSummary {
  name: string;
  current: number;
  n_versions: number;
}

export interface PromptDetail {
  name: string;
  current: number;
  default: string;
  versions: { v: number; text: string; by?: string; ts?: string }[];
}

let devMode = false;
export function isDevMode(): boolean {
  return devMode;
}

export async function loadConfig(): Promise<{ dev: boolean }> {
  try {
    const r = await fetch("/api/config");
    if (r.ok) {
      const d = await r.json();
      devMode = !!d.dev;
      return { dev: devMode };
    }
  } catch { /* the proxy predates /api/config — fall back to auth mode */ }
  return { dev: false };
}

async function req(url: string, options: RequestInit = {}): Promise<Response> {
  const headers = new Headers(options.headers);
  if (!devMode) {
    const token = await getToken();
    if (token) headers.set("X-Firebase-Token", token);
  }
  return fetch(url, { ...options, headers });
}

async function json<T>(url: string, options: RequestInit = {}): Promise<T> {
  const r = await req(url, options);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
  return r.json() as Promise<T>;
}

async function sendJson<T>(url: string, method: string, body: unknown): Promise<T> {
  return json<T>(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ─── Jobs ───

export const listJobs = () => json<JobSummary[]>("/api/jobs");
export const getStatus = (id: string) => json<JobStatus>(`/api/status/${encodeURIComponent(id)}`);
export const getEvents = (id: string, since = 0) =>
  json<{ events: JobEvent[]; total: number; status: string; phase: string }>(
    `/api/events/${encodeURIComponent(id)}?since=${since}`,
  );

/** Stable identity of an event, for dropping the ones the backend sends twice.
 *  See `dedupeEvents` in phases.ts for why that happens. */
export function eventKey(e: JobEvent): string {
  return `${e.ts}|${e.phase}|${e.kind}|${e.message}`;
}
export const getResults = (id: string) => json<any>(`/api/results/${encodeURIComponent(id)}`);
export const deleteJob = (id: string) => req(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
export const reportUrl = (id: string) => `/api/report/${encodeURIComponent(id)}`;

export interface SubmitOptions {
  file?: File | null;
  text?: string;
  filename?: string;
  inputMode: string;
  pauseAfter: string[];
  notifyEmail: string;
}

/** Cloud Run caps a request body at 32 MB, so anything near it goes to GCS direct. */
const DIRECT_UPLOAD_LIMIT = 25 * 1024 * 1024;

/** POST /analyze — multipart. The backend takes a file only, so pasted text is
 *  sent as a .md part under the same field. Large PDFs take the signed-URL route:
 *  /upload-url → PUT to GCS → /analyze-gcs. */
export async function submitJob(
  o: SubmitOptions,
  onProgress: (msg: string) => void = () => {},
): Promise<{ job_id: string; status: string; queue_position?: number }> {
  if (o.file && o.file.size > DIRECT_UPLOAD_LIMIT) return submitViaGcs(o.file, o, onProgress);
  const fd = new FormData();
  if (o.file) {
    fd.append("file", o.file);
  } else {
    const name = o.filename || "pasted.md";
    fd.append("file", new Blob([o.text || ""], { type: "text/markdown" }), name);
  }
  if (o.notifyEmail) fd.append("notify_email", o.notifyEmail);
  if (o.pauseAfter.length) {
    fd.append("pause_after", o.pauseAfter.join(","));
    fd.append("hitl_enabled", "true");
  }
  if (o.inputMode) fd.append("input_mode", o.inputMode);
  onProgress("Uploading…");
  const r = await req("/api/analyze", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`Upload failed (${r.status}): ${(await r.text()).slice(0, 300)}`);
  const d = await r.json();
  if (!d.job_id) throw new Error(d.error || "No job_id returned");
  return d;
}

async function submitViaGcs(
  file: File, o: SubmitOptions, onProgress: (msg: string) => void,
): Promise<{ job_id: string; status: string }> {
  onProgress("Requesting an upload URL…");
  const u = await json<{ signed_url: string; gcs_uri: string; job_id: string }>(
    `/api/upload-url?filename=${encodeURIComponent(file.name)}&content_type=application/pdf`,
  );
  onProgress(`Uploading ${(file.size / 1048576).toFixed(1)} MB to GCS…`);
  const put = await fetch(u.signed_url, {
    method: "PUT", headers: { "Content-Type": "application/pdf" }, body: file,
  });
  if (!put.ok) throw new Error(`GCS upload failed (${put.status}): ${(await put.text()).slice(0, 200)}`);
  onProgress("Starting the analysis…");
  return sendJson("/api/analyze-gcs", "POST", {
    job_id: u.job_id,
    gcs_uri: u.gcs_uri,
    filename: file.name,
    notify_email: o.notifyEmail,
    pause_after: o.pauseAfter,
    hitl_enabled: o.pauseAfter.length > 0,
    input_mode: o.inputMode,
  });
}

// ─── Phase gates (HITL) ───

export const getPhaseState = (id: string) => json<PhaseState>(`/api/jobs/${encodeURIComponent(id)}/state`);
export const patchPhaseState = (id: string, edits: Record<string, unknown>) =>
  sendJson<any>(`/api/jobs/${encodeURIComponent(id)}/state`, "PATCH", edits);
export const resumeJob = (id: string, action: "continue" | "rerun_phase", comment = "") =>
  sendJson<any>(`/api/jobs/${encodeURIComponent(id)}/resume`, "POST", { action, comment });

// ─── Prompt registry ───

export const listPrompts = () => json<PromptSummary[]>("/api/prompts");
export const getPrompt = (name: string) => json<PromptDetail>(`/api/prompts/${encodeURIComponent(name)}`);
export const putPrompt = (name: string, text: string, by = "reviewer", make_current = true) =>
  sendJson<{ name: string; version: number; current: number }>(
    `/api/prompts/${encodeURIComponent(name)}`, "PUT", { text, by, make_current },
  );
export const setPromptCurrent = (name: string, version: number) =>
  sendJson<{ name: string; current: number }>(
    `/api/prompts/${encodeURIComponent(name)}/current`, "PUT", { version },
  );
