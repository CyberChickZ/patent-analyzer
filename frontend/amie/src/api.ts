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

// ─── Ledger & quota ───

export interface LedgerRow {
  phase: string;
  kind: "model" | "embedding" | "bigquery" | "external";
  name: string;
  calls: number;
  input_tokens?: number;
  output_tokens?: number;
  thought_tokens?: number;
  gib_billed?: number;
  seconds: number;
  cost_usd: number;
  retries: number;
  failures: number;
  degradations: number;
  note?: string;
}

export interface Ledger {
  job_id: string;
  generated_at: string;
  totals: {
    cost_usd: number; seconds: number; llm_calls: number; external_calls: number;
    failures: number; degradations: number; retries: number;
  };
  most_expensive: { phase: string; cost_usd: number; share_of_total: number; seconds: number; driver: string } | null;
  rows: LedgerRow[];
  incidents: { phase: string; source: string; kind: string; detail: string }[];
  prices: Record<string, any>;
  caveats: string[];
}

export const getLedger = (id: string) =>
  json<{ job_id: string; ledger: Ledger }>(`/api/jobs/${encodeURIComponent(id)}/usage`);

/** One external source's headroom. Nulls are real: a source with no counter
 *  (a rate limit rather than an allowance) reports none rather than a zero. */
export interface QuotaRow {
  source: string;
  name: string;
  used: number | null;
  cap: number | null;
  remaining: number | null;
  unit: string;
  period: "month" | "week" | "minute" | "none";
  resets_at: string | null;
  resets_in_hours: number | null;
  exhausted: boolean;
  limits: string[];
  note: string;
  error: string;
  expires_on: string | null;
  expires_in_days: number | null;
}

export interface RateCard {
  source: string;
  review_by: string;
  models: { model: string; input_usd_per_mtok: number; output_usd_per_mtok: number; note?: string }[];
  embedding_usd_per_mtok: number;
  bigquery_usd_per_tib: number;
  /** Scheduled rate changes still ahead — the global model's price doubles on
   *  2027-01-01, which is exactly the kind of fact a hard-coded card misses. */
  upcoming_changes: {
    model: string; effective_from: string;
    input_usd_per_mtok: number; output_usd_per_mtok: number;
    multiple: number | null; from_input: number; from_output: number;
  }[];
  note: string;
}

export interface QuotaSnapshot {
  generated_at: string;
  month: string;
  week: string;
  sources: QuotaRow[];
  prices: RateCard;
  exhausted: string[];
  expiring_soon: string[];
  note: string;
}

export const getQuota = () => json<QuotaSnapshot>("/api/quota");

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

// ─── Missing full text ───
//
// What the deep read never saw, and the slot for the reviewer's own copy.
// Nothing here fetches a paper: OSU Libraries' Responsible Use policy forbids
// programmatic downloading of licensed content, so the backend lists what it
// could not reach and a person supplies the PDF (backend/patent_analyzer/
// fulltext.py carries the policy text).

export interface FulltextAttempt {
  tier: "bigquery_claims" | "arxiv" | "oa" | "pdf_download" | string;
  outcome: "ok" | "missed" | "failed" | "skipped" | "unknown";
  detail: string;
}

export interface FulltextUpload {
  ref_id: string;
  filename: string;
  bytes: number;
  doi: string;
  gcs_uri: string;
  uploaded_at: string;
  reread: boolean;
}

export interface FulltextGapRow {
  ref_id: string;
  pub_num: string;
  title: string;
  match_type: string;
  doi: string;
  landing_page: string;
  fulltext_tier: string;
  read_state: "full_text" | "abstract_only" | "nothing";
  read_reason: string;
  text_chars: number;
  similarity_score: number;
  attempts: FulltextAttempt[];
  upload: FulltextUpload | null;
}

export interface FulltextGaps {
  job_id: string;
  status: string;
  rows: FulltextGapRow[];
  summary: {
    evaluated: number; missing: number; abstract_only: number; nothing: number;
    uploaded: number; pending_reread: number; full_text: number;
  };
  policy_note: string;
  policy_url: string;
  rerun_history: {
    at: string; refs: string[]; read: string[]; failed: string[];
    changed?: boolean; label_before?: string; label_after?: string;
    determination_before: string; determination_after: string;
  }[];
}

export const getFulltextGaps = (id: string) =>
  json<FulltextGaps>(`/api/jobs/${encodeURIComponent(id)}/fulltext-gaps`);

/** One reference's PDF. Multipart under the direct-upload limit, signed URL
 *  above it — the same split `submitJob` makes for the job's own input. */
export async function uploadFulltext(
  id: string, refId: string, file: File,
): Promise<{ upload: FulltextUpload; pending_reread: number }> {
  const base = `/api/jobs/${encodeURIComponent(id)}/fulltext/${encodeURIComponent(refId)}`;
  if (file.size > DIRECT_UPLOAD_LIMIT) {
    const u = await json<{ signed_url: string; gcs_uri: string }>(
      `/api/upload-url?filename=${encodeURIComponent(file.name)}&content_type=application/pdf`,
    );
    const put = await fetch(u.signed_url, {
      method: "PUT", headers: { "Content-Type": "application/pdf" }, body: file,
    });
    if (!put.ok) throw new Error(`GCS upload failed (${put.status})`);
    return sendJson(`${base}/from-gcs`, "POST", { gcs_uri: u.gcs_uri, filename: file.name });
  }
  const fd = new FormData();
  fd.append("file", file);
  const r = await req(`${base}/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
  return r.json();
}

export const dropFulltextUpload = (id: string, refId: string) =>
  json<any>(`/api/jobs/${encodeURIComponent(id)}/fulltext/${encodeURIComponent(refId)}`, { method: "DELETE" });

/** Deep-read the uploaded PDFs, re-adjudicate, rewrite the report. Returns as
 *  soon as the job is queued; progress shows up on the run page's event feed. */
export const rerunEvidence = (id: string, refs?: string[]) =>
  sendJson<{ job_id: string; status: string; refs: string[] }>(
    `/api/jobs/${encodeURIComponent(id)}/rerun-evidence`, "POST", refs ? { refs } : {});
