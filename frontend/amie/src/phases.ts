import type { JobEvent, JobStatus } from "./api";

/** The six steps the UI shows, and the backend phase keys each one absorbs.
 *  The backend emits phase1 · phase2 · phase3 · phase4 · phase4b · phase5 plus the
 *  gate names `extract` / `draft`; there is no phase3b event stream, so semantic
 *  ranking is recognised by the prune events inside phase3. */
export interface Step {
  key: string;
  label: string;
  blurb: string;
  phases: string[];
}

export const STEPS: Step[] = [
  { key: "detect", label: "Invention Detection", blurb: "Read the document, decide whether an invention is present, classify it", phases: ["phase1", "idca"] },
  { key: "decompose", label: "Decomposition", blurb: "Candidate inventions, their elements and the checklist", phases: ["phase2", "extract"] },
  { key: "search", label: "Prior Art Search", blurb: "Query planning and the parallel recall channels", phases: ["phase3"] },
  { key: "rank", label: "Semantic Ranking", blurb: "Prune the pool down to what is worth reading", phases: ["phase3b"] },
  { key: "evaluate", label: "Deep Evaluation", blurb: "Read each document against the checklist, score, adjudicate, draft claims", phases: ["phase4", "phase4b", "evaluate", "draft"] },
  { key: "report", label: "Report", blurb: "Compile the report and notify", phases: ["phase5"] },
];

const RANK_KINDS = new Set(["prune_done"]);

export function stepOfEvent(e: JobEvent): string {
  if (e.phase === "phase3" && RANK_KINDS.has(e.kind)) return "rank";
  const s = STEPS.find((x) => x.phases.includes(e.phase));
  return s ? s.key : "search";
}

/** Gate name → the step it pauses after. */
export const PAUSE_STEP: Record<string, string> = {
  idca: "detect", extract: "decompose", search: "rank", evaluate: "evaluate", draft: "evaluate",
};

export const PAUSE_LABEL: Record<string, string> = {
  idca: "Detection", extract: "Decomposition", search: "Search", evaluate: "Evaluation", draft: "Draft claims",
};

export const PAUSE_ORDER = ["idca", "extract", "search", "evaluate", "draft"];

/** Prompts the reviewer may edit at each gate (names from GET /api/prompts). */
export const PHASE_PROMPTS: Record<string, string[]> = {
  idca: ["idca.summarize", "idca.docjson"],
  extract: ["extract.candidates", "extract.elements"],
  search: ["search.facets", "search.react_step"],
  evaluate: ["evaluate.document_text", "evaluate.document_pdf"],
  draft: ["draft.claims", "draft.reword", "draft.definiteness"],
};

export type StepState = "pending" | "running" | "completed" | "paused" | "failed";

/** Per-step status, derived from the job record and the event stream together —
 *  job.phases alone lags behind under LangGraph. */
export function stepStates(job: JobStatus, events: JobEvent[]): Record<string, StepState> {
  const seen = new Set(events.map(stepOfEvent));
  const done = new Set<string>();

  for (const k of Object.keys(job.phases || {})) {
    const s = STEPS.find((x) => x.phases.includes(k));
    if (s) done.add(s.key);
  }
  // any step before one that has produced events has finished
  let lastSeen = -1;
  STEPS.forEach((s, i) => { if (seen.has(s.key)) lastSeen = i; });
  for (let i = 0; i < lastSeen; i++) done.add(STEPS[i].key);

  const paused = job.status === "waiting_for_hitl";
  const pausedStep = paused ? PAUSE_STEP[job.paused_at || ""] || "" : "";
  const out: Record<string, StepState> = {};

  for (const s of STEPS) {
    let st: StepState = "pending";
    if (job.status === "completed") st = "completed";
    else if (done.has(s.key)) st = "completed";
    else if (seen.has(s.key)) st = paused ? "completed" : "running";
    if (job.status === "error" && !done.has(s.key) && seen.has(s.key)) st = "failed";
    out[s.key] = st;
  }
  if (pausedStep && out[pausedStep] !== "failed") out[pausedStep] = "paused";
  // a paused job has finished everything up to and including the gate's step
  if (pausedStep) {
    const idx = STEPS.findIndex((s) => s.key === pausedStep);
    for (let i = 0; i < idx; i++) out[STEPS[i].key] = "completed";
  }
  return out;
}

/** LLM calls the event stream actually shows (kind `llm`), by step. */
export function llmCallsByStep(events: JobEvent[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const e of events) {
    if (e.kind === "llm" || e.kind === "llm_call") {
      const k = stepOfEvent(e);
      out[k] = (out[k] || 0) + 1;
    }
  }
  return out;
}
