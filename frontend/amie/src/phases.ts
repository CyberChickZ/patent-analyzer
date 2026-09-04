import { eventKey, type JobEvent, type JobStatus } from "./api";

/** Drop events the page has already seen.
 *
 *  Two things produce repeats and neither is fixable from here:
 *
 *  1. The backend. `graph/extraction_subgraph.py` declares its own
 *     `events: Annotated[list, operator.add]`, so the compiled subgraph is
 *     handed the parent's accumulated events on entry and echoes them back in
 *     its update patch; `app/main.py` appends that patch wholesale, and every
 *     phase-1 event lands in `job["events"]` a second time. Verified against
 *     the persisted records of jobs 96eada38 / ff3c54ad / ba03cfe3: 8 phase-1
 *     entries, four of them byte-identical duplicates down to the timestamp.
 *  2. This page. `/events/{job}` is a `since`-indexed poll, not a stream, so
 *     two polls that overlap both ask from the same index and both get the
 *     same tail. The in-flight guard in pages/run.ts closes that one.
 *
 *  Identity is (ts, phase, kind, message): the backend's timestamps are
 *  microsecond ISO strings minted per event, so two genuinely distinct events
 *  never collide, while a re-sent copy is exact. */
export function dedupeEvents(seen: Set<string>, incoming: JobEvent[]): JobEvent[] {
  const out: JobEvent[] = [];
  for (const e of incoming) {
    const k = eventKey(e);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(e);
  }
  return out;
}

/** Events from code that no longer exists, kept out of the feed.
 *
 *  Saved job records are replayed by /events and /status long after the node
 *  that wrote them is gone, so the log of an old job otherwise advertises a
 *  stage the pipeline no longer has. Personas went with the legacy pipeline in
 *  d834d6a; nothing in backend/ emits this any more (`grep -r personas
 *  --include=*.py` finds only a comment), but three of the jobs still on disk
 *  carry it, and job records outlive deletions.
 *
 *  Anything listed here is dropped on arrival, not greyed out: a reviewer
 *  reading the timeline should see what this pipeline did, and a stage that no
 *  longer runs is noise whichever way it is styled. */
const LEGACY_EVENTS: RegExp[] = [
  /^Crafted \d+ domain-specific personas$/,
];

export function isLegacyEvent(e: JobEvent): boolean {
  return LEGACY_EVENTS.some((re) => re.test(e.message || ""));
}

/** The six components of architecture v2 §2, and the backend phase keys each
 *  one absorbs. The backend still emits phase1 · phase2 · phase3 · phase4 ·
 *  phase4b · phase5 plus the gate names (`idca` / `extract` / `search` /
 *  `evaluate` / `draft`, which a gate stamps on the reviewer's own edit
 *  events); those keys are load-bearing for checkpoints, so they stay as they
 *  are and only the labels follow the architecture.
 *
 *  Two changes from the first cut:
 *  — Semantic Ranking is no longer a step. §3 folds ranking into the
 *    prior-art loop's `judge` stage; it was never a phase of its own on the
 *    wire either (its prune events are emitted under phase3), so a step that
 *    could only ever light up at the very end of another one was misleading.
 *  — Draft is a step. `nodes/draft.py` emits under phase4b and has its own
 *    gate, so "Pause after → Draft" pointed at a step the timeline did not
 *    draw. §2 makes it a component in its own right (Claim Set Designer, second
 *    pass), which is what it should have been. */
export interface Step {
  key: string;
  label: string;
  blurb: string;
  phases: string[];
}

export const STEPS: Step[] = [
  { key: "read", label: "Read", blurb: "Document to text layer: detect whether an invention is present, and classify it", phases: ["phase1", "idca"] },
  { key: "claimset", label: "Claim set", blurb: "Candidate inventions and their elements, each pinned to verbatim text", phases: ["phase2", "extract"] },
  { key: "loop", label: "Prior-art loop", blurb: "Query, judge, expand from what was found — ranking included — until coverage or budget stops it", phases: ["phase3", "search"] },
  { key: "evidence", label: "Evidence & verdict", blurb: "Read each reference against the elements, chart the evidence, adjudicate", phases: ["phase4", "evaluate"] },
  { key: "draft", label: "Draft", blurb: "Claims written from the elements and narrowed against what the art covers", phases: ["phase4b", "draft"] },
  { key: "report", label: "Report", blurb: "Compile the report and notify", phases: ["phase5", "report"] },
];

export function stepOfEvent(e: JobEvent): string {
  const s = STEPS.find((x) => x.phases.includes(e.phase));
  return s ? s.key : "loop";
}

/** Gate name → the step it pauses after. */
export const PAUSE_STEP: Record<string, string> = {
  idca: "read", extract: "claimset", search: "loop", evaluate: "evidence", draft: "draft",
};

/** The gate keys go to the backend unchanged; only what the reviewer reads
 *  follows the architecture's vocabulary. */
export const PAUSE_LABEL: Record<string, string> = {
  idca: "Read", extract: "Claim set", search: "Prior-art loop", evaluate: "Evidence & verdict", draft: "Draft",
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
