"""
Patent Analyzer — FastAPI backend with A2A protocol support.

LLM: Google GenAI (Vertex AI) — Gemini
Search: SerpAPI + OpenAlex + sentence-transformers
Protocol: A2A JSON-RPC for agent-to-agent integration

Deploy: gcloud run deploy (see deploy/)
"""

import asyncio
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile


def _load_local_env() -> None:
    """Local dev (BACKEND_ENV=dev): read backend/.env.yaml so the process sees
    the same SERPAPI_KEYS etc. Cloud Run gets the file via --env-vars-file.
    Variables already in the environment win."""
    if os.environ.get("K_SERVICE"):
        return
    f = Path(__file__).parent.parent / ".env.yaml"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        m = re.match(r'^([A-Z][A-Z0-9_]*):\s*"([^"]*)"', line)
        if m and not os.environ.get(m.group(1)):
            os.environ[m.group(1)] = m.group(2)


_load_local_env()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.auth import is_developer, optional_auth, require_auth, set_developer
from patent_analyzer.funnel import slim_event, slim_search_stats

app = FastAPI(title="Patent Analyzer", version="0.4.0")

from app.hitl import router as hitl_router
app.include_router(hitl_router, prefix="/api")

from app.fulltext_uploads import router as fulltext_router
app.include_router(fulltext_router, prefix="/api")

from app.diagnostics import router as diagnostics_router
app.include_router(diagnostics_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_BASE = Path(os.getenv("OUTPUT_DIR", "/tmp/outputs"))
OUTPUT_BASE.mkdir(exist_ok=True)

GCS_BUCKET = os.getenv("GCS_BUCKET", "aime-hello-world-amie-uswest1")
GCS_PREFIX = os.getenv("GCS_PREFIX", "patent-analyzer/jobs/")

# StaticFiles raises at import time when the directory is missing, and
# `app/static/` is empty — git cannot store an empty directory, so it exists in
# a working tree that has ever had it and in no fresh checkout at all. Every
# deploy so far uploaded the developer's working tree and therefore carried it
# along by accident; the first deploy made from `git archive HEAD` died on
# startup with `RuntimeError: Directory '/app/app/static' does not exist`
# (revision patent-analyzer-00080-9c4, 2026-09-19). The repository could not
# build itself from its own contents, and nothing said so until the upload
# stopped including a file nobody had committed.
_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Job store — in-memory cache + GCS persistence
jobs: dict[str, dict] = {}
# Track which jobs are actively running on THIS instance
_active_pipelines: set[str] = set()

# Serial pipeline queue — jobs are processed one at a time
_pipeline_queue: asyncio.Queue[str] = asyncio.Queue()
_pending_jobs: list[str] = []  # ordered list for queue position lookups


def _queue_position(job_id: str) -> int | None:
    """Return 0-based position in queue, or None if not queued."""
    try:
        return _pending_jobs.index(job_id)
    except ValueError:
        return None


def _enqueue_job(job_id: str):
    """Add a job to the serial pipeline queue."""
    _pending_jobs.append(job_id)
    _pipeline_queue.put_nowait(job_id)


USE_LANGGRAPH = True  # Always use LangGraph pipeline

# How often the running pipeline refreshes job["last_heartbeat"], independently
# of node boundaries. Must stay well under the 900 s staleness threshold that
# /status and _reap_zombie_jobs use to declare a job dead.
HEARTBEAT_S = float(os.getenv("HEARTBEAT_S", "60"))


_checkpointer_singleton = None
PHASE_NODE = {"idca": "idca", "extract": "ssr", "search": "search", "evaluate": "evaluate", "draft": "draft"}
PHASE_OF_NODE = {v: k for k, v in PHASE_NODE.items()}
_SNAPSHOT_KEYS = ("summary", "checklist", "delegation", "innovation_axes", "technology_choices", "applicable_types",
                  "cpc_subclass", "fields_map", "source_title", "source_arxiv_id", "source_doi", "status_determination",
                  "doc_type", "input_mode", "input_local_path", "notify_email", "evolve", "hitl_enabled",
                  "phase_results", "extraction", "document_text", "ranked_candidates", "search_stats", "scoring_report",
                  "eval_stats", "pause_after", "user_edits", "prompt_versions", "category", "publication_date",
                  "adjudication", "draft_claims", "doc_json", "doc_json_stats")


def _checkpointer():
    """Process-wide checkpointer. CHECKPOINT_BACKEND=memory (default) keeps
    threads for the life of the instance; sqlite / gcs (later steps) survive
    restarts. The job record keeps a whitelist snapshot as the fallback."""
    global _checkpointer_singleton
    if _checkpointer_singleton is None:
        from patent_analyzer.checkpoint_store import make_checkpointer
        _checkpointer_singleton = make_checkpointer()
    return _checkpointer_singleton


def _pause_after(job: dict) -> list[str]:
    pa = job.get("pause_after") or []
    if isinstance(pa, str):
        pa = [x.strip() for x in pa.split(",") if x.strip()]
    if job.get("hitl_enabled") and not pa:
        pa = ["extract"]          # legacy toggle = pause after the checklist / extraction
    return [x for x in pa if x in PHASE_NODE]


def _refresh_accounting(job: dict) -> None:
    """Fold what this process has metered so far onto the job record.

    Runs at every point the job is persisted, because these two blocks are the
    only per-job view of a meter that is not per-job: app.llm.usage counts for
    the life of the process, and patent_analyzer.metering turns it into
    per-phase deltas by snapshotting it at each phase boundary. Merged rather
    than replaced — after a HITL resume on a restarted server, the early phases
    exist only on the record.
    """
    from patent_analyzer import metering
    job["phase_metrics"] = {**(job.get("phase_metrics") or {}), **metering.phases()}
    job["cost"] = metering.report(job["phase_metrics"])
    job["ledger"] = metering.ledger(job["phase_metrics"])


def _record_phase_failure(job: dict, graph, config: dict, exc: BaseException) -> None:
    """Turn a raised node into a job the user can see and act on.

    LangGraph leaves the thread positioned *at* the node that raised: after the
    exception, `get_state(config).next == (failed_node,)` and the checkpoint is
    the one taken before it ran, so re-invoking the same thread with input None
    retries exactly that node (verified on langgraph 0.2.76). So the failure is
    recorded with the phase name, the message, the traceback tail, and the
    checkpoint id — which is what `POST /api/jobs/{id}/resume
    {"action":"rerun_phase"}` needs to be usable on a failed job.
    """
    import traceback
    tb = traceback.format_exc()
    job_id = job.get("id", "?")
    print(f"[LANGGRAPH] job {job_id} failed: {type(exc).__name__}: {exc}\n{tb}")
    failed_node, values, checkpoint_id = "", {}, ""
    try:
        st = graph.get_state(config)
        failed_node = (st.next or ("",))[0] or ""
        values = st.values or {}
        checkpoint_id = (st.config or {}).get("configurable", {}).get("checkpoint_id", "")
    except Exception as e:                       # state unreadable: still fail loudly
        print(f"[LANGGRAPH] job {job_id}: could not read graph state after failure: {e}")
    phase = PHASE_OF_NODE.get(failed_node, "")
    job["status"] = "error"
    job["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    job["error_trace"] = tb[-4000:]
    job["failed_node"] = failed_node
    job["failed_phase"] = phase
    job["paused_at"] = ""
    job["hitl_pending"] = None
    if values:
        job["_hitl_saved_state"] = {k: values[k] for k in _SNAPSHOT_KEYS if k in values}
    if phase and checkpoint_id:
        job.setdefault("phase_checkpoints", {})[phase] = checkpoint_id
    job["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    _save_job(job)


async def _run_langgraph_pipeline(job_id: str):
    """Run (or resume, or replay) the pipeline graph for a job.

    One graph with a gate after every phase (graph/main_graph.py). A gate
    whose phase is in job["pause_after"] interrupts; the job is then
    `waiting_for_hitl` with `hitl_pending` = the HumanInterrupt payload.
    Resume: job["_pending_response"] (HumanResponse) → Command(resume=...).
    Replay: job["_replay_from"] = phase → invoke from the checkpoint taken
    before that phase (LangGraph replay), e.g. after a prompt change.
    Retry: job["_retry_failed"] → re-invoke the thread with input None, which
    re-runs the node that raised (the thread is still parked on it).
    """
    from langgraph.types import Command

    from graph.main_graph import build_graph

    job = _get_job(job_id)
    if not job:
        print(f"[LANGGRAPH] job {job_id} not found in memory or GCS")
        return
    jobs[job_id] = job
    # A reviewer's uploaded full text re-runs the evidence step for those
    # references alone — not the graph. It shares this worker (and so the queue,
    # the job record and the event feed) but rebuilds its inputs from
    # results.json, because a finished job has no live checkpoint to replay.
    if job.get("_rerun_evidence"):
        from app.fulltext_rerun import rerun_evidence
        try:
            await rerun_evidence(job)
        except Exception as exc:
            import traceback
            job["status"] = "error"
            job["error"] = f"Evidence re-run: {type(exc).__name__}: {exc}"[:2000]
            job["error_trace"] = traceback.format_exc()[-4000:]
            _save_job(job)
        return
    pause_after = _pause_after(job)
    graph = build_graph(checkpointer=_checkpointer())
    config = {"configurable": {"thread_id": job_id, "prompt_overrides": job.get("prompt_overrides") or {}}}
    pending = job.pop("_pending_response", None)
    replay_from = job.pop("_replay_from", None)
    retry_failed = job.pop("_retry_failed", None)

    def _has_thread() -> bool:
        try:
            return bool(graph.get_state(config).values)
        except Exception:
            return False

    if retry_failed and _has_thread():
        # The thread is still parked on the node that raised, so plain input
        # None re-runs it; no checkpoint_id, because the current checkpoint
        # already *is* the one taken before that node.
        graph_input = None
        mode = f"retry:{job.get('failed_node') or '?'}"
        for k in ("error", "error_trace", "failed_node", "failed_phase"):
            job.pop(k, None)
    elif replay_from and _has_thread():
        cid = (job.get("phase_checkpoints") or {}).get(replay_from)
        if cid:
            config = {"configurable": {**config["configurable"], "checkpoint_id": cid}}
        graph_input = None
        mode = f"replay:{replay_from}"
        for k in ("error", "error_trace", "failed_node", "failed_phase"):
            job.pop(k, None)
    elif pending is not None and _has_thread():
        graph_input = Command(resume=pending)
        mode = "resume"
    elif pending is not None or replay_from or retry_failed:
        # checkpoint gone (new instance): rebuild from the job snapshot and
        # continue from the phase after the pause, applying the edits ourselves.
        # A retry of a failed phase takes the same road — _record_phase_failure
        # snapshots the state, so a crash that outlives the instance is still
        # restartable at the phase that broke.
        from graph.gates import apply_response
        saved = dict(job.get("_hitl_saved_state") or {})
        paused = job.get("paused_at") or job.get("failed_phase") or "extract"
        saved.update(apply_response(paused, saved, pending) if pending else {})
        input_path = saved.get("input_local_path") or job.get("input_path", "")
        if input_path and not Path(input_path).exists() and job.get("gcs_uri"):
            try:
                Path(input_path).parent.mkdir(parents=True, exist_ok=True)
                path_part = job["gcs_uri"][len("gs://"):]
                bucket_name, _, object_key = path_part.partition("/")
                _get_gcs().bucket(bucket_name).blob(object_key).download_to_filename(input_path)
            except Exception as e:
                print(f"[HITL RESUME] Could not re-download input: {e}")
        resuming_here = bool(replay_from or retry_failed)   # re-run this phase, not the one after it
        graph_input = {**saved, "job_id": job_id, "output_dir": job["output_dir"], "status": "running",
                       "events": [], "pause_after": [p for p in pause_after if p != paused] if not resuming_here else pause_after}
        mode = "snapshot-rebuild:retry" if retry_failed else "snapshot-rebuild"
        print(f"[HITL] checkpoint missing for {job_id}, rebuilding from job snapshot (at={paused})")
        entry = PHASE_NODE.get(replay_from or paused, "ssr") if resuming_here else _next_node(paused)
        graph = build_graph(checkpointer=_checkpointer(), entry=entry)
        for k in ("error", "error_trace", "failed_node", "failed_phase"):
            job.pop(k, None)
        config = {"configurable": {"thread_id": f"{job_id}_{uuid.uuid4().hex[:6]}", "prompt_overrides": job.get("prompt_overrides") or {}}}
    else:
        graph_input = {
            "job_id": job_id,
            "input_local_path": job["input_path"],
            "output_dir": job["output_dir"],
            "hitl_enabled": bool(job.get("hitl_enabled", False)),
            "pause_after": pause_after,
            "evolve": bool(job.get("evolve", False)),
            "notify_email": job.get("notify_email", ""),
            "input_mode": _input_mode(job.get("input_mode")),
            "status": "running",
            "phase": "phase1",
            "events": [],
            "phase_results": {},
        }
        mode = "start"

    print(f"[LANGGRAPH] job {job_id} mode={mode} pause_after={pause_after}")
    from app import prompts as _prompts
    from patent_analyzer import metering
    # Same job_id on a resume keeps the phases already accounted for; a new job
    # starts the counters clean.
    metering.start_run(job_id)
    _prompts.set_overrides(job.get("prompt_overrides") or {})
    job["status"] = "running"
    job["paused_at"] = ""
    job["hitl_pending"] = None
    _save_job(job)

    # A phase is one node, and last_heartbeat only ticked on a node *update* —
    # so a 20-minute search node was 20 minutes of silence. Any instance other
    # than this one (Cloud Run scales out; _active_pipelines is per-instance)
    # then sees a >15 min stale heartbeat on /status and declares a perfectly
    # healthy job a zombie. Tick it from the side while the graph runs.
    async def _heartbeat():
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            job["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
            _save_job(job)

    hb = asyncio.create_task(_heartbeat())

    interrupted = None
    try:
        stream = graph.astream(graph_input, config=config, stream_mode="updates")
        async for update in stream:
            if not isinstance(update, dict):
                continue
            if "__interrupt__" in update:
                interrupted = update["__interrupt__"]
                continue
            for node_name, patch in update.items():
                if not isinstance(patch, dict):
                    continue
                print(f"[ASTREAM] node={node_name} keys={list(patch.keys())[:10]} events={len(patch.get('events', []))}")
                if node_name in PHASE_OF_NODE:
                    # checkpoint taken before this phase ran = replay point for "rerun this phase"
                    try:
                        hist = list(graph.get_state_history(config))
                        before = next((s for s in hist if s.next == (node_name,)), None)
                        if before is not None:
                            job.setdefault("phase_checkpoints", {})[PHASE_OF_NODE[node_name]] = before.config["configurable"]["checkpoint_id"]
                    except Exception:
                        pass
                if patch.get("phase"):
                    job["phase"] = patch["phase"]
                if patch.get("status") and patch["status"] != "running":
                    job["status"] = patch["status"]
                for evt in patch.get("events", []):
                    # the job record is rewritten to disk and GCS on every
                    # heartbeat; one round_done payload was 6.6 MB of provenance
                    # that funnel.json already holds (patent_analyzer.funnel)
                    job.setdefault("events", []).append(slim_event(evt))
                    if evt.get("phase"):
                        job["phase"] = evt["phase"]
                for pk, pd in patch.get("phase_results", {}).items():
                    job.setdefault("phases", {})[pk] = pd.get("data", {})
                    job["phase"] = pk
                if patch.get("search_stats"):
                    job.setdefault("phases", {})["phase3"] = slim_search_stats(patch["search_stats"], job_id)
                if patch.get("scoring_report"):
                    sr = patch["scoring_report"]
                    top = sr[0].get("similarity_score", 0) if sr else 0
                    job.setdefault("phases", {})["phase4"] = {"evaluated": len(sr), "top_score": round(top, 4)}
                if patch.get("user_edits"):
                    job.setdefault("user_edits", []).extend(patch["user_edits"])
                if patch.get("prompt_versions"):
                    job.setdefault("prompt_versions", {}).update(patch["prompt_versions"])
                # keep the cost/timing accounting on the job record too, so it
                # survives a pause and is readable from /status while running
                _refresh_accounting(job)
                job["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
                _save_job(job)
    except Exception as exc:
        # A node that raises used to propagate to _pipeline_worker, which wrote
        # a one-line "Pipeline worker crash" onto whatever it found in the
        # in-memory dict — often nothing, leaving the job "running" forever with
        # no way back in. Record it here instead, while the graph state (which
        # names the failed node and is resumable from it) is still in hand.
        _refresh_accounting(job)
        _record_phase_failure(job, graph, config, exc)
        return
    finally:
        hb.cancel()

    if interrupted:
        hi = interrupted[0].value if hasattr(interrupted[0], "value") else interrupted[0]
        phase = str(hi.get("action_request", {}).get("action", "")).replace("review_", "") or "extract"
        try:
            values = graph.get_state(config).values
        except Exception:
            values = {}
        job["_hitl_saved_state"] = {k: values[k] for k in _SNAPSHOT_KEYS if k in values}
        job.setdefault("prompt_versions", {}).update(_prompts.used_versions())
        # The gate marks the phase and then interrupts, so its patch never
        # reaches the loop above: refresh the accounting from the module or the
        # job shows $0 for everything it has already spent (seen live, job
        # 2f92d815 paused at idca).
        _refresh_accounting(job)
        job["status"] = "waiting_for_hitl"
        job["paused_at"] = phase
        # HumanInterrupt plus the legacy fields the current frontend form reads
        job["hitl_pending"] = {**hi, "type": f"{phase}_review", "phase": phase,
                               "prompt": hi.get("description") or f"Phase '{phase}' finished. Review, then continue.",
                               "options": ["A) Looks good, continue", "B) Need modifications"],
                               "data": {"checklist": values.get("checklist", []), "summary": (values.get("summary") or "")[:500],
                                        "extraction": values.get("extraction"), "next_node": _next_node(phase)}}
        print(f"[HITL] Job {job_id} paused after {phase}")
        _save_job(job)
        return

    try:
        final_state = graph.get_state(config).values
    except Exception:
        final_state = {}
    job["status"] = final_state.get("status", "completed")
    if job["status"] not in ("error", "waiting_for_hitl"):
        job["status"] = "completed"
    job["phase"] = final_state.get("phase", "phase5")
    if final_state.get("error"):
        job["error"] = final_state["error"]
    job.setdefault("prompt_versions", {}).update(_prompts.used_versions())
    _refresh_accounting(job)
    job.pop("_hitl_saved_state", None)
    _save_job(job)


async def _pipeline_worker():
    """Single worker that processes pipeline jobs sequentially."""
    while True:
        job_id = await _pipeline_queue.get()
        try:
            if job_id in _pending_jobs:
                _pending_jobs.remove(job_id)
            _active_pipelines.add(job_id)
            await _run_langgraph_pipeline(job_id)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"[PIPELINE WORKER] job {job_id} crashed: {exc}\n{tb}")
            # _get_job, not jobs.get: a crash before the pipeline put the job in
            # the in-memory dict (job not loadable, checkpointer construction,
            # build_graph) used to silently drop the error on the floor and
            # leave the job "running" forever.
            job = _get_job(job_id)
            if job is None:
                print(f"[PIPELINE WORKER] job {job_id} is not in memory, on disk or in GCS — "
                      "the crash cannot be recorded against it")
            elif job.get("status") not in ("completed", "waiting_for_hitl"):
                job["status"] = "error"
                job["error"] = f"Pipeline worker crash: {type(exc).__name__}: {exc}"[:2000]
                job["error_trace"] = tb[-4000:]
                job["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
                _save_job(job)
        finally:
            _active_pipelines.discard(job_id)
            _pipeline_queue.task_done()


_worker_task = None


@app.on_event("startup")
async def _start_pipeline_worker():
    # keep a reference: "a task disappearing mid-execution" when only a weak
    # reference exists (asyncio docs, create_task) — seen once: a job stayed
    # queued for 10 min with the worker gone
    global _worker_task
    _worker_task = asyncio.create_task(_pipeline_worker())


def _next_node(phase: str) -> str:
    order = ["idca", "ssr", "search", "evaluate", "draft", "report"]
    node = PHASE_NODE.get(phase, "ssr")
    return order[min(order.index(node) + 1, len(order) - 1)]


# GCS upload budget. Without one, a network hiccup blocks the pipeline inside
# what is meant to be a best-effort side write; the library default is no
# deadline at all on the retried path.
GCS_TIMEOUT_S = float(os.getenv("GCS_TIMEOUT_S", "60"))

# GCS failures are swallowed on purpose (local disk is the primary during a run),
# but swallowing them *silently* meant a deploy with no bucket access looked
# perfectly healthy until the report was missing. Log the first failure per
# operation, then stay quiet, and record it so /health can say so.
_gcs_failures: dict[str, dict] = {}


def _gcs_failed(op: str, exc: BaseException) -> None:
    first = op not in _gcs_failures
    rec = _gcs_failures.setdefault(op, {"n": 0, "last": ""})
    rec["n"] += 1
    rec["last"] = f"{type(exc).__name__}: {exc}"[:300]
    if first:
        print(f"[GCS] {op} failed ({rec['last']}) — continuing on local disk. "
              "Further failures of this operation are counted, not logged.")


# Lazily-built storage.Client. The module-level binding matters: `global
# _gcs_client` without it makes _get_gcs raise NameError, which the best-effort
# try/except around every upload then swallowed. Deleted by db92443 and unnoticed
# for exactly that reason until /health (then /healthz) started reporting GCS failures
# (b0488ae) — job state had not reached the bucket since.
_gcs_client = None


def _get_gcs():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def _gcs_put(op: str, blob_path: str, data: str, content_type: str) -> bool:
    """One best-effort upload, bounded and reported."""
    try:
        blob = _get_gcs().bucket(GCS_BUCKET).blob(blob_path)
        blob.upload_from_string(data, content_type=content_type, timeout=GCS_TIMEOUT_S)
        return True
    except Exception as exc:
        _gcs_failed(op, exc)
        return False


def _save_job(job: dict):
    """Persist job state to GCS."""
    # Local disk (for same-instance reads)
    job_dir = Path(job["output_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    state_path = job_dir / "state.json"
    state_data = json.dumps(job, ensure_ascii=False, default=str)
    state_path.write_text(state_data)
    # GCS (persistent across instances/deploys) — best-effort; local disk is the
    # primary during a pipeline run, so a failure is logged, not raised.
    _gcs_put("save_job", f"{GCS_PREFIX}{job['id']}/state.json", state_data, "application/json")


def _save_report_to_gcs(job_id: str, report_html: str):
    """Upload report.html to GCS."""
    _gcs_put("save_report", f"{GCS_PREFIX}{job_id}/report.html", report_html, "text/html")


def _save_results_to_gcs(job_id: str, results_json: str):
    """Upload results.json to GCS."""
    _gcs_put("save_results", f"{GCS_PREFIX}{job_id}/results.json", results_json, "application/json")


def _save_md_to_gcs(job_id: str, md_text: str):
    _gcs_put("save_md", f"{GCS_PREFIX}{job_id}/report.md", md_text, "text/markdown")


def _load_job(job_id: str) -> dict | None:
    """Try to load job state from local disk, then GCS."""
    # Local disk first
    state_path = OUTPUT_BASE / job_id / "state.json"
    if state_path.exists():
        job = json.loads(state_path.read_text())
        jobs[job_id] = job
        return job
    # GCS fallback
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/state.json")
        if blob.exists():
            job = json.loads(blob.download_as_text())
            jobs[job_id] = job
            return job
    except Exception:
        pass
    return None


def _list_jobs_from_gcs() -> list[dict]:
    """List all jobs from GCS."""
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blobs = bucket.list_blobs(prefix=GCS_PREFIX)
        for blob in blobs:
            if blob.name.endswith("/state.json"):
                job_id = blob.name.replace(GCS_PREFIX, "").split("/")[0]
                if job_id and job_id not in jobs:
                    try:
                        job = json.loads(blob.download_as_text())
                        jobs[job_id] = job
                    except Exception:
                        pass
    except Exception:
        pass
    return list(jobs.values())


def _get_job(job_id: str) -> dict | None:
    """Get job from memory, disk, or GCS."""
    if job_id in jobs:
        return jobs[job_id]
    return _load_job(job_id)


# ─── REST API ──────────────────────────────────────────────────

# `/healthz`, not `/health`, was the path until 2026-09-19, and on Cloud Run it
# never answered: something in front of the container serves its own branded HTML
# 404 for that exact path. Measured against revisions 00074 and 00075 through the
# IAM proxy — `/health`, `/healthz2` and a nonsense path all came back as
# FastAPI's `{"detail":"Not Found"}`, `/healthz` alone came back as Google's 404
# page, and unauthenticated `/` returns 403 while unauthenticated `/healthz`
# returns that same 404, so it is being taken before IAM, not by this app. The
# endpoint that exists to report GCS failures had therefore been unreadable from
# outside for its whole life — which is the only thing it is for.
@app.get("/health")
async def health():
    # `gcs` is empty on a healthy deploy; a populated one means job state and
    # reports are only on this instance's disk and will not survive it.
    return {"status": "ok", "version": "0.3.0",
            "gcs": {k: v["n"] for k, v in _gcs_failures.items()},
            "gcs_last_error": {k: v["last"] for k, v in _gcs_failures.items()}}


INPUT_MODES = ("academic_paper", "manuscript", "disclosure", "patent_draft")


def _input_mode(value) -> str:
    """Explicit input type from the upload form; "" (or "auto") = let IDCA detect."""
    v = str(value or "").strip().lower()
    return v if v in INPUT_MODES else ""


@app.post("/analyze")
async def start_analysis(
    file: UploadFile = File(...),
    evolve: bool = Form(False),
    notify_email: str = Form(""),
    hitl_enabled: bool = Form(False),
    pause_after: str = Form(""),
    input_mode: str = Form(""),
    user: dict = Depends(require_auth),
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True)

    input_path = job_dir / file.filename
    content = await file.read()
    input_path.write_bytes(content)

    queue_depth = len(_pending_jobs)
    job = {
        "id": job_id,
        "status": "queued",
        "filename": file.filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "starting",
        "phases": {},
        "output_dir": str(job_dir),
        "input_path": str(input_path),
        "evolve": bool(evolve),
        "hitl_enabled": bool(hitl_enabled),
        "pause_after": [x.strip() for x in (pause_after or "").split(",") if x.strip()],
        "input_mode": _input_mode(input_mode),
        "notify_email": notify_email or "",
        "submitted_by": user.get("email", ""),
    }
    jobs[job_id] = job
    _save_job(job)
    _enqueue_job(job_id)

    return {"job_id": job_id, "status": "queued", "queue_position": queue_depth, "hitl_enabled": bool(hitl_enabled)}


@app.get("/upload-url")
async def get_upload_url(filename: str, content_type: str = "application/pdf"):
    """Generate a signed PUT URL so the client can upload directly to GCS, bypassing 32MB Cloud Run limit."""
    from datetime import timedelta

    from google.auth import default
    from google.auth.transport.requests import Request as AuthRequest

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    job_id = str(uuid.uuid4())[:8]
    safe_name = re.sub(r"[^\w.-]", "_", filename)
    object_key = f"{GCS_PREFIX}{job_id}/upload/{safe_name}"

    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(object_key)
        # Use service-account credentials for signing (required on Cloud Run)
        creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(AuthRequest())
        signed_url = blob.generate_signed_url(
            expiration=timedelta(hours=1),
            method="PUT",
            version="v4",
            service_account_email="amie-backend-sa@aime-hello-world.iam.gserviceaccount.com",
            access_token=creds.token,
            content_type=content_type,
        )
    except Exception as e:
        raise HTTPException(500, f"Signed URL generation failed: {e}")

    return {
        "job_id": job_id,
        "signed_url": signed_url,
        "gcs_uri": f"gs://{GCS_BUCKET}/{object_key}",
        "object_key": object_key,
        "content_type": content_type,
        "headers": {"Content-Type": content_type},
    }


@app.post("/analyze-gcs")
async def start_analysis_from_gcs(
    payload: dict,
    evolve: bool = False,
    user: dict = Depends(require_auth),
):
    """Start analysis with a file already uploaded to GCS via signed URL."""
    job_id = payload.get("job_id") or str(uuid.uuid4())[:8]
    gcs_uri = payload.get("gcs_uri")
    filename = payload.get("filename", "upload.pdf")
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        raise HTTPException(400, "Missing/invalid gcs_uri")

    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / filename

    # Download from GCS to local
    try:
        path_part = gcs_uri[len("gs://"):]
        bucket_name, _, object_key = path_part.partition("/")
        bucket = _get_gcs().bucket(bucket_name)
        blob = bucket.blob(object_key)
        blob.download_to_filename(str(input_path))
    except Exception as e:
        raise HTTPException(500, f"Failed to download from GCS: {e}")

    queue_depth = len(_pending_jobs)
    job = {
        "id": job_id,
        "status": "queued",
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "starting",
        "phases": {},
        "output_dir": str(job_dir),
        "input_path": str(input_path),
        "gcs_uri": gcs_uri,
        "evolve": bool(evolve or payload.get("evolve")),
        "hitl_enabled": bool(payload.get("hitl_enabled", False)),
        "pause_after": [x.strip() for x in str(payload.get("pause_after") or "").split(",") if x.strip()]
                       if not isinstance(payload.get("pause_after"), list) else list(payload.get("pause_after")),
        "input_mode": _input_mode(payload.get("input_mode")),
        "notify_email": payload.get("notify_email", ""),
        "submitted_by": user.get("email", ""),
    }
    jobs[job_id] = job
    _save_job(job)
    _enqueue_job(job_id)

    return {"job_id": job_id, "status": "queued", "queue_position": queue_depth, "hitl_enabled": job["hitl_enabled"]}


@app.get("/status/{job_id}")
async def get_status(job_id: str, user: dict | None = Depends(optional_auth)):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Detect zombie jobs: running with no heartbeat for >15 min.
    # The real fix for Phase 4 going silent is heartbeat-after-each-doc inside
    # evaluate_batch (see on_doc_done hook). The threshold itself stays tight
    # so a *genuinely* stuck pipeline doesn't hide.
    #
    # Cross-instance reconciliation: if Phase 5 already wrote the report to GCS,
    # the pipeline finished on another Cloud Run instance — this instance's
    # in-memory view is stale. Trust the GCS artifact, not the heartbeat.
    response = {k: v for k, v in job.items() if not k.startswith("_")}   # _hitl_saved_state etc. stay server-side
    if job.get("status") == "queued":
        pos = _queue_position(job_id)
        if pos is not None:
            response["queue_position"] = pos
            response["queue_depth"] = len(_pending_jobs)
    elif job.get("status") == "running":
        # Cross-instance reconciliation: if report already in GCS, promote to completed
        try:
            bucket = _get_gcs().bucket(GCS_BUCKET)
            if bucket.blob(f"{GCS_PREFIX}{job_id}/report.html").exists():
                job["status"] = "completed"
                job["phase"] = "phase5"
                _save_job(job)
                return job
        except Exception:
            pass
        # Zombie detection: job claims running but heartbeat stale for >15 min
        if job_id not in _active_pipelines:
            last_hb = job.get("last_heartbeat", job.get("created_at", ""))
            try:
                hb_time = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
                stale_seconds = (datetime.now(timezone.utc) - hb_time).total_seconds()
            except Exception:
                stale_seconds = 0
            if stale_seconds > 900:
                job["status"] = "error"
                job["error"] = "Pipeline terminated — server instance was recycled before completion"
                _save_job(job)
                return job
    return response


@app.get("/events/{job_id}")
async def get_events(job_id: str, since: int = 0):
    """Get events for a job, optionally since a given index."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    events = job.get("events", [])
    return {
        "events": events[since:],
        "total": len(events),
        "status": job.get("status"),
        "phase": job.get("phase"),
    }


@app.get("/report/{job_id}")
async def get_report(job_id: str, user: dict | None = Depends(optional_auth)):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Try local first, then GCS
    report_path = Path(job.get("output_dir", f"/tmp/outputs/{job_id}")) / "report.html"
    if report_path.exists():
        return FileResponse(str(report_path), media_type="text/html")
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/report.html")
        if blob.exists():
            return Response(content=blob.download_as_bytes(), media_type="text/html")
    except Exception:
        pass
    raise HTTPException(404, "Report not generated yet")


@app.get("/report-md/{job_id}")
async def get_report_md(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    md_path = Path(job.get("output_dir", f"/tmp/outputs/{job_id}")) / "report.md"
    if md_path.exists():
        return FileResponse(str(md_path), media_type="text/markdown")
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/report.md")
        if blob.exists():
            return Response(content=blob.download_as_bytes(), media_type="text/markdown")
    except Exception:
        pass
    raise HTTPException(404, "Markdown report not generated yet")


@app.get("/results/{job_id}")
async def get_results(job_id: str, user: dict | None = Depends(optional_auth)):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    results_path = Path(job.get("output_dir", f"/tmp/outputs/{job_id}")) / "results.json"
    if results_path.exists():
        return FileResponse(str(results_path), media_type="application/json")
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/results.json")
        if blob.exists():
            return Response(content=blob.download_as_bytes(), media_type="application/json")
    except Exception:
        pass
    raise HTTPException(404, "Results not ready")


def _job_artifact(job_id: str, name: str, what: str):
    """Serve a per-job JSON artefact: local disk first, then the bucket."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = Path(job.get("output_dir", f"/tmp/outputs/{job_id}")) / name
    if path.exists():
        return FileResponse(str(path), media_type="application/json")
    try:
        blob = _get_gcs().bucket(GCS_BUCKET).blob(f"{GCS_PREFIX}{job_id}/{name}")
        if blob.exists():
            return Response(content=blob.download_as_bytes(), media_type="application/json")
    except Exception as exc:
        _gcs_failed(f"read_{name}", exc)
    raise HTTPException(404, f"{what} not ready")


@app.get("/api/jobs/{job_id}/funnel")
async def get_funnel(job_id: str, user: dict = Depends(require_auth)):
    """The per-document search funnel, split out of results.json because it is
    most of its size (job 075b99c1: 12.0 MB of a 20.5 MB file) and almost nobody
    reads it — the Express proxy re-serialises whatever it forwards, so the
    bytes are paid for twice. results.json keeps a projection plus a
    `search.summary.funnel_ref` pointing here."""
    return _job_artifact(job_id, "funnel.json", "Funnel")


@app.get("/api/jobs/{job_id}/usage")
async def get_usage(job_id: str, user: dict = Depends(require_auth)):
    """Per-phase LLM / external / BigQuery counts for one job.

    The job record is the source: app.llm.usage is a process-wide meter, so
    reading it live would report whatever else this instance has run since.
    patent_analyzer.metering diffs it per phase while the job runs and the
    result is persisted on the job (and in results.json under `cost`).
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    cost = job.get("cost") or {}
    phases = job.get("phase_metrics") or cost.get("phases") or {}
    if not phases:
        # finished before the job record carried it, or an older job: fall back
        # to results.json, which report_node writes with the same block
        try:
            rp = Path(job.get("output_dir", f"/tmp/outputs/{job_id}")) / "results.json"
            if rp.exists():
                cost = (json.loads(rp.read_text()) or {}).get("cost") or {}
                phases = cost.get("phases") or {}
        except Exception:
            pass
    if not phases:
        raise HTTPException(404, "No usage recorded for this job yet")
    # The ledger is the same numbers flattened into rows, plus which phase cost
    # the most and which calls failed or degraded. Recomputed from `phases` when
    # the record predates it, so an old job still answers the three questions.
    ledger = job.get("ledger")
    if not ledger:
        from patent_analyzer import metering
        ledger = metering.ledger(phases)
    return {"job_id": job_id, "status": job.get("status"), "phase": job.get("phase"),
            "phases": phases, "totals": cost.get("totals") or {},
            "prices_usd_per_mtok": cost.get("prices_usd_per_mtok") or {},
            "bigquery_usd_per_tib": cost.get("bigquery_usd_per_tib"),
            "note": cost.get("note") or "", "ledger": ledger}


@app.get("/api/quota")
async def get_quota(user: dict = Depends(require_auth)):
    """What is left on every external source, in one shape.

    Not per job: these counters are shared by every job on the deployment, and
    the reason to look at them is to find out *before* submitting that the
    channel a run depends on is already spent. SerpAPI keys are identified by
    the same 8-character fingerprint the recall channel logs — never the key.
    """
    from patent_analyzer import quota
    return await quota.snapshot()


@app.post("/feedback/{job_id}")
async def submit_feedback(job_id: str, payload: dict):
    """Capture user feedback on a completed job. Used later (offline) to pair
    feedback with state.json events and refine the pipeline.

    Payload shape:
      {"rating": 1-5 or null, "comment": "...", "tags": ["hallucination","missed_prior_art",...]}
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rating": payload.get("rating"),
        "comment": (payload.get("comment") or "")[:4000],
        "tags": (payload.get("tags") or [])[:20],
    }
    fb_path = job_dir / "feedback.json"
    existing = []
    if fb_path.exists():
        try:
            existing = json.loads(fb_path.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    existing.append(entry)
    fb_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    # Best-effort mirror to GCS
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        bucket.blob(f"{GCS_PREFIX}{job_id}/feedback.json").upload_from_string(
            json.dumps(existing, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
    except Exception:
        pass
    return {"ok": True, "total_feedback": len(existing)}


@app.get("/feedback/{job_id}")
async def get_feedback(job_id: str):
    job_dir = OUTPUT_BASE / job_id
    fb_path = job_dir / "feedback.json"
    if fb_path.exists():
        return json.loads(fb_path.read_text())
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_PREFIX}{job_id}/feedback.json")
        if blob.exists():
            return json.loads(blob.download_as_bytes())
    except Exception:
        pass
    return []


@app.get("/jobs")
async def list_jobs(user: dict = Depends(require_auth)):
    _list_jobs_from_gcs()
    _reap_zombie_jobs()
    user_email = user.get("email", "")
    dev = is_developer(user)
    return [{"id": j["id"], "status": j["status"], "phase": j["phase"], "filename": j["filename"], "created_at": j.get("created_at")}
            for j in jobs.values()
            if dev or j.get("submitted_by", "") == user_email]


def _reap_zombie_jobs():
    """Mark any running job with stale heartbeat (>15 min) as error."""
    for job in jobs.values():
        if job.get("status") not in ("running", "queued"):
            continue
        job_id = job.get("id", "")
        if job_id in _active_pipelines:
            continue
        last_hb = job.get("last_heartbeat", job.get("created_at", ""))
        try:
            hb_time = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
            stale_seconds = (datetime.now(timezone.utc) - hb_time).total_seconds()
        except Exception:
            stale_seconds = 9999
        if stale_seconds > 900:
            job["status"] = "error"
            job["error"] = "Pipeline terminated — server instance was recycled before completion"
            _save_job(job)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, user: dict = Depends(require_auth)):
    job = _get_job(job_id)
    if job and not is_developer(user) and job.get("submitted_by", "") != user.get("email", ""):
        raise HTTPException(403, "You can only delete your own jobs")
    return await _delete_job_internal(job_id)


async def _delete_job_internal(job_id: str):
    jobs.pop(job_id, None)
    local_dir = OUTPUT_BASE / job_id
    if local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)
    deleted_blobs = 0
    try:
        bucket = _get_gcs().bucket(GCS_BUCKET)
        blobs = list(bucket.list_blobs(prefix=f"{GCS_PREFIX}{job_id}/"))
        for blob in blobs:
            blob.delete()
            deleted_blobs += 1
    except Exception as e:
        return {"deleted": True, "gcs_error": str(e), "local": True}
    return {"deleted": True, "blobs_removed": deleted_blobs, "local": True}


@app.post("/jobs/cleanup")
async def cleanup_old_jobs(days: int = 7, user: dict = Depends(require_auth)):
    if not is_developer(user):
        raise HTTPException(403, "Developer access required")
    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    _list_jobs_from_gcs()
    deleted = []
    for job_id in list(jobs.keys()):
        job = jobs[job_id]
        created_at = job.get("created_at", "")
        try:
            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            if ts < cutoff:
                await _delete_job_internal(job_id)
                deleted.append(job_id)
        except Exception:
            pass
    return {"deleted_count": len(deleted), "deleted_ids": deleted}


@app.post("/admin/set-developer")
async def admin_set_developer(payload: dict, user: dict = Depends(require_auth)):
    if not is_developer(user):
        raise HTTPException(403, "Developer access required")
    email = payload.get("email")
    is_dev = payload.get("developer", True)
    if not email:
        raise HTTPException(400, "Missing email")
    uid = set_developer(email, is_dev)
    return {"ok": True, "uid": uid, "email": email, "developer": is_dev}


# ─── A2A Protocol ──────────────────────────────────────────────

A2A_AGENT_CARD = {
    "name": "Patent Analyzer",
    "description": "AI-powered patent novelty analysis. Upload a paper/patent PDF and get a comprehensive prior art search and novelty assessment report.",
    "url": "",
    "version": "0.3.0",
    "defaultInputModes": ["application/pdf", "text/plain", "application/json"],
    "defaultOutputModes": ["application/json", "text/plain"],
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    },
    "skills": [
        {
            "id": "patent_analyze",
            "name": "Analyze patent novelty",
            "description": "Upload a PDF and run full novelty analysis pipeline: invention detection, decomposition, prior art search, deep evaluation, and report generation.",
            "tags": ["patent", "novelty", "analysis", "prior-art"],
            "examples": ["Send a PDF file part to start analysis"],
        },
        {
            "id": "patent_status",
            "name": "Get analysis status",
            "description": "Poll the status of a running or completed patent analysis job.",
            "tags": ["patent", "status", "polling"],
            "examples": ["Send metadata.task_id to get job status"],
        },
    ],
    "supportsAuthenticatedExtendedCard": False,
}


def _build_agent_card(request: Request) -> dict:
    base_url = os.getenv("A2A_PUBLIC_URL") or str(request.base_url).rstrip("/")
    if base_url.startswith("http://") and "run.app" in base_url:
        base_url = base_url.replace("http://", "https://", 1)
    card = dict(A2A_AGENT_CARD)
    card["url"] = f"{base_url}/a2a"
    return card


def _job_to_a2a_task(job: dict) -> dict:
    status_map = {
        "queued": "submitted",
        "running": "working",
        "completed": "completed",
        "error": "failed",
    }
    job_status = job.get("status", "queued")
    # Determine which phase we're in for running jobs
    if job_status == "running":
        phase_label = job.get("phase", "running")
    else:
        phase_label = job_status

    task = {
        "id": job["id"],
        "contextId": job["id"],
        "status": {
            "state": status_map.get(job_status, "unknown"),
            "timestamp": job.get("created_at", datetime.now(timezone.utc).isoformat()),
            "message": {
                "messageId": f"{job['id']}-status",
                "role": "agent",
                "parts": [{"text": f"Phase: {phase_label}, Status: {job_status}"}],
            },
        },
        "metadata": {
            "phase": job.get("phase"),
            "phases": job.get("phases", {}),
            "filename": job.get("filename"),
        },
    }

    if job_status == "completed":
        task["metadata"]["report_url"] = f"/report/{job['id']}"
        task["metadata"]["report_md_url"] = f"/report-md/{job['id']}"
        task["metadata"]["results_url"] = f"/results/{job['id']}"
        md_path = Path(job.get("output_dir", f"/tmp/outputs/{job['id']}")) / "report.md"
        md_text = None
        if md_path.exists():
            md_text = md_path.read_text(encoding="utf-8")
        else:
            try:
                bucket = _get_gcs().bucket(GCS_BUCKET)
                blob = bucket.blob(f"{GCS_PREFIX}{job['id']}/report.md")
                if blob.exists():
                    md_text = blob.download_as_text()
            except Exception:
                pass
        if md_text:
            task["artifacts"] = [{
                "name": "report.md",
                "parts": [{"type": "text", "text": md_text}],
            }]

    if job_status == "error":
        task["metadata"]["error"] = job.get("error")

    return task


@app.get("/.well-known/agent-card.json")
async def well_known_agent_card(request: Request):
    return JSONResponse(_build_agent_card(request))


@app.get("/agent-card.json")
async def agent_card(request: Request):
    return JSONResponse(_build_agent_card(request))


@app.post("/a2a")
async def a2a_jsonrpc(request: Request):
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32600, "message": "Invalid JSON body"},
        }, status_code=400)

    rpc_id = raw.get("id") if isinstance(raw, dict) else None
    method = raw.get("method") if isinstance(raw, dict) else None
    params = raw.get("params", {}) if isinstance(raw, dict) else {}

    # agent.getCard
    if method == "agent.getCard":
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": _build_agent_card(request),
        })

    # tasks.get / tasks/get
    if method in ("tasks.get", "tasks/get"):
        task_id = params.get("taskId") or params.get("id") or params.get("task_id")
        job = _get_job(task_id) if task_id else None
        if not job:
            return JSONResponse({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32001, "message": f"Task not found: {task_id}"},
            }, status_code=404)
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": _job_to_a2a_task(job),
        })

    # message/send — the main entry point
    if method in ("message/send", "tasks.create"):
        return await _handle_a2a_send(raw, rpc_id, params, request)

    return JSONResponse({
        "jsonrpc": "2.0", "id": rpc_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }, status_code=404)


async def _handle_a2a_send(
    raw: dict, rpc_id: Any, params: dict,
    request: Request,
):
    import base64
    import binascii

    # Determine skill
    metadata = params.get("metadata", {})
    message = params.get("message", {})
    msg_meta = message.get("metadata", {})
    skill_id = metadata.get("skill_id") or msg_meta.get("skill_id") or "patent_analyze"

    # Handle patent_status
    if skill_id == "patent_status":
        task_id = metadata.get("task_id") or msg_meta.get("task_id")
        job = _get_job(task_id) if task_id else None
        if not job:
            return JSONResponse({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32001, "message": f"Task not found: {task_id}"},
            }, status_code=404)
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": _job_to_a2a_task(job),
        })

    # Handle patent_analyze — extract PDF from file parts
    pdf_bytes = None
    filename = "upload.pdf"

    for part in message.get("parts", []):
        file_obj = part.get("file")
        if not file_obj:
            continue
        fname = file_obj.get("name", "upload.pdf")
        mime = (file_obj.get("mimeType") or "application/pdf").lower()
        is_pdf = fname.lower().endswith(".pdf") or mime == "application/pdf"
        if not is_pdf:
            continue
        b64 = file_obj.get("bytes")
        if b64:
            try:
                pdf_bytes = base64.b64decode(b64, validate=True)
                filename = fname
                break
            except (binascii.Error, ValueError):
                continue

    if pdf_bytes is None:
        # Check for text-based invocation (e.g. from n8n with just a URL or instructions)
        text_parts = [p.get("text", "") for p in message.get("parts", []) if p.get("text")]
        if not text_parts:
            return JSONResponse({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32602, "message": "No PDF file part found. Send a FilePart with PDF bytes."},
            }, status_code=400)
        # For now, return helpful error. Future: support URL-based PDF fetching.
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "error": {"code": -32602, "message": "Text-only invocation not yet supported. Please send a PDF file part."},
        }, status_code=400)

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True)
    input_path = job_dir / filename
    input_path.write_bytes(pdf_bytes)

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "starting",
        "phases": {},
        "output_dir": str(job_dir),
        "input_path": str(input_path),
        "notify_email": metadata.get("notify_email", ""),
    }
    _save_job(jobs[job_id])
    _enqueue_job(job_id)

    return JSONResponse({
        "jsonrpc": "2.0", "id": rpc_id,
        "result": _job_to_a2a_task(jobs[job_id]),
    })


# ─── Legacy pipeline (removed 2026-09-18) ──────────────────────
#
# `run_pipeline` — the pre-LangGraph Phase 1-5 implementation, ~1,260 lines —
# lived here with `_load_cpc_context` and `save_json`, which nothing else used.
# USE_LANGGRAPH has been unconditionally True and `run_pipeline` had no caller;
# it was deleted with the `personas` field it was the last user of. The live
# pipeline is _run_langgraph_pipeline above, over graph/main_graph.py.
