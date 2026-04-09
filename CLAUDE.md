# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Patent novelty analyzer. Upload a paper/patent PDF → 5-phase pipeline produces a prior art report. Two interfaces share the same backend:
- **REST + Web UI** for human users (faculty inventors)
- **A2A JSON-RPC** so other agents (n8n, etc.) can call it as a service

Stack: FastAPI · google-genai (Vertex AI Gemini 2.5 Pro) · SerpAPI · OpenAlex · sentence-transformers · PyMuPDF · a2a-sdk · GCS · Vite/TS frontend with Express IAM proxy.

## Commands

```bash
# Backend (local)
pip install -e ".[dev]"
gcloud auth application-default login          # Vertex AI auth
export SERPAPI_KEY=...
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install
BACKEND_URL=http://localhost:8000 BACKEND_ENV=dev npm run serve   # → :8080

# Full local stack via Docker
docker-compose up --build

# Lint
ruff check .

# Tests
pytest                      # tests/ is currently empty

# Deploy to Cloud Run (us-west1, project aime-hello-world)
./deploy/deploy.sh backend     # rebuilds patent-analyzer
./deploy/deploy.sh frontend    # rebuilds patent-analyzer-frontend
./deploy/deploy.sh both
```

There are also CLI entry points (`patent-search`, `patent-prefilter`, `patent-report`) defined in `pyproject.toml`. These existed before the FastAPI pipeline and are still useful for ad-hoc work but are NOT part of the running service — `app/main.py:run_pipeline` is the source of truth.

## Architecture

### Two execution paths, one pipeline
`app/main.py` exposes both REST endpoints and an A2A JSON-RPC endpoint at `/a2a`. Both end up calling the same `run_pipeline(job_id)` background task. The A2A handler is at `_handle_a2a_send` (decodes base64 PDF from a `FilePart`); the REST handler is `start_analysis` (multipart upload, ≤32 MB) plus `start_analysis_from_gcs` (signed-URL flow for large files). The agent card is served at `/.well-known/agent-card.json`.

### Pipeline (5 phases) — `app/main.py:run_pipeline`
1. **Phase 1 (IDCA)** — read PDF (PyMuPDF, first 10 pages), `detect_invention` (deterministic keyword rules), `classify_document` (deterministic), `summarize_invention` (LLM, harnessed), `classify_category` (deterministic).
2. **Phase 2** — `decompose_invention` (LLM, harnessed) → `generate_checklist` (LLM, 20-30 items) → `plan_delegation` (LLM, harnessed) → `build_all_queries` (deterministic).
3. **Phase 3** — Adaptive SerpAPI search across groups via `serpapi_search`. After each query, quick-rerank with `quick_rerank` (sentence-transformers) and **stop early** when top similarity ≥ 0.65 (`SIM_THRESHOLD`). If a group ends weak (< 0.50, `REFINE_THRESHOLD`), call `refine_search_query` (LLM) for a second-round refined query.
4. **Phase 3b** — Filter self-citations (title token similarity ≥ 0.75 against source manuscript title), rerank ALL by embedding similarity, then download top 30 PDFs only. Cap deep-evaluation candidates at 20.
5. **Phase 4** — `evaluate_batch` runs `evaluate_single_document` (LLM, sees PDF text) with `max_concurrent=5`. Each call evaluates one prior art doc against the full checklist. Then `generate_overall_summary` writes a plain-English novelty assessment.
6. **Phase 5** — `report_generator.generate_html` produces the static HTML, uploaded to GCS.

### Self-checking harness — `with_harness`
Critical LLM steps (summarize, decompose, plan_delegation) are wrapped: call → `self_check` → if `ok=False`, retry the original call **with feedback injected via `_feedback_block`** (issues, suggestion, previous response). The `feedback` arg is threaded through `summarize_invention`, `decompose_invention`, `plan_delegation`, `refine_search_query` — preserve this signature when adding new LLM steps that need self-correction.

### Token budget — `app/llm.py`
Hard rule: only **6** LLM calls in the pipeline (down from 10). Tokens go to PDF analysis, not routing. `detect_invention`, `classify_document`, `classify_category` are deterministic keyword/metadata rules — do NOT replace them with LLM calls. New LLM-using features should justify the token cost.

### LLM call hook + event timeline
Every `call_llm()` invocation passes through `_emit(system, user, response)`. The pipeline installs `llm_hook` via `set_llm_hook` so each call records an `llm_response` event into `job["events"]` with the **complete system prompt, user prompt, and response**. The frontend renders these as a clickable timeline. Set `llm_label(phase, label)` before each call to label what's happening. When adding LLM calls inside `run_pipeline`, always set the label first or events get mis-attributed.

### Job state persistence — three layers
`_save_job` writes to (1) in-memory `jobs` dict, (2) local disk `OUTPUT_DIR/{job_id}/state.json`, (3) GCS at `gs://{GCS_BUCKET}/{GCS_PREFIX}{job_id}/state.json` (best-effort). `_get_job` falls back through memory → disk → GCS. This means a job can be polled across Cloud Run instance restarts. Use `heartbeat()` for cheap progress updates (disk only, no GCS write); use `event()` for state-changing observations that the frontend should see.

### Zombie-job detection
`/status/{job_id}` checks `last_heartbeat`; if a "running" job hasn't updated in >900s the status is force-flipped to "error". Always update `last_heartbeat` in long inner loops (the PDF download loop calls `heartbeat()` per file).

### Frontend proxy — `frontend/server.js`
The frontend is a thin Express server that serves `dist/` and proxies `/api/*` → backend `/`. In production it mints a GCP IAM ID token from the metadata server (`getIdToken`, audience = `BACKEND_URL`) so it can call the **non-public** backend service (`--no-allow-unauthenticated`). Locally, set `BACKEND_ENV=dev` to skip auth. The signed-URL flow (`/api/upload-url` → direct PUT to GCS → `/api/analyze-gcs`) is what makes >32 MB uploads possible — Cloud Run rejects bodies above that limit.

## GCP / deployment

- Project: `aime-hello-world`, region: `us-west1` (same as Vertex AI Gemini)
- Backend SA: `amie-backend-sa@aime-hello-world.iam.gserviceaccount.com`
- Frontend SA: `amie-frontend-sa@aime-hello-world.iam.gserviceaccount.com`
- Backend is `--no-allow-unauthenticated`; frontend SA has `roles/run.invoker` on backend
- Backend Cloud Run config: 4Gi / 2 CPU / 900s timeout / **concurrency 1** / max 3 instances. Concurrency 1 is intentional — the long pipeline + poll requests on the same instance previously caused zombie jobs.
- LLM auth: service account → Vertex AI (`vertexai=True`, location `us-west1`), no API key. The `.env.example` mentioning `LLM_API_KEY` / `LLM_BASE_URL` is stale from the OpenAI-compatible era.
- Secrets live in `.env.yaml` (gitignored): `SERPAPI_KEY`, `OPENALEX_KEY`. There is also a `frontend/.env.yaml` with the prod `BACKEND_URL`.
- Persistent storage: `gs://aime-hello-world-amie-uswest1/patent-analyzer/jobs/{job_id}/` holds `state.json`, `results.json`, `report.html`, and `upload/{filename}` (signed-URL uploads only).

## Conventions / gotchas

- Pipeline step categories are documented in `patent_analyzer/pipeline.py` as a docstring — every step is either `[FIXED]` (deterministic) or `[LLM]`. Maintain this distinction; deterministic should never get an LLM call sneaked in.
- LLM responses that should be JSON are extracted with `re.search(r'\{.*\}', resp, re.DOTALL)` (or `\[.*\]` for arrays). The system prompt always says "Output JSON only" — keep that contract when adding prompts.
- `call_llm` strips ` ```json ` fences automatically.
- `call_llm_with_pdf` extracts the first 10 pages via PyMuPDF and truncates to 60k chars before sending — Gemini doesn't see the raw PDF.
- The license is `PolyForm-Noncommercial-1.0.0` (see `pyproject.toml` and `LICENSE`).
- `patent_analyzer/` package has both library code (`searcher.py`, `semantic_search.py`, `report_generator.py`, `scorer.py`, `query_builder.py`) and standalone script entry points. The FastAPI service imports from this package — don't break those imports when refactoring CLI scripts.
- `SKILL.md` documents an older "skill" workflow (per-task agent files in `eval_tasks/`, checkpoint reviews). The Cloud Run service does NOT follow that flow — `evaluate_batch` runs all evals in-process. Treat `SKILL.md` as historical / reference for the CLI flow, not the live system.
