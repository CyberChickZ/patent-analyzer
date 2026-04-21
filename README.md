# Patent Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

AI-powered patent novelty analysis. Upload a paper or patent PDF → get a full prior art search + novelty assessment report. Two ways to use it:

- **Web app** (faculty inventors, manual upload)
- **A2A protocol** (other agents / n8n / scripts can call it as a service)

---

## How it works

The analyzer ingests a single PDF (a paper or a patent) and produces a novelty assessment grounded in real prior art. Internally it runs a five-phase pipeline:

1. **Detect & summarize** — PyMuPDF extracts page 1 only for Phase 1 (abstract + intro live there; keeps the LLM focused on what the authors say they built, not on experimental tables). Deterministic keyword rules classify document type (patent / paper) and §101 category. Gemini 2.5 Pro then writes a faithful summary. A self-check LLM call verifies the summary against the source and emits a `quality_warning` if anything is hallucinated.
2. **Decompose & plan** — The invention is broken into 8–20 atomic elements, a 20–30 item testable checklist is generated, and a search plan (atoms × groups with anchor and expansion terms) is produced. Plan self-check is **retry-enabled**: if the reviewer flags a hallucinated atom (e.g. "the paper mentions X" when it doesn't), the plan is regenerated once with the reviewer's feedback injected into the prompt. This prevents one fake atom from poisoning all five search groups downstream.
3. **Multi-channel recall (parallel)** — Five channels run concurrently via `asyncio.gather`: SerpAPI Google Patents, SerpAPI Google Scholar, Semantic Scholar, OpenAlex, arXiv. Each returns `(candidates, error)`; per-query errors that didn't kill the channel are downgraded from `channel_error` to a compact `query_partial` note (no log spam). SerpAPI is globally throttled to 1 in-flight request with a 1.5 s cooldown to stay under the account quota. The pool layer deduplicates candidates by stable identity (DOI / arXiv ID / patent number / title hash) and boosts cross-channel consensus.
4. **Semantic rerank & self-citation filter (Phase 3b)** — `sentence-transformers` (all-MiniLM-L6-v2) reranks the pool against the invention fingerprint. A three-layer self-citation filter then removes the source paper from the results: (A) arXiv ID / DOI identity match extracted from the source PDF, (B) title token similarity ≥ 0.75, (C) a Phase 4 post-eval backstop that drops any candidate scoring ≥ 95 % with ≥ 60 % title overlap. This matters: without it, the paper can win "92 % prior art" against itself.
5. **Deep eval & report** — Up to 20 top candidates go to Phase 4. For each candidate the LLM sees **both the source PDF and the candidate PDF as native multi-modal inputs** (no text extraction, no truncation, up to 18 MB per file inline) along with the checklist. It explicitly answers `is_source_duplicate: true/false` before scoring. When a candidate PDF can't be downloaded (paywall), a text-fallback path evaluates against the abstract and marks the result with an "abstract only" badge. Post-eval, docs with zero matches are dropped (noise, not prior art). Phase 5 generates an interactive HTML report with a feedback form and uploads everything to GCS.

Throughout, every LLM call (system prompt, user prompt, response, thinking trace) is captured in `state.json` so the frontend can render a full event timeline — click any step to see exactly what Gemini saw.

**Defensive design principles** informing every phase:
- Refuse-to-hallucinate guards: if recall returns zero candidates, the pipeline marks the job `failed_recall` and does **not** call the novelty-summary LLM. An earlier incident had the summarizer invent three fake US patent numbers when given an empty matches list.
- Honest failure signals over silent retries: a failed step emits a human-readable `failure_reason` (LLM-summarized from the raw error) to the event timeline. No hidden retries that mask real problems.
- Truth-grounded output: the novelty summary is only allowed to cite documents that actually appear in the scoring report, and the LLM is instructed to flag self-duplicates explicitly.

---

## Live URLs

| Component | URL |
|---|---|
| Web app | https://patent-analyzer-frontend-2mk262glgq-uw.a.run.app |
| Backend API | https://patent-analyzer-2mk262glgq-uw.a.run.app |
| Agent Card (A2A discovery) | https://patent-analyzer-2mk262glgq-uw.a.run.app/.well-known/agent-card.json |
| A2A JSON-RPC endpoint | `POST https://patent-analyzer-2mk262glgq-uw.a.run.app/a2a` |

## Architecture

```
┌──────────────────────┐                ┌─────────────────────────┐
│  Web frontend        │                │  Other agents / n8n     │
│  (Vite + TS)         │                │  (A2A clients)          │
│  Cloud Run            │                │                         │
└──────────┬───────────┘                └─────────────┬───────────┘
           │ HTTPS (IAM proxy)                        │ JSON-RPC over HTTPS
           ▼                                          ▼
┌────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI)  ·  Cloud Run  ·  us-west1                  │
│  app/main.py · run_pipeline()                                  │
│                                                                │
│  REST  /analyze[?evolve=true]  /analyze-gcs[?evolve=true]      │
│        /status/{id}  /events/{id}  /report/{id}  /results/{id} │
│        /upload-url   /jobs   /jobs/{id}                        │
│  A2A   /.well-known/agent-card.json                            │
│        /a2a   (message/send, tasks.get, agent.getCard)         │
└────┬───────────────────────────────────────────────────────────┘
     │
     │  Phase 1-2: Gemini 2.5 Pro w/ thinking budget (4096-8192 tok)
     │             summarize · decompose · checklist · plan
     │
     │  Phase 3: Multi-channel parallel recall + pool + rerank
     ▼
┌──────────────────────────────────────────────────────────────────┐
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│   │ SerpAPI      │  │ SerpAPI      │  │ Semantic Scholar     │   │
│   │ Google       │  │ Google       │  │ /paper/search        │   │
│   │ Patents      │  │ Scholar      │  │ /recommendations     │   │
│   │              │  │              │  │ /references /citations│  │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────────────┘   │
│          │                 │                 │                   │
│   ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐           │
│   │ OpenAlex     │  │ arXiv API    │  │ (evolve mode)│           │
│   │ /works       │  │              │  │ S2 fan-out   │           │
│   │ + concepts   │  │ HTTPS Atom   │  │ from top hit │           │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│          │                 │                 │                   │
│          └─────────────────┴────────┬────────┘                   │
│                                     ▼                            │
│                  ┌──────────────────────────┐                    │
│                  │ Pool: dedupe by          │                    │
│                  │  DOI / arxiv id /        │                    │
│                  │  pub_num / title hash    │                    │
│                  │ + sqrt(N) consensus boost│                    │
│                  └────────────┬─────────────┘                    │
│                               ▼                                  │
│                  ┌──────────────────────────┐                    │
│                  │ sentence-transformers    │                    │
│                  │ all-MiniLM-L6-v2         │                    │
│                  │ rerank vs invention      │                    │
│                  │ fingerprint → top 30     │                    │
│                  └────────────┬─────────────┘                    │
└─────────────────────────────────┼────────────────────────────────┘
                                  │
                                  ▼
   Phase 3b: filter self-citations · download PDFs (top 30)
   Phase 4: per-doc deep eval (Gemini thinking budget 8192) ×5 concurrent
            (evolve mode = elastic batches of 5 with mid-loop review)
   Phase 5: HTML report → GCS

   ┌──────┐  state.json + report.html persisted to
   │ GCS  │  gs://aime-hello-world-amie-uswest1/patent-analyzer/jobs/{id}/
   └──────┘
```

**Stack**: FastAPI · google-genai (Vertex AI Gemini 2.5 Pro w/ extended thinking) · 5 parallel recall channels (SerpAPI Patents, SerpAPI Scholar, Semantic Scholar, OpenAlex, arXiv) · sentence-transformers (local rerank) · PyMuPDF · a2a-sdk · GCS · Vite + TypeScript frontend with Express IAM proxy

## Pipeline — 5 phases

| Phase | Name | What happens |
|---|---|---|
| 1 | Invention Detection | Read PDF, detect invention (deterministic keyword), classify document type & §101 category (deterministic), LLM-summarize with **thinking budget 4096**. `try_step` wraps the call: failures emit `failure_reason` (LLM-summarized human-readable cause) instead of silent retry. |
| 2 | Decomposition | LLM-decompose elements (thinking 4096), generate 20-30 testable checklist (thinking 4096), plan search atoms + groups (thinking 4096). |
| 3 | Multi-channel recall | **5 channels in parallel** via `asyncio.gather`: SerpAPI Patents, SerpAPI Scholar, Semantic Scholar, OpenAlex, arXiv. Each channel returns `(candidates, error)` — channel-level failures are surfaced as `channel_error` events with `failure_reason`, the pipeline keeps going on the surviving channels. Pool layer dedupes by stable identity (DOI / arXiv ID / patent number / title hash) and applies a √N consensus bonus. |
| 3b | Semantic Ranking | sentence-transformers reranks the pooled candidates against the invention fingerprint (no PDFs yet), filter self-citations, then download top 30 PDFs only. |
| 4 | Deep Evaluation | LLM evaluates each top doc against the checklist with **thinking budget 8192**, max 5 concurrent. In evolve mode, evaluates in elastic batches of 5 (cap 30) with a reviewer between batches. |
| 5 | Report | Generate HTML report, upload to GCS. |

Every LLM call is captured (system + user prompt + response + **thought summary** when thinking is enabled) and persisted to `state.json` in GCS, so the frontend can show a complete event timeline. Click any event to see the exact prompt, response, and reasoning trace.

## Evolve mode (`?evolve=true`)

Opt-in per request: `POST /analyze?evolve=true` or `POST /analyze-gcs?evolve=true`. When enabled, the pipeline runs an additional **full-context reviewer** after each phase output. The reviewer sees both the original input (e.g. paper text) and the produced output (e.g. invention summary) and decides whether the output is good enough for the next step.

| Phase | What evolve mode adds |
|---|---|
| 1 | Reviewer judges the invention summary against the full PDF text → `review_pass` / `review_warning` events. No backtracking — warnings are dev signal only. |
| 2 | Reviewer judges the 20-30 item checklist for goldilocks specificity → warning events. |
| 3 | Reviewer judges the pooled candidate set. If `do_more`, **fan out via Semantic Scholar** from the top pooled hit: pull `recommendations` + `references` + `citations` (1-hop citation graph) and re-pool. Bounded to one expansion round. |
| 4 | Evaluates in **elastic batches of 5** (up to 30 docs total). Reviewer between batches can request `elastic_stop` to finish early when the top hits are clearly irrelevant. |

Failure observability is **always on** regardless of evolve mode: every failure point produces an LLM-summarized `failure_reason` in plain language so the developer can see in the timeline why a step did not produce useful output (e.g. *"All 5 recall channels returned 0 candidates. SerpAPI hit HTTP 401 — quota likely exhausted. Semantic Scholar / OpenAlex / arXiv each returned 0 — the query is genuinely too niche or the invention summary is too vague."*).

## API Endpoints

### REST

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Multipart upload (≤32 MB), returns `{job_id}` |
| `GET` | `/upload-url?filename=&content_type=` | Get a v4 signed PUT URL for direct GCS upload (large files) |
| `POST` | `/analyze-gcs` | Start analysis from a `gs://` URI uploaded via signed URL |
| `GET` | `/status/{job_id}` | Job state + per-phase data + zombie detection |
| `GET` | `/events/{job_id}?since=N` | Event timeline (incremental). Each event has phase, kind, message, optional payload (LLM prompts/responses) |
| `GET` | `/report/{job_id}` | Generated HTML report (falls back to GCS) |
| `GET` | `/results/{job_id}` | Raw `results.json` |
| `GET` | `/jobs` | List all jobs (scans GCS) |
| `DELETE` | `/jobs/{job_id}` | Remove job from memory + local + GCS |
| `POST` | `/jobs/cleanup?days=7` | Delete jobs older than N days |

### A2A protocol

Agent Card at `GET /.well-known/agent-card.json` (also `/agent-card.json`).

JSON-RPC at `POST /a2a`. Methods:

| Method | Description |
|---|---|
| `agent.getCard` | Returns the agent card |
| `message/send` | Start an analysis (skill `patent_analyze`) or poll status (skill `patent_status`) |
| `tasks.get` / `tasks/get` | Get a task by `taskId` |
| `tasks.create` | Compatibility shim |

#### Example: send a PDF via A2A

```bash
# Base64-encode the PDF
BYTES=$(base64 -w 0 paper.pdf)

curl -X POST https://patent-analyzer-2mk262glgq-uw.a.run.app/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "metadata": {"skill_id": "patent_analyze"},
      "message": {
        "role": "user",
        "parts": [
          {"file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "'"$BYTES"'"}}
        ]
      }
    }
  }'
# → {"jsonrpc":"2.0","id":"1","result":{"id":"abc12345","status":{"state":"submitted",...}}}
```

#### Example: poll status via A2A

```bash
curl -X POST https://patent-analyzer-2mk262glgq-uw.a.run.app/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "2",
    "method": "tasks/get",
    "params": {"taskId": "abc12345"}
  }'
```

The returned task `metadata` includes `phase`, `phases` (per-phase data), and once completed, `report_url` and `results_url`.

### REST example (small file)

```bash
curl -X POST https://patent-analyzer-2mk262glgq-uw.a.run.app/analyze \
  -F "file=@paper.pdf"
# → {"job_id": "a1b2c3d4", "status": "queued"}

curl https://patent-analyzer-2mk262glgq-uw.a.run.app/status/a1b2c3d4
curl https://patent-analyzer-2mk262glgq-uw.a.run.app/events/a1b2c3d4
open  https://patent-analyzer-2mk262glgq-uw.a.run.app/report/a1b2c3d4
```

### Large file (>25 MB) — signed URL flow

```bash
# 1. Get signed PUT URL
curl "https://.../upload-url?filename=big.pdf&content_type=application/pdf"
# → {job_id, signed_url, gcs_uri, ...}

# 2. PUT directly to GCS (bypasses Cloud Run 32 MB limit)
curl -X PUT -H "Content-Type: application/pdf" --data-binary @big.pdf "$SIGNED_URL"

# 3. Start analysis from GCS URI
curl -X POST -H "Content-Type: application/json" \
  -d '{"job_id":"...","gcs_uri":"gs://...","filename":"big.pdf"}' \
  https://.../analyze-gcs
```

## Observability

Every job's state is persisted to GCS at `gs://aime-hello-world-amie-uswest1/patent-analyzer/jobs/{job_id}/state.json`. Each `state.json` contains an `events` array with one entry per pipeline step. LLM-call events include the **complete system prompt, complete user prompt, and complete response**, so you can audit and debug what the AI saw and produced at every step.

The frontend renders this as a timeline: click any phase to expand events, click any LLM event to see the full conversation in a modal (with template/data sections color-coded).

## Deployment (GCP Cloud Run)

```bash
gcloud auth login
gcloud config set project aime-hello-world

# Backend
gcloud run deploy patent-analyzer \
  --source . --region us-west1 \
  --service-account amie-backend-sa@aime-hello-world.iam.gserviceaccount.com \
  --allow-unauthenticated --env-vars-file .env.yaml \
  --memory 4Gi --cpu 2 --timeout 900 --concurrency 80 --max-instances 3

# Frontend
cd frontend
gcloud run deploy patent-analyzer-frontend \
  --source . --region us-west1 \
  --service-account amie-frontend-sa@aime-hello-world.iam.gserviceaccount.com \
  --allow-unauthenticated --env-vars-file .env.yaml
```

Or use the helper: `./deploy/deploy.sh [backend|frontend|both]`

### Cloud Run config

| Parameter | Value | Reason |
|---|---|---|
| Region | us-west1 | Same region as Vertex AI Gemini |
| Memory | 4 GiB | sentence-transformers + PDF parsing |
| Timeout | 900 s | Full pipeline can take 5-10 min |
| Concurrency | 80 | Pipeline + poll requests share an instance (avoids zombie jobs) |
| Max instances | 3 | Cost cap |
| LLM auth | Service account → Vertex AI | No API key needed |

## Environment Variables (`.env.yaml`)

| Variable | Default | Description |
|---|---|---|
| `GC_PROJECT` | `aime-hello-world` | GCP project for Vertex AI |
| `LLM_MODEL` | `gemini-2.5-pro` | Gemini model ID |
| `SERPAPI_KEY` | (in .env.yaml) | SerpAPI key for patent + Scholar search |
| `OPENALEX_KEY` | (in .env.yaml) | OpenAlex key (free, optional) |
| `OUTPUT_DIR` | `/tmp/outputs` | Local job dir (Cloud Run ephemeral) |
| `GCS_BUCKET` | `aime-hello-world-amie-uswest1` | Persistent storage bucket |
| `GCS_PREFIX` | `patent-analyzer/jobs/` | Path prefix in bucket |

## Local development

```bash
# Backend
pip install -e ".[dev]"
gcloud auth application-default login   # for Vertex AI
export SERPAPI_KEY=...
uvicorn app.main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend
npm install
BACKEND_URL=http://localhost:8000 BACKEND_ENV=dev npm run serve
# → http://localhost:8080
```

## Per-job storage layout

```
gs://aime-hello-world-amie-uswest1/patent-analyzer/jobs/{job_id}/
├── state.json          # full job state including events with complete LLM prompts
├── results.json        # final scored results
├── report.html         # interactive HTML report
└── upload/{filename}   # original uploaded PDF (only when via signed-URL flow)
```

Local copies during processing live under `OUTPUT_DIR/{job_id}/`.
