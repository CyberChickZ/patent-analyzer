# Patent Analyzer

AI-powered patent novelty analysis. Upload a paper or patent PDF → get a full prior art search + novelty assessment report. Two ways to use it:

- **Web app** (faculty inventors, manual upload)
- **A2A protocol** (other agents / n8n / scripts can call it as a service)

## Live URLs

| Component | URL |
|---|---|
| Web app | https://patent-analyzer-frontend-839195423256.us-west1.run.app |
| Backend API | https://patent-analyzer-839195423256.us-west1.run.app |
| Agent Card (A2A discovery) | https://patent-analyzer-839195423256.us-west1.run.app/.well-known/agent-card.json |
| A2A JSON-RPC endpoint | `POST https://patent-analyzer-839195423256.us-west1.run.app/a2a` |

## Architecture

```
┌──────────────────────┐         ┌─────────────────────────┐
│  Web frontend        │         │  Other agents / n8n     │
│  (Vite + TS)         │         │  (A2A clients)          │
│  Cloud Run           │         │                         │
└──────────┬───────────┘         └─────────────┬───────────┘
           │ HTTPS (proxy)                     │ JSON-RPC over HTTPS
           ▼                                   ▼
┌──────────────────────────────────────────────────────────┐
│  Backend (FastAPI)  ·  Cloud Run  ·  us-west1            │
│  app/main.py                                             │
│                                                          │
│  REST  /analyze    /status/{id}   /events/{id}           │
│        /report/{id} /results/{id} /jobs                  │
│        /upload-url /analyze-gcs                          │
│  A2A   /.well-known/agent-card.json                      │
│        /a2a   (message/send, tasks.get, agent.getCard)   │
└────────────┬─────────────────────────────────────────────┘
             │
   ┌─────────┼──────────┬──────────────┬────────────┐
   ▼         ▼          ▼              ▼            ▼
┌──────┐ ┌────────┐ ┌─────────┐  ┌──────────┐ ┌────────┐
│ GCS  │ │Vertex  │ │ SerpAPI │  │ OpenAlex │ │ sent.- │
│bucket│ │AI/Gemini│ │ Patents │ │  free    │ │trans.  │
│state │ │ 2.5 Pro │ │+Scholar │ │abstracts │ │embed   │
└──────┘ └────────┘ └─────────┘  └──────────┘ └────────┘
```

**Stack**: FastAPI · google-genai (Vertex AI Gemini 2.5 Pro) · SerpAPI · OpenAlex · sentence-transformers · PyMuPDF · a2a-sdk · GCS · Vite + TypeScript frontend with Express IAM proxy

## Pipeline — 5 phases

| Phase | Name | What happens |
|---|---|---|
| 1 | Invention Detection | Read PDF, detect invention, classify document type, LLM-summarize, classify §101 category, **self-check the summary** |
| 2 | Decomposition | LLM-decompose into atomic elements, generate 20-30 testable checklist, plan search atoms + groups, **self-check the plan** |
| 3 | Prior Art Search | SerpAPI Google Patents + Scholar, **adaptive** — quick rerank after each query, skip remaining group queries when top similarity ≥ 0.65 |
| 3b | Semantic Ranking | sentence-transformers reranks ALL results by title+snippet (no PDFs), filter self-citations, then download top 30 PDFs only |
| 4 | Deep Evaluation | LLM evaluates each top doc against checklist (max 20 docs, 5 concurrent), score + sort |
| 5 | Report | Generate final HTML report, upload to GCS |

Every LLM call is captured (system + user prompt + response) and persisted to `state.json` in GCS, so the frontend can show a complete event timeline. Click any event to see the exact prompt and the LLM's response.

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

curl -X POST https://patent-analyzer-839195423256.us-west1.run.app/a2a \
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
curl -X POST https://patent-analyzer-839195423256.us-west1.run.app/a2a \
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
curl -X POST https://patent-analyzer-839195423256.us-west1.run.app/analyze \
  -F "file=@paper.pdf"
# → {"job_id": "a1b2c3d4", "status": "queued"}

curl https://patent-analyzer-839195423256.us-west1.run.app/status/a1b2c3d4
curl https://patent-analyzer-839195423256.us-west1.run.app/events/a1b2c3d4
open  https://patent-analyzer-839195423256.us-west1.run.app/report/a1b2c3d4
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
