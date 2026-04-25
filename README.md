# Patent Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

AI-powered patent novelty analysis. Upload a paper or patent PDF → get a full prior art search + novelty assessment report. Two ways to use it:

- **Web app** (faculty inventors, manual upload)
- **A2A protocol** (other agents / n8n / scripts can call it as a service)

---

## How it works

The analyzer ingests a single PDF and produces a novelty assessment grounded in real prior art. The pipeline has five phases:

### Phase 1 — Invention Detection, Classification & Assignment (IDCA)

PyMuPDF reads the PDF. Deterministic keyword rules run first (`detect_invention`, `classify_document`, `classify_category` — zero LLM tokens). Then a single Gemini 2.5 Pro call performs IDCA:

- **Status determination**: Present (concrete invention) / Implied (concept without implementation) / Absent (no invention — survey, review, etc.)
- **Document type**: invention / design_engineering / literature_review / talks_about_invention_but_no_invention
- **Field classification**: 3-7 technical field labels, best-guess CPC subclass (4-char code like G06N), §101 patent category
- **Summary**: 200-400 word canonical description of the invention

A self-check LLM call verifies the IDCA output against the source PDF. If the status is Absent or the doc type is `talks_about_invention_but_no_invention`, the pipeline exits early.

**Persona crafting**: For Present documents, a single LLM call rewrites 8 generic expert personas into domain-specific versions using the detected fields, CPC subclass, and a CPC taxonomy reference (`cpc_reference.json`, 25 subclasses). These personas are threaded through all downstream LLM calls. The 8 roles:

| Role | Purpose |
|---|---|
| `landscape` | Identifies innovation axes — technical dimensions with deliberate design choices |
| `technology` | Enumerates known approaches per axis from the literature |
| `reviewer` | Reviews checklists for specificity, gaps, overlaps, weight calibration |
| `decompose` | (reserved) Patent claim element extraction |
| `checklist` | Converts innovation analysis into testable evaluation criteria |
| `plan` | Designs search strategies with anchor/expansion terms |
| `evaluate` | Side-by-side prior art comparison with evidence requirements |
| `summary` | Explains results to faculty inventors in their own vocabulary |

### Phase 2 — Expert-Driven Innovation Analysis

Six sequential LLM steps build the evaluation framework:

1. **Scan innovation landscape** (`scan_innovation_landscape`) — Identifies 3-7 innovation axes: technical dimensions where the paper made deliberate choices among known alternatives. Each axis maps to a CPC group. Uses the full PDF as native multimodal input.

2. **Expand technology choices** (`expand_technology_choices`, sequential per axis) — For each axis, enumerates 4-8 known approaches from the literature (including ones the paper did NOT cite), identifies which approach the paper chose, and states what makes it distinct. Sequential execution avoids Gemini API rate limits.

3. **Determine patent types** (`determine_patent_types`) — Classifies which §101 types apply (Process / Machine / Manufacture / Composition / Design) based on the invention summary and technology choices.

4. **Generate checklist per type** (`generate_checklist_for_type`, parallel per type) — For each applicable patent type, generates 10-15 evaluation criteria. Each criterion:
   - Tests ONE specific technical choice (not "uses ML" but "uses stop-gradient on the context branch rather than GRL")
   - Includes `known_approaches` so evaluators can distinguish Present (exact match) from Partial (same category, different method)
   - Has a weight reflecting novelty contribution and a 3-level scale (0=Absent, 1=Partial, 2=Present) with specific definitions

5. **Review checklist** (`review_checklist`) — Expert review merges duplicates across patent types, tightens vague items, fills coverage gaps, splits multi-test items, and normalizes weights (sum ≈ 1.0, no single item > 0.15).

6. **Generate search queries** (`generate_search_queries`) — Converts the final checklist into search groups. Each checklist item gets 2-3 query formulations: paper's exact terminology, synonyms/alternatives, and adjacent-field targeting. Grouped by innovation axis with anchor and expansion terms.

### Phase 3 — Multi-Channel Recall

Five channels run concurrently via `asyncio.gather`:

- SerpAPI Google Patents
- SerpAPI Google Scholar
- Semantic Scholar (search + recommendations + references + citations)
- OpenAlex
- arXiv

SerpAPI is globally throttled (1 in-flight, 1.5s cooldown). Each channel returns `(candidates, error)` — channel failures don't kill the pipeline. The pool layer deduplicates by stable identity (DOI / arXiv ID / patent number / title hash) and applies a √N consensus bonus for cross-channel hits.

**Adaptive refinement**: After each query, quick-rerank with sentence-transformers. If a group's top similarity is ≥ 0.65, stop early. If a group ends weak (< 0.50), `refine_search_query` asks the LLM to diagnose why queries failed and propose refined versions.

### Phase 3b — Semantic Ranking & Self-Citation Filter

`sentence-transformers` (all-MiniLM-L6-v2) reranks all pooled candidates against the invention fingerprint. A three-layer self-citation filter removes the source paper:

1. **Identity match**: arXiv ID or DOI extracted from the source PDF
2. **Title similarity**: Token Jaccard similarity ≥ 0.75 (with stop word filtering)
3. **Post-eval backstop**: Phase 4 drops any candidate scoring ≥ 95% with ≥ 60% title overlap

Top 30 candidates have their PDFs downloaded. Up to 20 go to deep evaluation.

### Phase 4 — Deep Evaluation

`evaluate_batch` runs up to 20 evaluations with `max_concurrent=2`. For each candidate:

- The LLM sees **both the source PDF and the candidate PDF** as native multimodal inputs (no text extraction) plus the full weighted checklist with known_approaches and 3-level scales
- First checks: is this the same document as the source? (`is_source_duplicate`)
- Then scores each criterion: 0 (Absent), 1 (Partial), 2 (Present) with mandatory `evidence_quote` for scores ≥ 1
- When a PDF can't be downloaded (paywall), falls back to abstract-only evaluation

**Dual scoring**:
- **CSS** (Conservative Similarity Score): unknown criteria scored as 0
- **EWSS** (Evidence-Weighted Similarity Score): unknown criteria excluded from denominator
- Low-sample penalty: when EWSS denominator < 5, `adjusted_ewss = ewss × min(1.0, denom_count / 5)`

`generate_overall_summary` writes a plain-English novelty assessment structured for faculty inventors (not patent lawyers): what you invented, what's already out there, what appears genuinely new, honest assessment, suggested next steps.

### Phase 5 — Report Generation

Interactive HTML report uploaded to GCS with:
- Confidence banner (high/medium/low) based on entropy profile
- Per-document cards showing CSS + EWSS + evidence quotes
- SSR criteria with weights and 3-level scale definitions
- Exact Find section for title-matched results
- Full event timeline (click any LLM event to see complete prompts and responses)

### Observability — Entropy Profile

Every pipeline run computes deterministic grounding metrics:

- **SSR grounding**: evidence coverage, weight concentration
- **Eval grounding**: average denominator coverage, average evidence density, low-confidence doc count
- **Overall confidence**: high (all metrics above threshold) / medium / low

These are stored in `results.json` as `entropy_profile` and displayed in the report's confidence banner with specific degradation points when confidence is not high.

### Retry & Error Handling

- All LLM calls (`call_llm`, `call_llm_with_pdfs`) have exponential backoff retry via `tenacity` (up to 4 attempts, randomized wait 1-60s) for 429/503/500 and timeout errors
- Critical steps (IDCA summary) are wrapped in `with_harness`: self-check → if hallucination detected, retry with feedback injected
- Failed steps emit LLM-summarized `failure_reason` to the event timeline
- If recall returns zero candidates, the pipeline marks the job `failed_recall` and does NOT call the novelty-summary LLM

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
│  REST  /analyze  /analyze-gcs  /status/{id}  /events/{id}      │
│        /report/{id}  /results/{id}  /upload-url  /jobs         │
│  A2A   /.well-known/agent-card.json  /a2a                      │
└────┬───────────────────────────────────────────────────────────┘
     │
     │  Phase 1: IDCA (Gemini 2.5 Pro, thinking 4096)
     │           + persona crafting (8 domain-specific roles)
     │
     │  Phase 2: Innovation landscape → technology choices →
     │           patent types → checklist → review → search queries
     │           (Gemini 2.5 Pro, thinking 2048-8192)
     │
     │  Phase 3: 5-channel parallel recall + adaptive refinement
     ▼
┌──────────────────────────────────────────────────────────────────┐
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│   │ SerpAPI      │  │ SerpAPI      │  │ Semantic Scholar     │   │
│   │ Google       │  │ Google       │  │ search + refs +      │   │
│   │ Patents      │  │ Scholar      │  │ citations + recs     │   │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────────────┘   │
│          │                 │                 │                   │
│   ┌──────┴───────┐  ┌──────┴───────┐                             │
│   │ OpenAlex     │  │ arXiv API    │                             │
│   │ /works       │  │ HTTPS Atom   │                             │
│   └──────┬───────┘  └──────┬───────┘                             │
│          └─────────────────┴────────┬────────────────────────┘   │
│                                     ▼                            │
│                  ┌──────────────────────────┐                    │
│                  │ Pool: dedupe + consensus │                    │
│                  │ sentence-transformers    │                    │
│                  │ rerank → top 30 → DL PDFs│                    │
│                  └────────────┬─────────────┘                    │
└──────────────────────────────┼───────────────────────────────────┘
                               ▼
   Phase 3b: self-citation filter (identity + Jaccard + post-eval)
   Phase 4:  deep eval (Gemini, thinking 8192, max 2 concurrent)
             dual scoring (CSS + EWSS)
   Phase 5:  HTML report + entropy profile → GCS
```

**Stack**: FastAPI · google-genai (Vertex AI Gemini 2.5 Pro w/ extended thinking) · tenacity (retry w/ backoff) · 5 recall channels (SerpAPI Patents, SerpAPI Scholar, Semantic Scholar, OpenAlex, arXiv) · sentence-transformers (local rerank) · PyMuPDF · a2a-sdk · GCS · Vite + TypeScript frontend with Express IAM proxy

## API Endpoints

### REST

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Multipart upload (≤32 MB), returns `{job_id}` |
| `GET` | `/upload-url?filename=&content_type=` | Get a v4 signed PUT URL for direct GCS upload (large files) |
| `POST` | `/analyze-gcs` | Start analysis from a `gs://` URI uploaded via signed URL |
| `GET` | `/status/{job_id}` | Job state + per-phase data + stale heartbeat warning |
| `GET` | `/events/{job_id}?since=N` | Event timeline (incremental) |
| `GET` | `/report/{job_id}` | Generated HTML report (falls back to GCS) |
| `GET` | `/results/{job_id}` | Raw `results.json` |
| `GET` | `/jobs` | List all jobs (scans GCS) |
| `DELETE` | `/jobs/{job_id}` | Remove job from memory + local + GCS |
| `POST` | `/jobs/cleanup?days=7` | Delete jobs older than N days |

### A2A protocol

Agent Card at `GET /.well-known/agent-card.json`.

JSON-RPC at `POST /a2a`. Methods:

| Method | Description |
|---|---|
| `agent.getCard` | Returns the agent card |
| `message/send` | Start analysis (send PDF as FilePart) or poll status |
| `tasks.get` / `tasks/get` | Get a task by `taskId` |

#### Example: send a PDF via A2A

```bash
BYTES=$(base64 -w 0 paper.pdf)

curl -X POST https://patent-analyzer-2mk262glgq-uw.a.run.app/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "id": "my-task-id",
      "message": {
        "role": "user",
        "parts": [
          {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "'"$BYTES"'"}}
        ]
      }
    }
  }'
```

#### Example: poll status

```bash
curl -X POST https://patent-analyzer-2mk262glgq-uw.a.run.app/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "2",
    "method": "tasks/get",
    "params": {"taskId": "my-task-id"}
  }'
```

## Deployment (GCP Cloud Run)

```bash
gcloud auth login
gcloud config set project aime-hello-world

# Deploy backend + frontend
./deploy/deploy.sh both
```

### Cloud Run config

| Parameter | Value | Reason |
|---|---|---|
| Region | us-west1 | Same region as Vertex AI Gemini |
| Memory | 4 GiB | sentence-transformers + PDF parsing |
| Timeout | 900 s | Full pipeline can take 10-15 min |
| Concurrency | 1 | Long pipeline + poll on same instance caused zombie jobs |
| Max instances | 3 | Cost cap |
| LLM auth | Service account → Vertex AI | No API key needed |

## Environment Variables (`.env.yaml`)

| Variable | Default | Description |
|---|---|---|
| `GC_PROJECT` | `aime-hello-world` | GCP project for Vertex AI |
| `LLM_MODEL` | `gemini-2.5-pro` | Gemini model ID |
| `SERPAPI_KEY` | — | SerpAPI key for patent + Scholar search |
| `OPENALEX_KEY` | — | OpenAlex key (free, optional) |
| `OUTPUT_DIR` | `/tmp/outputs` | Local job dir (Cloud Run ephemeral) |
| `GCS_BUCKET` | `aime-hello-world-amie-uswest1` | Persistent storage bucket |
| `GCS_PREFIX` | `patent-analyzer/jobs/` | Path prefix in bucket |

## Local development

```bash
# Backend
pip install -e ".[dev]"
gcloud auth application-default login
export SERPAPI_KEY=...
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install
BACKEND_URL=http://localhost:8000 BACKEND_ENV=dev npm run serve
```

## Per-job storage layout

```
gs://aime-hello-world-amie-uswest1/patent-analyzer/jobs/{job_id}/
├── state.json          # full job state + events with complete LLM prompts
├── results.json        # scored results + entropy_profile
├── report.html         # interactive HTML report
└── upload/{filename}   # original PDF (signed-URL uploads only)
```
