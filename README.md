# Patent Analyzer

Standalone FastAPI web service that analyzes papers and patents for patentability under US patent law (35 USC 102/103), searches prior art, and generates an interactive HTML report.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Web Service                   │
│                      app/main.py                        │
└────────────┬────────────────────────────────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐    ┌──────────────┐
│Anthropic│    │  SerpAPI     │
│  API    │    │ (Patents +   │
│(Claude) │    │  Scholar)    │
└─────────┘    └──────────────┘
     │                │
     └───────┬────────┘
             ▼
  ┌──────────────────────┐     ┌────────────────────────┐
  │ patent_analyzer/     │     │ OpenAlex API (free)    │
  │  query_builder.py    │     │ real abstracts, no key │
  │  scorer.py           │     └────────────────────────┘
  │  semantic_search.py  │◄────────────────────────────
  │  report_generator.py │
  └──────────────────────┘
             │
             ▼
  sentence-transformers (local, all-MiniLM-L6-v2)
  semantic re-ranking, no API call
```

**Stack:** FastAPI · Anthropic API · SerpAPI · OpenAlex · sentence-transformers · PyMuPDF

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/patent-analyzer.git
cd patent-analyzer

# 2. Set environment variables
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and SERPAPI_KEY

# 3. Run
docker compose up
```

Open http://localhost:8000 — upload a PDF, click Analyze, wait for the report.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Upload UI (HTML) |
| `POST` | `/analyze` | Upload PDF, start analysis → returns `{ job_id }` |
| `GET` | `/status/{job_id}` | Poll job status + per-phase breakdown |
| `GET` | `/report/{job_id}` | Serve generated HTML report |
| `GET` | `/results/{job_id}` | Raw `results.json` (full data) |
| `GET` | `/jobs` | List all jobs in current session |

### Example

```bash
# Upload and start
curl -X POST http://localhost:8000/analyze \
  -F "file=@paper.pdf"
# → {"job_id": "a1b2c3d4", "status": "queued"}

# Poll until completed
curl http://localhost:8000/status/a1b2c3d4

# Open report
open http://localhost:8000/report/a1b2c3d4
```

## Pipeline — 5 Phases

| Phase | Name | Type | What happens |
|-------|------|------|-------------|
| 1 | IDCA | LLM | Invention detection · document classification · summary · 35 USC §101 category |
| 2 | Decomposition | LLM + FIXED | Independent/dependent claim scope · 20-30 item checklist · USPTO search delegation plan |
| 3 | Search | FIXED | SerpAPI → Google Patents + Google Scholar · PDF download · OpenAlex abstracts · sentence-transformer re-ranking |
| 4 | Deep Evaluation | LLM | 1:1 per-document checklist evaluation (max 20 docs, up to 5 concurrent) · 102 anticipation + 103 obviousness |
| 5 | Report | FIXED | Score computation · risk classification · static HTML report generation |

**Key design principle:** LLM responses begin with a decision word (`present`/`absent`/`implied`, `patent`/`paper`, `Process`/`Machine`/…) enabling deterministic routing without parsing free text.

### Fixed vs LLM breakdown

| Step | Type | Module |
|------|------|--------|
| Query generation | FIXED | `patent_analyzer/query_builder.py` |
| SerpAPI search + retry | FIXED | `patent_analyzer/searcher.py` |
| Result deduplication | FIXED | `patent_analyzer/searcher.py` |
| PDF download | FIXED | `patent_analyzer/searcher.py` |
| Semantic re-ranking | FIXED | `patent_analyzer/semantic_search.py` |
| Score computation | FIXED | `patent_analyzer/scorer.py` |
| Risk classification | FIXED | `patent_analyzer/scorer.py` |
| HTML report generation | FIXED | `patent_analyzer/report_generator.py` |
| Invention detection | LLM | `app/llm.py` |
| Document summarization | LLM | `app/llm.py` |
| Checklist generation | LLM | `app/llm.py` |
| Delegation planning | LLM | `app/llm.py` |
| Per-document evaluation | LLM | `app/llm.py` |
| Overall summary | LLM | `app/llm.py` |

## GCP Cloud Run Deployment

```bash
# Deploy (builds, pushes to GCR, creates secrets, deploys)
./deploy/deploy.sh YOUR_GCP_PROJECT_ID

# Get service URL
gcloud run services describe patent-analyzer \
  --project=YOUR_GCP_PROJECT_ID \
  --region=us-central1 \
  --format='value(status.url)'
```

The deploy script:
- Builds and pushes Docker image to GCR
- Creates `anthropic-api-key` and `serpapi-key` in GCP Secret Manager
- Deploys with 4 GiB RAM, 2 vCPU, 15-min timeout, max 3 instances
- Sets `OUTPUT_DIR=/tmp/outputs` (ephemeral; use GCS bucket for persistence)

### Cloud Run configuration highlights

| Parameter | Value | Reason |
|-----------|-------|--------|
| Memory | 4 GiB | sentence-transformers model + PDF parsing |
| Timeout | 900 s | Full pipeline can take 10-15 min |
| Concurrency | 1 | CPU-bound embedding; scale via instances |
| Max instances | 3 | Cost cap |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key ([console.anthropic.com](https://console.anthropic.com)) |
| `SERPAPI_KEY` | Yes | — | SerpAPI key ([serpapi.com](https://serpapi.com)) for patent + paper search |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-20250514` | Claude model ID |
| `OUTPUT_DIR` | No | `./outputs` | Directory for job artifacts (phase JSONs, PDFs, report) |

## Local Development (without Docker)

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
export SERPAPI_KEY=...
uvicorn app.main:app --reload
```

## Output Structure

Each job produces a directory under `OUTPUT_DIR/{job_id}/`:

```
outputs/
└── a1b2c3d4/
    ├── paper.pdf          # uploaded file
    ├── phase1.json        # IDCA results
    ├── phase2.json        # checklist + delegation plan
    ├── queries.json       # generated search queries
    ├── phase3_search.json # raw SerpAPI results
    ├── papers/            # downloaded PDFs
    ├── results.json       # final merged + scored results
    └── report.html        # interactive HTML report
```
