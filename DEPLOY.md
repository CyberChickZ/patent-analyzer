# Deployment Guide

## Prerequisites

- `gcloud` CLI authenticated with project `aime-hello-world`
- Service account: `amie-backend-sa@aime-hello-world.iam.gserviceaccount.com`

## Environment Variables

Backend env vars are stored in `backend/.env.yaml` (gitignored). Required keys:

```yaml
SERPAPI_KEY: "..."
GOOGLE_CLOUD_PROJECT: "aime-hello-world"
GCS_BUCKET: "aime-hello-world-amie-uswest1"
SENDGRID_API_KEY: "..."         # optional, for email notifications
SENDGRID_FROM: "noreply@..."    # optional
```

To create/update `.env.yaml`, copy from `backend/.env.yaml.example` (if exists) or ask the team for current values.

**Never commit `.env.yaml`** — it contains API keys.

## Deploy Backend

```bash
cd backend
gcloud run deploy patent-analyzer --source . --region us-west1 \
  --service-account amie-backend-sa@aime-hello-world.iam.gserviceaccount.com \
  --env-vars-file .env.yaml --no-allow-unauthenticated \
  --timeout=1800 --memory=4Gi --cpu=2 --concurrency=1 --max-instances=3
```

## Deploy Frontend

```bash
cd frontend/amie
gcloud run deploy patent-analyzer-frontend --source . --region us-west1 \
  --service-account amie-backend-sa@aime-hello-world.iam.gserviceaccount.com \
  --allow-unauthenticated
```

Frontend is public-facing; auth is handled by Firebase on the client side.

## Architecture

```
User → patent-analyzer-frontend (Express proxy + Vite SPA)
         → patent-analyzer (FastAPI + LangGraph pipeline)
            → Vertex AI (Gemini 2.5 Pro)
            → SerpAPI / Semantic Scholar / OpenAlex / arXiv / BigQuery Patents
            → GCS (reports, job state)
```

## Local Development

```bash
# Backend
cd backend
PYTHONPATH=$(pwd) uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend/amie
BACKEND_URL=http://localhost:8000 BACKEND_ENV=dev npm run serve
```

`BACKEND_ENV=dev` skips IAM token for local backend calls.
