## Backend Sync & Deployment Guide

### Sync backend branch

In the top-level directory, run:

```bash
bash scripts/sync_backend_amie_branch.sh
````

This syncs the code to `backend-dev`.

Then fetch and check out the remote branch:

```bash
git fetch origin
git branch -r
git checkout -t origin/<branch-name>
```

---

### Google Cloud authentication

```bash
gcloud auth login
gcloud config set project aime-hello-world
gcloud auth application-default login
```

---

## Automated deployment (recommended)

### First-time deployment

```bash
gcloud run deploy amie-backend \
  --source . \
  --region us-west1 \
  --allow-unauthenticated \
  --service-account 839195423256-compute@developer.gserviceaccount.com \
  --env-vars-file .env.yaml
```

---

## Accelerated build (recommended for faster iteration)

Build the image using Cloud Build (or Docker locally) first:

```bash
gcloud builds submit \
  --tag us-west1-docker.pkg.dev/aime-hello-world/cloud-run-source-deploy/amie-backend:latest .
```

This step will:

* Build the image using `Dockerfile`
* Upload it to Artifact Registry
* Output the image URL

Repeated build example:

```bash
gcloud builds submit \
  --tag us-west1-docker.pkg.dev/aime-hello-world/cloud-run-source-deploy/amie-backend:latest .
```

---

## Local dockerized build (high-performance)

```bash
gcloud builds submit \
  --tag us-west1-docker.pkg.dev/aime-hello-world/cloud-run-source-deploy/amie-backend:latest \
  --machine-type=e2-highcpu-8 \
  --build-env-vars=GOOGLE_USE_CLOUD_BUILD_CACHING=true
```

---

## Debugging & logs

Read Cloud Run service logs (errors only):

```bash
gcloud run services logs read amie-backend \
  --project=aime-hello-world \
  --region=us-west1 \
  --limit=100 \
  --level=error
```
