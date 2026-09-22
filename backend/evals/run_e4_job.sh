#!/usr/bin/env bash
# F1: run the open-world eval as a Cloud Run Job in the production environment.
# Prereq: `gcloud auth login` (personal account). Builds the current backend
# image once, then runs the harness over all gold-bearing queries.
set -euo pipefail
cd "$(dirname "$0")/.."
REGION=us-west1; PROJECT=aime-hello-world
SA=amie-backend-sa@$PROJECT.iam.gserviceaccount.com
BUCKET=gs://aime-hello-world-amie-uswest1/evals/e4
LIMIT=${1:-35}

gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT/cloud-run-source-deploy/patent-analyzer-eval:e4" --project $PROJECT --quiet .
gcloud run jobs delete e4-openworld --region $REGION --project $PROJECT --quiet 2>/dev/null || true
gcloud run jobs create e4-openworld --region $REGION --project $PROJECT \
  --image "$REGION-docker.pkg.dev/$PROJECT/cloud-run-source-deploy/patent-analyzer-eval:e4" \
  --service-account $SA --memory 4Gi --cpu 2 --task-timeout 3h --max-retries 0 \
  --env-vars-file .env.yaml \
  --set-env-vars "E4_BUNDLE=$BUCKET/bundle.tar.gz,E4_UPLOAD=$BUCKET/results,E4_RUN_DIR=/tmp/e4,LLM_CACHE_DIR=/tmp/llm_cache,SERPAPI_MAX_CALLS_PER_JOB=2" \
  --command python3 --args="^|^evals/openworld_eval.py|--n|50|--stage|pipeline|--limit|$LIMIT|--concurrency|2"
gcloud run jobs execute e4-openworld --region $REGION --project $PROJECT --wait
echo "results: $BUCKET/results/pipeline_result.json"
