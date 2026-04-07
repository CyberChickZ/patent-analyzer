#!/bin/bash
# Deploy Patent Analyzer backend + frontend to GCP Cloud Run
# Usage: ./deploy/deploy.sh [backend|frontend|both]
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project aime-hello-world

set -euo pipefail

PROJECT="aime-hello-world"
REGION="us-west1"
BACKEND_SA="amie-backend-sa@${PROJECT}.iam.gserviceaccount.com"
FRONTEND_SA="amie-frontend-sa@${PROJECT}.iam.gserviceaccount.com"
TARGET="${1:-both}"

deploy_backend() {
  echo "=== Deploying backend ==="
  cd "$(git rev-parse --show-toplevel)"

  gcloud run deploy patent-analyzer \
    --source . \
    --region "$REGION" \
    --service-account "$BACKEND_SA" \
    --no-allow-unauthenticated \
    --env-vars-file .env.yaml \
    --memory 4Gi \
    --cpu 2 \
    --timeout 900 \
    --concurrency 1 \
    --max-instances 3

  BACKEND_URL=$(gcloud run services describe patent-analyzer \
    --project="$PROJECT" --region="$REGION" --format='value(status.url)')
  echo "Backend URL: $BACKEND_URL"
  echo ""
  echo "Update frontend/.env.yaml with:"
  echo "  BACKEND_URL: \"$BACKEND_URL\""
}

deploy_frontend() {
  echo "=== Deploying frontend ==="
  cd "$(git rev-parse --show-toplevel)/frontend"

  gcloud run deploy patent-analyzer-frontend \
    --source . \
    --region "$REGION" \
    --service-account "$FRONTEND_SA" \
    --allow-unauthenticated \
    --env-vars-file .env.yaml

  FRONTEND_URL=$(gcloud run services describe patent-analyzer-frontend \
    --project="$PROJECT" --region="$REGION" --format='value(status.url)')
  echo "Frontend URL: $FRONTEND_URL"
}

case "$TARGET" in
  backend)  deploy_backend ;;
  frontend) deploy_frontend ;;
  both)     deploy_backend && deploy_frontend ;;
  *)        echo "Usage: $0 [backend|frontend|both]"; exit 1 ;;
esac
