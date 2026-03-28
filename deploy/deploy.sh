#!/bin/bash
# Deploy to GCP Cloud Run
# Usage: ./deploy/deploy.sh PROJECT_ID

set -euo pipefail

PROJECT_ID="${1:?Usage: deploy.sh PROJECT_ID}"
IMAGE="gcr.io/${PROJECT_ID}/patent-analyzer:latest"

echo "Building image..."
docker build -t "$IMAGE" .

echo "Pushing to GCR..."
docker push "$IMAGE"

echo "Creating secrets (if not exists)..."
echo -n "$ANTHROPIC_API_KEY" | gcloud secrets create anthropic-api-key --data-file=- --project="$PROJECT_ID" 2>/dev/null || true
echo -n "$SERPAPI_KEY" | gcloud secrets create serpapi-key --data-file=- --project="$PROJECT_ID" 2>/dev/null || true

echo "Deploying to Cloud Run..."
gcloud run deploy patent-analyzer \
  --image="$IMAGE" \
  --project="$PROJECT_ID" \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --memory=4Gi \
  --cpu=2 \
  --timeout=900 \
  --concurrency=1 \
  --max-instances=3 \
  --set-secrets="ANTHROPIC_API_KEY=anthropic-api-key:latest,SERPAPI_KEY=serpapi-key:latest" \
  --set-env-vars="ANTHROPIC_MODEL=claude-sonnet-4-20250514,OUTPUT_DIR=/tmp/outputs"

echo "Done! URL:"
gcloud run services describe patent-analyzer --project="$PROJECT_ID" --region=us-central1 --format='value(status.url)'
