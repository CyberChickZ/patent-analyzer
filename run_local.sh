#!/usr/bin/env bash
# One command to run the whole thing locally: backend on :8000, the UI proxy on
# :5173. Ctrl-C stops both. Everything it needs is already in backend/.env.yaml
# (git-ignored): the Vertex project, the SerpAPI keys, Lens, USPTO ODP.
#
#   ./run_local.sh                 # start both, open http://localhost:5173
#   ./run_local.sh --backend-only  # just the API
#   BACKEND_PORT=8010 ./run_local.sh
#
# Auth is off locally (AUTH_DISABLED=1) — the UI skips the Firebase sign-in and
# the API takes requests without a token. Never set that in Cloud Run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"
BACKEND_ONLY=0
[ "${1:-}" = "--backend-only" ] && BACKEND_ONLY=1

command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
[ -f "$ROOT/backend/.env.yaml" ] || { echo "backend/.env.yaml missing — it holds the API keys and is git-ignored"; exit 1; }
for p in "$BACKEND_PORT" "$UI_PORT"; do
  if lsof -ti:"$p" >/dev/null 2>&1; then echo "port $p is busy — set BACKEND_PORT / UI_PORT"; exit 1; fi
done

pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "backend  → http://localhost:$BACKEND_PORT   (logs: $ROOT/backend/local.log)"
(
  cd "$ROOT/backend"
  PYTHONPATH="$ROOT/backend" AUTH_DISABLED=1 BACKEND_ENV=dev CHECKPOINT_BACKEND=file \
    python3 -m uvicorn app.main:app --port "$BACKEND_PORT" > local.log 2>&1
) &
pids+=($!)

for _ in $(seq 1 60); do
  curl -sf "http://localhost:$BACKEND_PORT/healthz" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "http://localhost:$BACKEND_PORT/healthz" >/dev/null 2>&1 \
  || { echo "backend did not come up — see backend/local.log"; exit 1; }

if [ "$BACKEND_ONLY" = "1" ]; then
  echo "backend is up. Ctrl-C to stop."
  wait
fi

command -v node >/dev/null || { echo "node not found (the UI proxy needs Node 20)"; exit 1; }
[ -d "$ROOT/frontend/amie/node_modules" ] || (cd "$ROOT/frontend/amie" && npm install --silent)
echo "UI       → http://localhost:$UI_PORT"
(
  cd "$ROOT/frontend/amie"
  npm run build --silent
  BACKEND_URL="http://localhost:$BACKEND_PORT" BACKEND_ENV=dev PORT="$UI_PORT" node server.js
) &
pids+=($!)

echo
echo "Open http://localhost:$UI_PORT — Ctrl-C stops both."
wait
