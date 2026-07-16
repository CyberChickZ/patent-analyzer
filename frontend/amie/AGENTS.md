# Repository Guidelines

## Project Structure & Module Organization
- App shell lives in `index.html` with styles in `src/style.css` and client logic in `src/main.ts`.
- API proxy and preview server helpers sit in `src/server.js` (Express-based).
- Static assets go in `public/`; build output is emitted to `dist/` by Vite.
- Tooling configs: `vite.config.ts` (dev server + proxy), `tsconfig.json`, and `package.json`.

## Build, Test, and Development Commands
- Install deps: `npm install` (Node 18+ recommended).
- Local dev with proxy: `npm run dev` (Vite on `5173`, proxies `/upload-file|/invoke|/state|/debug_state|/debug` to `VITE_API_BASE` or `http://127.0.0.1:8000`).
- Production build: `npm run build` → outputs `dist/`.
- Preview built assets: `npm run preview -- --host 0.0.0.0 --port 8080`.
- Standalone proxy server: `PORT=8787 API_BASE=http://127.0.0.1:8000 npm run serve` (for CORS-friendly JSON endpoints; `/upload-file` is intentionally unimplemented here).

## Coding Style & Naming Conventions
- TypeScript + ES modules; prefer `const`/`let` over `var` and arrow functions for callbacks.
- Two-space indentation, single quotes in TS, and keep helper utilities small and scoped near their usage.
- Keep DOM IDs and CSS classnames descriptive (`card-naa`, `details-ia`, `badge-*`).
- Vite env vars: `VITE_USE_PROXY` (`true` default) and `VITE_API_BASE` for direct API host; avoid hardcoding service URLs.

## Testing Guidelines
- No automated test suite yet; verify flows manually via `npm run dev`:
  - Upload a PDF/image → confirm `doc_gcs_uri` is returned.
  - Invoke with a `gs://` URI → confirm `request_id` populates.
  - Poll `/debug_state` → check agent cards update and novelty assessment summary renders.
- Add lightweight smoke tests (e.g., using Playwright/Vitest) when expanding UI or state parsing logic.

## Commit & Pull Request Guidelines
- Use concise, action-oriented commit subjects similar to existing history (e.g., `Add NAA similarity summary`, `Deploy frontend`); keep under ~72 chars.
- PRs should describe intent, key changes, and how to reproduce/verify the flows above. Include screenshots/GIFs for UI updates and note any env var expectations.
- Link issues or tickets when available; call out risk areas (API contract changes, storage paths, polling intervals) and any manual steps required after merge.
