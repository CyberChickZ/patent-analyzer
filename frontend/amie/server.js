import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 8080;

const BACKEND_URL = process.env.BACKEND_URL;
const LOCAL_RUN = process.env.BACKEND_ENV === "dev";

if (!BACKEND_URL) {
  throw new Error("BACKEND_URL env var is required");
}

// ─── GCP IAM token (for Cloud Run service-to-service auth) ───

const METADATA_URL = `http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=${encodeURIComponent(BACKEND_URL)}`;
let cachedToken = null;
let cachedExpiryMs = 0;

function decodeJwtExpiryMs(token) {
  try {
    const payload = token.split(".")[1];
    if (!payload) return 0;
    const normalized = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
    const json = JSON.parse(Buffer.from(padded, "base64").toString("utf8"));
    return typeof json.exp === "number" ? json.exp * 1000 : 0;
  } catch {
    return 0;
  }
}

async function getIdToken() {
  if (LOCAL_RUN) return null;
  const now = Date.now();
  if (cachedToken && cachedExpiryMs - 60_000 > now) return cachedToken;
  const res = await fetch(METADATA_URL, { headers: { "Metadata-Flavor": "Google" } });
  if (!res.ok) throw new Error(`Metadata token fetch failed (${res.status})`);
  const token = await res.text();
  cachedToken = token;
  cachedExpiryMs = decodeJwtExpiryMs(token) || now + 55 * 60 * 1000;
  return token;
}

// ─── Backend proxy ───

function buildUrl(p) {
  return `${BACKEND_URL.replace(/\/$/, "")}${p.startsWith("/") ? p : `/${p}`}`;
}

async function proxyBackend(p, options = {}) {
  const url = buildUrl(p);
  const headers = { ...(options.headers || {}) };
  let body;
  if (options.body) {
    body = options.body;
    // Don't set content-type for FormData
  } else if (options.data !== undefined) {
    body = JSON.stringify(options.data);
    headers["Content-Type"] = "application/json";
  }
  if (!LOCAL_RUN) {
    const token = await getIdToken();
    headers.Authorization = `Bearer ${token}`;
  }
  if (options.firebaseToken) {
    headers["X-Firebase-Token"] = options.firebaseToken;
  }
  const res = await fetch(url, { method: options.method || "GET", headers, body });
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  const data = ct.includes("json") ? await res.json().catch(() => ({})) : await res.text().catch(() => "");
  return { status: res.status, data, contentType: ct };
}

function fbToken(req) {
  return req.headers["x-firebase-token"] || "";
}

// Parse JSON only for non-upload routes
app.use((req, res, next) => {
  // Multipart bodies are re-streamed to the backend untouched; parsing them as
  // JSON here would consume the stream and hand the backend an empty body.
  if (req.path === "/api/analyze" || /^\/api\/jobs\/[^/]+\/fulltext\/[^/]+\/upload$/.test(req.path)) return next();
  express.json()(req, res, next);
});

// ─── Public A2A discovery (no auth required from caller) ───
// The backend is --no-allow-unauthenticated, so other agents (n8n, etc.)
// cannot reach /.well-known/agent-card.json or /a2a directly. The public
// frontend proxies these requests with an IAM token attached.

app.get("/.well-known/agent-card.json", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/.well-known/agent-card.json");
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in agent-card proxy:", e);
    res.status(502).json({ error: "Backend unreachable" });
  }
});

app.get("/agent-card.json", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/agent-card.json");
    res.status(status).json(data);
  } catch (e) {
    res.status(502).json({ error: "Backend unreachable" });
  }
});

app.post("/a2a", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/a2a", { method: "POST", data: req.body });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in a2a proxy:", e);
    res.status(502).json({ error: "Backend unreachable" });
  }
});

// ─── API routes ───

// Frontend runtime config. BACKEND_ENV=dev means the backend runs with
// AUTH_DISABLED, so the UI skips the Firebase sign-in gate.
app.get("/api/config", (req, res) => {
  res.json({ dev: LOCAL_RUN, backend: LOCAL_RUN ? BACKEND_URL : undefined });
});


// Get signed upload URL (for large files > 25MB)
app.get("/api/upload-url", async (req, res) => {
  try {
    const filename = String(req.query.filename || "");
    const content_type = String(req.query.content_type || "application/pdf");
    if (!filename) return res.status(400).json({ error: "Missing filename" });
    const qs = `?filename=${encodeURIComponent(filename)}&content_type=${encodeURIComponent(content_type)}`;
    const { status, data } = await proxyBackend(`/upload-url${qs}`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/upload-url:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Start analysis from a GCS URI (after client-side direct upload)
app.post("/api/analyze-gcs", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/analyze-gcs", { method: "POST", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/analyze-gcs:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Upload file and start analysis
app.post("/api/analyze", async (req, res) => {
  try {
    // Re-stream the multipart body to backend
    const backendUrl = buildUrl("/analyze");
    const headers = {};
    if (!LOCAL_RUN) {
      const token = await getIdToken();
      headers.Authorization = `Bearer ${token}`;
    }
    // Forward the raw request
    const contentType = req.headers["content-type"];
    if (contentType) headers["Content-Type"] = contentType;
    const ft = fbToken(req);
    if (ft) headers["X-Firebase-Token"] = ft;

    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", async () => {
      try {
        const body = Buffer.concat(chunks);
        const backendRes = await fetch(backendUrl, {
          method: "POST",
          headers,
          body,
        });
        const data = await backendRes.json().catch(() => ({}));
        res.status(backendRes.status).json(data);
      } catch (e) {
        console.error("Error forwarding /api/analyze:", e);
        res.status(500).json({ error: "Proxy error" });
      }
    });
  } catch (e) {
    console.error("Error in /api/analyze:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Poll job status
app.get("/api/status/:jobId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/status/${encodeURIComponent(req.params.jobId)}`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/status:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Get events timeline
app.get("/api/events/:jobId", async (req, res) => {
  try {
    const since = req.query.since ? `?since=${req.query.since}` : "";
    const { status, data } = await proxyBackend(`/events/${encodeURIComponent(req.params.jobId)}${since}`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/events:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// HITL status
app.get("/api/hitl-status/:jobId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/hitl-status/${encodeURIComponent(req.params.jobId)}`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/hitl-status:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// HITL response (resume pipeline)
app.post("/api/hitl-response/:jobId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/hitl-response/${encodeURIComponent(req.params.jobId)}`, { method: "POST", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/hitl-response:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// HITL revise checklist via LLM
app.post("/api/hitl-revise/:jobId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/hitl-revise/${encodeURIComponent(req.params.jobId)}`, { method: "POST", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/hitl-revise:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Phase-level pause / edit / resume (graph gates)
app.get("/api/jobs/:jobId/state", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/jobs/${encodeURIComponent(req.params.jobId)}/state`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in GET /api/jobs/:id/state:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

app.patch("/api/jobs/:jobId/state", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/jobs/${encodeURIComponent(req.params.jobId)}/state`, { method: "PATCH", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in PATCH /api/jobs/:id/state:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

app.post("/api/jobs/:jobId/resume", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/jobs/${encodeURIComponent(req.params.jobId)}/resume`, { method: "POST", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in POST /api/jobs/:id/resume:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// ─── Missing full text: the list, the upload slot, the evidence re-run ───
//
// The reviewer's own PDF for a reference the run could not fetch. The upload is
// multipart and is re-streamed exactly like /api/analyze; the rest are plain
// JSON pass-throughs.

app.get("/api/jobs/:jobId/fulltext-gaps", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/jobs/${encodeURIComponent(req.params.jobId)}/fulltext-gaps`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in GET /api/jobs/:id/fulltext-gaps:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

app.post("/api/jobs/:jobId/fulltext/:refId/upload", async (req, res) => {
  try {
    const backendUrl = buildUrl(`/api/jobs/${encodeURIComponent(req.params.jobId)}/fulltext/${encodeURIComponent(req.params.refId)}/upload`);
    const headers = {};
    if (!LOCAL_RUN) headers.Authorization = `Bearer ${await getIdToken()}`;
    if (req.headers["content-type"]) headers["Content-Type"] = req.headers["content-type"];
    const ft = fbToken(req);
    if (ft) headers["X-Firebase-Token"] = ft;
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", async () => {
      try {
        const backendRes = await fetch(backendUrl, { method: "POST", headers, body: Buffer.concat(chunks) });
        res.status(backendRes.status).json(await backendRes.json().catch(() => ({})));
      } catch (e) {
        console.error("Error forwarding the full-text upload:", e);
        res.status(500).json({ error: "Proxy error" });
      }
    });
  } catch (e) {
    console.error("Error in POST /api/jobs/:id/fulltext/:ref/upload:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

app.post("/api/jobs/:jobId/fulltext/:refId/from-gcs", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/jobs/${encodeURIComponent(req.params.jobId)}/fulltext/${encodeURIComponent(req.params.refId)}/from-gcs`, { method: "POST", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in POST /api/jobs/:id/fulltext/:ref/from-gcs:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

app.delete("/api/jobs/:jobId/fulltext/:refId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/jobs/${encodeURIComponent(req.params.jobId)}/fulltext/${encodeURIComponent(req.params.refId)}`, { method: "DELETE", firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in DELETE /api/jobs/:id/fulltext/:ref:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

app.post("/api/jobs/:jobId/rerun-evidence", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/jobs/${encodeURIComponent(req.params.jobId)}/rerun-evidence`, { method: "POST", data: req.body || {}, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in POST /api/jobs/:id/rerun-evidence:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Prompt registry
app.get("/api/prompts", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/prompts`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) { res.status(500).json({ error: "Proxy error" }); }
});
app.get("/api/prompts/:name", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/prompts/${encodeURIComponent(req.params.name)}`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) { res.status(500).json({ error: "Proxy error" }); }
});
app.put("/api/prompts/:name", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/prompts/${encodeURIComponent(req.params.name)}`, { method: "PUT", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) { res.status(500).json({ error: "Proxy error" }); }
});
app.post("/api/prompts/:name/revise", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/prompts/${encodeURIComponent(req.params.name)}/revise`,
      { method: "POST", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    res.status(502).json({ error: "Proxy error" });
  }
});

app.put("/api/prompts/:name/current", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/prompts/${encodeURIComponent(req.params.name)}/current`, { method: "PUT", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) { res.status(500).json({ error: "Proxy error" }); }
});

// Get report HTML
app.get("/api/report/:jobId", async (req, res) => {
  try {
    const { status, data, contentType } = await proxyBackend(`/report/${encodeURIComponent(req.params.jobId)}`, { firebaseToken: fbToken(req) });
    if (contentType.includes("html")) {
      res.status(status).type("html").send(data);
    } else {
      res.status(status).json(data);
    }
  } catch (e) {
    console.error("Error in /api/report:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Get results JSON
app.get("/api/results/:jobId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/results/${encodeURIComponent(req.params.jobId)}`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/results:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// List jobs
app.get("/api/jobs", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/jobs", { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/jobs:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Delete a job
app.delete("/api/jobs/:jobId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/jobs/${encodeURIComponent(req.params.jobId)}`, { method: "DELETE", firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in DELETE /api/jobs:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Cleanup old jobs
app.post("/api/jobs/cleanup", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/jobs/cleanup", { method: "POST", firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/jobs/cleanup:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Feedback capture — lets the HTML report POST user feedback per job
app.post("/api/feedback/:jobId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(
      `/feedback/${encodeURIComponent(req.params.jobId)}`,
      { method: "POST", data: req.body },
    );
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/feedback POST:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

app.get("/api/feedback/:jobId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(
      `/feedback/${encodeURIComponent(req.params.jobId)}`,
    );
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/feedback GET:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

app.post("/api/admin/set-developer", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/admin/set-developer", { method: "POST", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/admin/set-developer:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// What one job actually spent. The backend has served this since f85ef88, but
// nothing proxied it, so every page that wanted a per-job figure had to print
// an em dash next to an estimate. It is the measured number — the per-phase
// meter diffed while the job ran — and it is the only one worth showing beside
// a rate card, which is a list price and not a bill.
app.get("/api/jobs/:jobId/usage", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/api/jobs/${encodeURIComponent(req.params.jobId)}/usage`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in GET /api/jobs/:id/usage:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Feedback timeline — prompt edits, reviewer edits, comments and ratings.
// Distinct from /api/feedback/:jobId below, which is the old per-job rating file.
app.get("/api/feedback", async (req, res) => {
  try {
    const qs = new URLSearchParams(req.query).toString();
    const { status, data } = await proxyBackend(`/feedback${qs ? "?" + qs : ""}`, { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    res.status(502).json({ error: "Backend unreachable" });
  }
});

app.post("/api/feedback", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/feedback", { method: "POST", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    res.status(502).json({ error: "Backend unreachable" });
  }
});

app.patch("/api/feedback/:id", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/feedback/${encodeURIComponent(req.params.id)}`,
      { method: "PATCH", data: req.body, firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    res.status(502).json({ error: "Backend unreachable" });
  }
});

// Health
// Rate card / quota — what every external source has left, plus the prices the
// backend costs a run with. Live as of N1; the backend serves it at /api/quota.
app.get("/api/quota", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/api/quota", { firebaseToken: fbToken(req) });
    res.status(status).json(data);
  } catch (e) {
    res.status(502).json({ error: "Backend unreachable" });
  }
});

// The backend path moved from /healthz to /health: on Cloud Run, /healthz is
// intercepted in front of the container and answered with Google's own 404 page,
// so this proxy has been forwarding to a path that could not reply — from here
// too, since the hop goes out to the same *.run.app host.
app.get("/api/health", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/health");
    res.status(status).json(data);
  } catch (e) {
    res.status(502).json({ error: "Backend unreachable" });
  }
});

// ─── Static files ───

app.use(express.static(path.join(__dirname, "dist")));

app.get(/.*/, (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

app.listen(port, () => {
  console.log(`Frontend ${LOCAL_RUN ? "(dev)" : "(prod)"} → ${BACKEND_URL} on :${port}`);
});
