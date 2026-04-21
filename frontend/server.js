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
  const res = await fetch(url, { method: options.method || "GET", headers, body });
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  const data = ct.includes("json") ? await res.json().catch(() => ({})) : await res.text().catch(() => "");
  return { status: res.status, data, contentType: ct };
}

// Parse JSON only for non-upload routes
app.use((req, res, next) => {
  if (req.path === "/api/analyze") return next();
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

// Get signed upload URL (for large files > 25MB)
app.get("/api/upload-url", async (req, res) => {
  try {
    const filename = String(req.query.filename || "");
    const content_type = String(req.query.content_type || "application/pdf");
    if (!filename) return res.status(400).json({ error: "Missing filename" });
    const qs = `?filename=${encodeURIComponent(filename)}&content_type=${encodeURIComponent(content_type)}`;
    const { status, data } = await proxyBackend(`/upload-url${qs}`);
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/upload-url:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Start analysis from a GCS URI (after client-side direct upload)
app.post("/api/analyze-gcs", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/analyze-gcs", { method: "POST", data: req.body });
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
    const { status, data } = await proxyBackend(`/status/${encodeURIComponent(req.params.jobId)}`);
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
    const { status, data } = await proxyBackend(`/events/${encodeURIComponent(req.params.jobId)}${since}`);
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/events:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Get report HTML
app.get("/api/report/:jobId", async (req, res) => {
  try {
    const { status, data, contentType } = await proxyBackend(`/report/${encodeURIComponent(req.params.jobId)}`);
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
    const { status, data } = await proxyBackend(`/results/${encodeURIComponent(req.params.jobId)}`);
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/results:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// List jobs
app.get("/api/jobs", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/jobs");
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in /api/jobs:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Delete a job
app.delete("/api/jobs/:jobId", async (req, res) => {
  try {
    const { status, data } = await proxyBackend(`/jobs/${encodeURIComponent(req.params.jobId)}`, { method: "DELETE" });
    res.status(status).json(data);
  } catch (e) {
    console.error("Error in DELETE /api/jobs:", e);
    res.status(500).json({ error: "Proxy error" });
  }
});

// Cleanup old jobs
app.post("/api/jobs/cleanup", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/jobs/cleanup", { method: "POST" });
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

// Health
app.get("/api/healthz", async (req, res) => {
  try {
    const { status, data } = await proxyBackend("/healthz");
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
