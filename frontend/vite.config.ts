import { defineConfig } from "vite";

const BACKEND = process.env.VITE_API_BASE || "http://127.0.0.1:8000";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/analyze": BACKEND,
      "/status": BACKEND,
      "/report": BACKEND,
      "/results": BACKEND,
      "/jobs": BACKEND,
      "/healthz": BACKEND,
      "/a2a": BACKEND,
      "/.well-known": BACKEND,
    },
  },
});
