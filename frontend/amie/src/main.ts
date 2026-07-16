import "./style.css";
import { login, logout, getToken, onAuth, isDeveloper } from './auth';

const API = "";

interface Job {
  id: string;
  status: string;
  phase: string;
  filename: string;
  phases: Record<string, any>;
  error?: string;
}

const PHASES = [
  { key: "phase1", label: "Invention Detection" },
  { key: "phase2", label: "Decomposition" },
  { key: "phase3", label: "Prior Art Search" },
  { key: "phase3b", label: "Semantic Ranking" },
  { key: "phase4", label: "Deep Evaluation" },
  { key: "phase5", label: "Report Generation" },
];

// ─── Render app shell ───

const root = document.getElementById("app")!;
root.innerHTML = `
<div class="container">
  <div class="auth-bar" id="auth-bar">
    <div id="auth-status">Loading...</div>
  </div>
  <h1>Patent Analyzer</h1>
  <div class="sub">Upload a paper or patent PDF — get a full novelty analysis</div>

  <div class="card hidden" id="upload-card">
    <h2>Upload</h2>
    <div class="drop" id="drop">
      <div class="icon">&#x1F4C4;</div>
      <p>Drop PDF here or click to upload</p>
      <input type="file" id="fileIn" accept=".pdf,.txt,.md">
    </div>
    <div class="fname" id="fname"></div>
    <div class="email-row" id="email-row">
      <label class="email-toggle">
        <input type="checkbox" id="emailCheck">
        <span>Email report on completion</span>
      </label>
      <input type="email" id="emailInput" class="email-input" placeholder="recipient@example.com" style="display:none">
    </div>
    <div class="hitl-row">
      <label class="hitl-toggle">
        <input type="checkbox" id="hitlCheck">
        <span>Expert Review Mode</span>
      </label>
      <span class="hitl-hint">Pause after checklist &amp; search for your review</span>
    </div>
    <button class="btn" id="startBtn" disabled>Analyze</button>
  </div>

  <div class="card hidden" id="status-card">
    <h2>Analysis Progress</h2>
    <div class="phases" id="phases"></div>
    <div id="stats-area"></div>
    <div id="error-area"></div>
    <a class="report-link hidden" id="reportLink" target="_blank">View Full Report &rarr;</a>
  </div>

  <div class="card" id="jobs-card">
    <h2>Recent Jobs</h2>
    <div class="job-list" id="job-list"><span style="color:var(--muted);font-size:0.83rem">Loading...</span></div>
  </div>
</div>

<div class="modal-overlay" id="evt-modal">
  <div class="modal-box">
    <button class="modal-close" id="evt-modal-close">×</button>
    <div id="evt-modal-content"></div>
  </div>
</div>
`;
// Modal close handlers
document.getElementById("evt-modal")!.addEventListener("click", (e) => {
  if ((e.target as HTMLElement).id === "evt-modal") {
    document.getElementById("evt-modal")!.classList.remove("open");
  }
});
document.getElementById("evt-modal-close")!.addEventListener("click", () => {
  document.getElementById("evt-modal")!.classList.remove("open");
});

// ─── Auth ───

async function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const token = await getToken();
  const headers = new Headers(options.headers);
  if (token) headers.set('X-Firebase-Token', token);
  return fetch(url, { ...options, headers });
}

const uploadCard = document.getElementById("upload-card")!;
const authBar = document.getElementById("auth-bar")!;

onAuth(async (user) => {
  authBar.innerHTML = '';
  if (user) {
    const dev = await isDeveloper();
    const info = document.createElement('div');
    info.innerHTML = `<span class="user-info"><span class="user-email">${escapeHtml(user.email || '')}</span>${dev ? '<span class="dev-badge">dev</span>' : ''}</span>`;
    authBar.appendChild(info);
    const logoutBtn = document.createElement('button');
    logoutBtn.className = 'btn-logout';
    logoutBtn.textContent = 'Sign out';
    logoutBtn.addEventListener('click', () => logout());
    authBar.appendChild(logoutBtn);
    uploadCard.classList.remove('hidden');
    refreshJobs();
  } else {
    const loginBtn = document.createElement('button');
    loginBtn.className = 'btn-login';
    loginBtn.textContent = 'Sign in with Google';
    loginBtn.addEventListener('click', async () => {
      try {
        await login();
      } catch (e: any) {
        alert(e.message || 'Login failed');
      }
    });
    authBar.appendChild(loginBtn);
    uploadCard.classList.add('hidden');
  }
});

// ─── Elements ───

const drop = document.getElementById("drop")!;
const fileIn = document.getElementById("fileIn") as HTMLInputElement;
const fnameEl = document.getElementById("fname")!;
const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
const statusCard = document.getElementById("status-card")!;
const phasesEl = document.getElementById("phases")!;
const statsArea = document.getElementById("stats-area")!;
const errorArea = document.getElementById("error-area")!;
const reportLink = document.getElementById("reportLink") as HTMLAnchorElement;
const jobList = document.getElementById("job-list")!;

const emailCheck = document.getElementById("emailCheck") as HTMLInputElement;
const emailInput = document.getElementById("emailInput") as HTMLInputElement;

emailCheck.addEventListener("change", () => {
  emailInput.style.display = emailCheck.checked ? "" : "none";
  if (emailCheck.checked) emailInput.focus();
});

// Restore last-used email from localStorage
const savedEmail = localStorage.getItem("pa_notify_email");
if (savedEmail) {
  emailInput.value = savedEmail;
  emailCheck.checked = true;
  emailInput.style.display = "";
}

let selectedFile: File | null = null;
let pollTimer: number | null = null;

// ─── Upload logic ───

drop.addEventListener("click", () => fileIn.click());
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if ((e as DragEvent).dataTransfer?.files.length) {
    selectedFile = (e as DragEvent).dataTransfer!.files[0];
    showFile();
  }
});
fileIn.addEventListener("change", () => {
  if (fileIn.files?.length) {
    selectedFile = fileIn.files[0];
    showFile();
  }
});

function showFile() {
  if (!selectedFile) return;
  fnameEl.textContent = selectedFile.name;
  startBtn.disabled = false;
}

startBtn.addEventListener("click", startAnalysis);

async function startAnalysis() {
  if (!selectedFile) return;

  const notifyEmail = emailCheck.checked ? emailInput.value.trim() : "";
  if (emailCheck.checked && !notifyEmail) {
    emailInput.focus();
    emailInput.classList.add("shake");
    setTimeout(() => emailInput.classList.remove("shake"), 400);
    return;
  }
  if (notifyEmail) localStorage.setItem("pa_notify_email", notifyEmail);

  startBtn.disabled = true;
  startBtn.textContent = "Preparing...";
  errorArea.innerHTML = "";

  // Show progress card immediately with a "starting" state
  statusCard.classList.remove("hidden");
  phasesEl.innerHTML = PHASES.map(p =>
    `<div class="phase-row"><span>• ${p.label}</span><span class="badge pending">pending</span></div>`
  ).join("");

  const LARGE_FILE_THRESHOLD = 25 * 1024 * 1024; // 25 MB — Cloud Run hard limit is 32 MB
  const useSignedUrl = selectedFile.size > LARGE_FILE_THRESHOLD;

  try {
    let jobId: string;

    if (useSignedUrl) {
      // Step 1: Get signed PUT URL
      startBtn.textContent = "Requesting upload URL...";
      const urlResp = await authFetch(`${API}/api/upload-url?filename=${encodeURIComponent(selectedFile.name)}&content_type=application/pdf`);
      if (!urlResp.ok) {
        const errText = await urlResp.text();
        throw new Error(`Failed to get upload URL: ${errText.slice(0, 200)}`);
      }
      const urlData = await urlResp.json();

      // Step 2: PUT file directly to GCS
      startBtn.textContent = `Uploading ${(selectedFile.size / 1024 / 1024).toFixed(1)} MB to GCS...`;
      const putResp = await fetch(urlData.signed_url, {
        method: "PUT",
        headers: { "Content-Type": "application/pdf" },
        body: selectedFile,
      });
      if (!putResp.ok) {
        const errText = await putResp.text();
        throw new Error(`GCS upload failed (${putResp.status}): ${errText.slice(0, 200)}`);
      }

      // Step 3: Start analysis with the GCS URI
      startBtn.textContent = "Starting analysis...";
      const analyzeResp = await authFetch(`${API}/api/analyze-gcs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: urlData.job_id,
          gcs_uri: urlData.gcs_uri,
          filename: selectedFile.name,
          notify_email: notifyEmail,
          hitl_enabled: (document.getElementById("hitlCheck") as HTMLInputElement).checked,
        }),
      });
      if (!analyzeResp.ok) {
        const errText = await analyzeResp.text();
        throw new Error(`Analyze start failed: ${errText.slice(0, 200)}`);
      }
      const analyzeData = await analyzeResp.json();
      jobId = analyzeData.job_id;
    } else {
      // Small file: direct multipart upload
      startBtn.textContent = "Uploading...";
      const fd = new FormData();
      fd.append("file", selectedFile);
      const hitlEnabled = (document.getElementById("hitlCheck") as HTMLInputElement).checked;
      if (notifyEmail) fd.append("notify_email", notifyEmail);
      if (hitlEnabled) fd.append("hitl_enabled", "true");
      const resp = await authFetch(`${API}/api/analyze`, { method: "POST", body: fd });
      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`Upload failed (${resp.status}): ${errText.slice(0, 200)}`);
      }
      const data = await resp.json();
      if (!data.job_id) throw new Error(data.error || "No job_id returned");
      jobId = data.job_id;
    }

    // Hide upload card, show only progress
    uploadCard.classList.add("hidden");
    statusCard.classList.remove("hidden");
    startPolling(jobId);
    refreshJobs();
  } catch (e: any) {
    errorArea.innerHTML = `<div class="error-box">${escapeHtml(e.message || String(e))}</div>`;
    statusCard.classList.remove("hidden");
    // Re-enable upload on error
    startBtn.disabled = false;
    startBtn.textContent = "Analyze";
  }
  startBtn.textContent = "Analyze";
  startBtn.disabled = false;
}

// ─── Polling ───

function startPolling(jobId: string) {
  if (pollTimer) clearInterval(pollTimer);
  reportLink.classList.add("hidden");
  errorArea.innerHTML = "";
  statsArea.innerHTML = "";
  allEvents = [];
  expandedPhase = null;

  const poll = async () => {
    try {
      const [statusResp, eventsResp] = await Promise.all([
        authFetch(`${API}/api/status/${jobId}`),
        authFetch(`${API}/api/events/${jobId}?since=${allEvents.length}`),
      ]);
      const job: Job = await statusResp.json();
      const eventsData = await eventsResp.json();
      if (eventsData.events) {
        allEvents = allEvents.concat(eventsData.events);
      }
      renderPhases(job);
      renderHitl(jobId, job);

      if (job.status === "completed") {
        if (pollTimer) clearInterval(pollTimer);
        reportLink.href = `${API}/api/report/${jobId}`;
        reportLink.classList.remove("hidden");
        renderStats(job);
        // Re-show upload card for next analysis
        uploadCard.classList.remove("hidden");
        startBtn.disabled = false;
        startBtn.textContent = "Analyze";
      } else if (job.status === "error") {
        if (pollTimer) clearInterval(pollTimer);
        errorArea.innerHTML = `<div class="error-box">${job.error || "Pipeline error"}</div>`;
        uploadCard.classList.remove("hidden");
        startBtn.disabled = false;
        startBtn.textContent = "Analyze";
      }
    } catch {
      // ignore transient fetch errors
    }
  };

  poll();
  pollTimer = window.setInterval(poll, 2000);
}

let allEvents: any[] = [];
let expandedPhase: string | null = null;
let lastRenderHash = "";
let latestJob: Job | null = null;

function hasActiveSelectionInside(container: HTMLElement): boolean {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return false;
  const range = sel.getRangeAt(0);
  return container.contains(range.commonAncestorContainer);
}

function renderPhases(job: Job) {
  latestJob = job;
  // Skip re-render if user is selecting text inside the phases element
  if (hasActiveSelectionInside(phasesEl)) return;
  // Skip re-render if nothing actually changed
  const phasesHash = JSON.stringify(Object.keys(job.phases || {}));
  const hash = `${job.status}|${job.phase}|${allEvents.length}|${expandedPhase}|${phasesHash}`;
  if (hash === lastRenderHash) return;
  lastRenderHash = hash;

  // Derive completed phases from events (more reliable than job.phases with LangGraph)
  const phasesWithEvents = new Set(allEvents.map(e => e.phase));
  const completedPhases = new Set<string>();
  for (const pk of phasesWithEvents) {
    // A phase is "done" if a later phase has events
    const idx = PHASES.findIndex(p => p.key === pk);
    if (idx >= 0) {
      for (let i = 0; i < idx; i++) completedPhases.add(PHASES[i].key);
    }
  }
  // Also trust job.phases from backend
  for (const pk of Object.keys(job.phases || {})) completedPhases.add(pk);

  const isHitlWaiting = job.status === "waiting_for_hitl";
  let html = "";
  for (const p of PHASES) {
    let st = "pending";
    if (job.status === "completed") {
      st = "completed";
    } else if (completedPhases.has(p.key)) {
      st = "completed";
    } else if (job.phases[p.key]) {
      st = "completed";
    } else if (phasesWithEvents.has(p.key)) {
      st = isHitlWaiting ? "completed" : "running";
    } else if (job.phase === p.key) {
      st = job.status === "error" ? "error" : (isHitlWaiting ? "completed" : "running");
    }
    const phaseEvents = allEvents.filter((e) => e.phase === p.key);
    const hasEvents = phaseEvents.length > 0;
    const isOpen = expandedPhase === p.key;
    html += `<div class="phase-block ${isOpen ? 'open' : ''}">
      <div class="phase-row" data-phase="${p.key}" ${hasEvents ? 'style="cursor:pointer"' : ''}>
        <span class="label">${hasEvents ? '▸ ' : ''}${p.label}</span>
        <span class="badge badge-${st}">${st}</span>
      </div>
      ${isOpen ? renderEventTimeline(phaseEvents) : ''}
    </div>`;
    // Insert HITL waiting indicator after phase2
    if (p.key === "phase2" && isHitlWaiting) {
      html += `<div class="phase-block">
        <div class="phase-row hitl-waiting">
          <span class="label">⏸ Waiting for Expert Review</span>
          <span class="badge badge-waiting">review</span>
        </div>
      </div>`;
    }
  }
  phasesEl.innerHTML = html;
  renderedEvents = allEvents.slice();
  // Attach click handlers
  phasesEl.querySelectorAll('.phase-row[data-phase]').forEach((el) => {
    el.addEventListener('click', () => {
      const key = (el as HTMLElement).dataset.phase!;
      expandedPhase = expandedPhase === key ? null : key;
      lastRenderHash = ""; // force re-render
      renderPhases(latestJob || job);
    });
  });
  // Event row click → open modal
  phasesEl.querySelectorAll('.event-row.evt-clickable').forEach((el) => {
    el.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const idx = parseInt((el as HTMLElement).dataset.evtIdx || '-1');
      // Need to find the corresponding event in allEvents filtered by phase
      const phase = expandedPhase;
      if (!phase) return;
      const phaseEvts = renderedEvents.filter((e) => e.phase === phase);
      const e = phaseEvts[idx];
      if (e) openEventModal(e);
    });
  });
}

function renderEventTimeline(events: any[]): string {
  if (!events.length) return '<div class="event-empty">No events yet</div>';
  return `<div class="event-timeline">${events.map((e, i) => renderEvent(e, i)).join('')}</div>`;
}

// Store events globally so modal can reference by index
let renderedEvents: any[] = [];

function renderEvent(e: any, idx: number): string {
  const time = new Date(e.ts).toLocaleTimeString();
  const kindIcon: Record<string, string> = {
    start: '▶',
    info: '·',
    llm_call: '?',
    llm_response: '!',
    search: 'q',
    search_result: '✓',
    download: '⬇',
    evaluating: '⚙',
    group_start: '┌',
    group_done: '└',
    done: '✓',
    error: '✗',
    warn: '⚠',
  };
  const icon = kindIcon[e.kind] || '·';
  const cls = e.kind === 'error' ? 'evt-error' : e.kind === 'warn' ? 'evt-warn' : e.kind.startsWith('llm') ? 'evt-llm' : '';
  const hasDetail = e.payload && (e.payload.system || e.payload.user || e.payload.response || e.payload.full || e.payload.traceback || e.payload.query);
  const clickable = hasDetail ? 'evt-clickable' : '';
  let payloadHtml = '';
  if (e.payload) {
    if (e.payload.preview) {
      payloadHtml = `<div class="evt-payload">${escapeHtml(e.payload.preview)}</div>`;
    } else if (e.payload.query) {
      payloadHtml = `<div class="evt-payload"><code>${escapeHtml(e.payload.query)}</code></div>`;
    }
  }
  return `<div class="event-row ${cls} ${clickable}" data-evt-idx="${idx}">
    <span class="evt-icon">${icon}</span>
    <span class="evt-time">${time}</span>
    <span class="evt-msg">${escapeHtml(e.message)}${hasDetail ? ' <span class="evt-detail-hint">(click)</span>' : ''}</span>
    ${payloadHtml}
  </div>`;
}

function openEventModal(e: any) {
  const overlay = document.getElementById("evt-modal") as HTMLElement;
  const content = document.getElementById("evt-modal-content") as HTMLElement;
  const p = e.payload || {};
  let html = `<h3>${escapeHtml(e.kind)}: ${escapeHtml(e.message)}</h3>`;
  html += `<div class="modal-meta">${new Date(e.ts).toLocaleString()} · phase: ${e.phase}</div>`;
  if (p.system) {
    html += `<div class="modal-section"><h4>System Prompt <span class="tag tag-template">hardcoded template</span></h4><pre>${escapeHtml(p.system)}</pre></div>`;
  }
  if (p.user) {
    html += `<div class="modal-section"><h4>User Prompt</h4>${renderUserPrompt(p.user)}</div>`;
  }
  if (p.response) {
    html += `<div class="modal-section"><h4>Response <span class="tag tag-llm">LLM output</span></h4><div class="modal-markdown">${renderMarkdown(p.response)}</div></div>`;
  }
  if (p.query) {
    html += `<div class="modal-section"><h4>Query</h4><pre>${escapeHtml(p.query)}</pre></div>`;
  }
  if (p.full) {
    html += `<div class="modal-section"><h4>Full Data</h4><pre>${escapeHtml(typeof p.full === 'string' ? p.full : JSON.stringify(p.full, null, 2))}</pre></div>`;
  }
  if (p.traceback) {
    html += `<div class="modal-section"><h4>Traceback</h4><pre>${escapeHtml(p.traceback)}</pre></div>`;
  }
  content.innerHTML = html;
  overlay.classList.add("open");
}

function renderUserPrompt(text: string): string {
  // Split prompt by ════ TASK ════ / ════ INPUT ════ markers
  const sections = text.split(/════ ([^═]+) ════/g);
  // sections will alternate: [intro, label1, content1, label2, content2, ...]
  if (sections.length < 3) {
    return `<pre>${escapeHtml(text)}</pre>`;
  }
  let html = '';
  if (sections[0].trim()) {
    html += `<pre>${escapeHtml(sections[0].trim())}</pre>`;
  }
  for (let i = 1; i < sections.length; i += 2) {
    const label = sections[i].trim();
    const content = (sections[i + 1] || '').trim();
    const isTemplate = /template/i.test(label);
    const tag = isTemplate
      ? '<span class="tag tag-template">hardcoded</span>'
      : '<span class="tag tag-data">data</span>';
    html += `<div class="prompt-block ${isTemplate ? 'pb-template' : 'pb-data'}">
      <div class="pb-label">${escapeHtml(label)} ${tag}</div>
      <pre>${escapeHtml(content)}</pre>
    </div>`;
  }
  return html;
}

function renderMarkdown(text: string): string {
  // Lightweight markdown: bold, italic, code, headers, lists, paragraphs
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*?<\/li>\n?)+/gs, (m) => `<ul>${m}</ul>`);
  html = html.split(/\n\n+/).map((p) => (p.startsWith('<') ? p : `<p>${p.replace(/\n/g, '<br>')}</p>`)).join('');
  return html;
}

function escapeHtml(s: string): string {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]!));
}

function renderStats(job: Job) {
  const p3 = job.phases.phase3 || job.phases.phase3b || {};
  const p4 = job.phases.phase4 || {};
  // Try phases data first, then extract from events as fallback
  let patents = p3.patents;
  let papers = p3.papers;
  let evaluated = p4.evaluated;
  let topScore = p4.top_score;

  // Fallback: extract from event messages if phases data is missing
  if (patents == null || papers == null) {
    for (const e of allEvents) {
      const m = e.message || "";
      const poolMatch = m.match(/Pool:\s*(\d+)\s*unique/);
      if (poolMatch) { papers = (papers ?? 0) + 0; }
      if (m.includes("patents") && m.includes("papers")) {
        const pm = m.match(/(\d+)\s*patents/);
        const pp = m.match(/(\d+)\s*papers/);
        if (pm) patents = parseInt(pm[1]);
        if (pp) papers = parseInt(pp[1]);
      }
    }
  }
  if (evaluated == null) {
    for (const e of allEvents) {
      const m = e.message || "";
      const em = m.match(/Evaluated\s*(\d+)/i);
      if (em) evaluated = parseInt(em[1]);
      const sm = m.match(/top.?score.*?(\d+\.?\d*)%/i);
      if (sm) topScore = parseFloat(sm[1]) / 100;
    }
  }

  const patentsStr = patents ?? "-";
  const papersStr = papers ?? "-";
  const evalStr = evaluated ?? "-";
  const topStr = topScore != null ? `${(topScore * 100).toFixed(1)}%` : "-";

  statsArea.innerHTML = `
  <div class="stats">
    <div class="stat"><div class="val">${patentsStr}</div><div class="lbl">Patents Found</div></div>
    <div class="stat"><div class="val">${papersStr}</div><div class="lbl">Papers Found</div></div>
    <div class="stat"><div class="val">${evalStr}</div><div class="lbl">Evaluated</div></div>
    <div class="stat"><div class="val">${topStr}</div><div class="lbl">Top Similarity</div></div>
  </div>`;
}

// ─── Job list ───

async function refreshJobs() {
  try {
    const resp = await authFetch(`${API}/api/jobs`);
    const data = await resp.json();
    if (!Array.isArray(data) || data.length === 0) {
      jobList.innerHTML = `<span style="color:var(--muted);font-size:0.83rem">No jobs yet</span>`;
      return;
    }
    // Sort by created_at desc (newest first)
    data.sort((a: any, b: any) => (b.created_at || "").localeCompare(a.created_at || ""));
    jobList.innerHTML = data
      .map(
        (j: any) => `<div class="job-item" data-id="${j.id}">
          <span class="job-name">${j.id} — ${j.filename || "?"}</span>
          <div class="job-actions">
            <span class="badge badge-${j.status}">${j.status}</span>
            <button class="del-btn" data-del="${j.id}" title="Delete job">✕</button>
          </div>
        </div>`
      )
      .join("");
    jobList.querySelectorAll(".job-item").forEach((el) => {
      el.addEventListener("click", (e) => {
        // Don't navigate if delete button was clicked
        if ((e.target as HTMLElement).classList.contains("del-btn")) return;
        const id = (el as HTMLElement).dataset.id!;
        uploadCard.classList.add("hidden");
        statusCard.classList.remove("hidden");
        startPolling(id);
      });
    });
    jobList.querySelectorAll(".del-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const id = (btn as HTMLElement).dataset.del!;
        if (!confirm(`Delete job ${id}? This removes all files from GCS.`)) return;
        try {
          await authFetch(`${API}/api/jobs/${id}`, { method: "DELETE" });
          // If the deleted job is currently being viewed, close progress card
          if (pollTimer) clearInterval(pollTimer);
          statusCard.classList.add("hidden");
          uploadCard.classList.remove("hidden");
          errorArea.innerHTML = "";
          const hitlEl = document.getElementById("hitl-inline");
          if (hitlEl) hitlEl.remove();
          await refreshJobs();
        } catch {
          alert("Failed to delete job");
        }
      });
    });
  } catch {
    jobList.innerHTML = `<span style="color:var(--muted);font-size:0.83rem">Could not load jobs</span>`;
  }
}

// ─── HITL UI (inline in progress stepper) ───

let hitlRendered = false;

function renderHitl(jobId: string, job: Job) {
  const hitlEl = document.getElementById("hitl-inline");

  if (job.status !== "waiting_for_hitl") {
    if (hitlEl) hitlEl.remove();
    hitlRendered = false;
    return;
  }

  if (hitlRendered) return;
  hitlRendered = true;

  const hitlPending = (job as any).hitl_pending;
  if (!hitlPending) return;

  const data = hitlPending.data || {};
  const checklist: any[] = data.checklist || [];

  const phasesEl = document.getElementById("phases");
  if (!phasesEl) return;

  _mountHitlForm(phasesEl, jobId, hitlPending.prompt || "", checklist);
}

function _mountHitlForm(phasesEl: HTMLElement, jobId: string, prompt: string, checklist: any[]) {
  const old = document.getElementById("hitl-inline");
  if (old) old.remove();

  const inlineDiv = document.createElement("div");
  inlineDiv.id = "hitl-inline";
  inlineDiv.className = "hitl-inline-form";

  const clHtml = checklist.map((c: any, i: number) => {
    const id = c.id || `c${i + 1}`;
    const w = parseFloat(c.weight || 0);
    return `<li><strong>${id}</strong>: ${c.criterion || c}${w > 0 ? ` <span class="hitl-weight">(w=${w.toFixed(2)})</span>` : ""}</li>`;
  }).join("");

  inlineDiv.innerHTML = `
    <p class="hitl-prompt">${prompt}</p>
    <div class="hitl-checklist"><h4>Checklist (${checklist.length} items)</h4><ul>${clHtml}</ul></div>
    <div class="hitl-options">
      <label class="hitl-option"><input type="radio" name="hitl-choice" value="A">Looks good, continue</label>
      <label class="hitl-option"><input type="radio" name="hitl-choice" value="B">I want changes</label>
    </div>
    <div id="hitl-edit-area" class="hitl-edit-area hidden">
      <textarea id="hitl-instructions" class="hitl-comment" placeholder="Describe what to change, e.g.:\n• Remove c3 and c5\n• Add a criterion about thermal stability\n• Make c1 more specific to epoxy systems" rows="4"></textarea>
      <button class="btn hitl-apply" id="hitl-apply">Apply Changes</button>
    </div>
    <button class="btn hitl-submit hidden" id="hitl-submit">Confirm & Continue to Search</button>
  `;

  const waitingRow = phasesEl.querySelector(".hitl-waiting");
  if (waitingRow) {
    waitingRow.closest(".phase-block")!.after(inlineDiv);
  } else {
    phasesEl.appendChild(inlineDiv);
  }
  inlineDiv.scrollIntoView({ behavior: "smooth", block: "center" });

  const editArea = document.getElementById("hitl-edit-area")!;
  const submitBtn = document.getElementById("hitl-submit")!;

  // Toggle edit area based on selection
  inlineDiv.querySelectorAll('input[name="hitl-choice"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      const val = (radio as HTMLInputElement).value;
      if (val === "A") {
        editArea.classList.add("hidden");
        submitBtn.classList.remove("hidden");
      } else {
        editArea.classList.remove("hidden");
        submitBtn.classList.add("hidden");
      }
    });
  });

  // "Apply Changes" — send instructions to backend LLM to revise checklist
  document.getElementById("hitl-apply")!.addEventListener("click", async () => {
    const instructions = (document.getElementById("hitl-instructions") as HTMLTextAreaElement).value.trim();
    if (!instructions) { alert("Please describe what to change"); return; }
    const applyBtn = document.getElementById("hitl-apply") as HTMLButtonElement;
    applyBtn.disabled = true;
    applyBtn.textContent = "Revising checklist...";

    try {
      const resp = await authFetch(`${API}/api/hitl-revise/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instructions, current_checklist: checklist }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const result = await resp.json();
      const newChecklist = result.checklist || checklist;
      // Re-mount form with revised checklist
      hitlRendered = false;
      _mountHitlForm(phasesEl, jobId, "Revised checklist below. Review and confirm, or request more changes.", newChecklist);
    } catch (e) {
      applyBtn.disabled = false;
      applyBtn.textContent = "Apply Changes";
      alert(`Revision failed: ${e}`);
    }
  });

  // "Confirm & Continue" — resume pipeline with current checklist
  submitBtn.addEventListener("click", async () => {
    (submitBtn as HTMLButtonElement).disabled = true;
    submitBtn.textContent = "Submitting...";
    try {
      const resp = await authFetch(`${API}/api/hitl-response/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ choice: "A", comment: "", modifications: { checklist } }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      inlineDiv.innerHTML = `<div class="hitl-resuming"><span class="badge badge-running">resuming</span> Pipeline resuming with ${checklist.length} checklist items...</div>`;
      hitlRendered = false;
      statusCard.scrollIntoView({ behavior: "smooth" });
    } catch (e) {
      (submitBtn as HTMLButtonElement).disabled = false;
      submitBtn.textContent = "Confirm & Continue to Search";
      alert(`Failed: ${e}`);
    }
  });
}
