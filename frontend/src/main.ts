import "./style.css";

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
  <h1>Patent Analyzer</h1>
  <div class="sub">Upload a paper or patent PDF — get a full novelty analysis</div>

  <div class="card" id="upload-card">
    <h2>Upload</h2>
    <div class="drop" id="drop">
      <div class="icon">&#x1F4C4;</div>
      <p>Drop PDF here or click to upload</p>
      <input type="file" id="fileIn" accept=".pdf,.txt,.md">
    </div>
    <div class="fname" id="fname"></div>
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
  startBtn.disabled = true;
  errorArea.innerHTML = "";

  const LARGE_FILE_THRESHOLD = 25 * 1024 * 1024; // 25 MB — Cloud Run hard limit is 32 MB
  const useSignedUrl = selectedFile.size > LARGE_FILE_THRESHOLD;

  try {
    let jobId: string;

    if (useSignedUrl) {
      // Step 1: Get signed PUT URL
      startBtn.textContent = "Requesting upload URL...";
      const urlResp = await fetch(`${API}/api/upload-url?filename=${encodeURIComponent(selectedFile.name)}&content_type=application/pdf`);
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
      const analyzeResp = await fetch(`${API}/api/analyze-gcs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: urlData.job_id,
          gcs_uri: urlData.gcs_uri,
          filename: selectedFile.name,
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
      const resp = await fetch(`${API}/api/analyze`, { method: "POST", body: fd });
      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`Upload failed (${resp.status}): ${errText.slice(0, 200)}`);
      }
      const data = await resp.json();
      if (!data.job_id) throw new Error(data.error || "No job_id returned");
      jobId = data.job_id;
    }

    statusCard.classList.remove("hidden");
    startPolling(jobId);
    refreshJobs();
  } catch (e: any) {
    errorArea.innerHTML = `<div class="error-box">${escapeHtml(e.message || String(e))}</div>`;
    statusCard.classList.remove("hidden");
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
        fetch(`${API}/api/status/${jobId}`),
        fetch(`${API}/api/events/${jobId}?since=${allEvents.length}`),
      ]);
      const job: Job = await statusResp.json();
      const eventsData = await eventsResp.json();
      if (eventsData.events) {
        allEvents = allEvents.concat(eventsData.events);
      }
      renderPhases(job);

      if (job.status === "completed") {
        if (pollTimer) clearInterval(pollTimer);
        reportLink.href = `${API}/api/report/${jobId}`;
        reportLink.classList.remove("hidden");
        renderStats(job);
      } else if (job.status === "error") {
        if (pollTimer) clearInterval(pollTimer);
        errorArea.innerHTML = `<div class="error-box">${job.error || "Pipeline error"}</div>`;
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

function hasActiveSelectionInside(container: HTMLElement): boolean {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return false;
  const range = sel.getRangeAt(0);
  return container.contains(range.commonAncestorContainer);
}

function renderPhases(job: Job) {
  // Skip re-render if user is selecting text inside the phases element
  if (hasActiveSelectionInside(phasesEl)) return;
  // Skip re-render if nothing actually changed
  const hash = `${job.status}|${job.phase}|${allEvents.length}|${expandedPhase}`;
  if (hash === lastRenderHash) return;
  lastRenderHash = hash;

  let html = "";
  for (const p of PHASES) {
    let st = "pending";
    if (job.status === "completed") {
      st = "completed";
    } else if (job.phases[p.key]) {
      st = "completed";
    } else if (job.phase === p.key) {
      st = job.status === "error" ? "error" : "running";
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
  }
  phasesEl.innerHTML = html;
  renderedEvents = allEvents.slice();
  // Attach click handlers
  phasesEl.querySelectorAll('.phase-row[data-phase]').forEach((el) => {
    el.addEventListener('click', () => {
      const key = (el as HTMLElement).dataset.phase!;
      expandedPhase = expandedPhase === key ? null : key;
      lastRenderHash = ""; // force re-render
      renderPhases(job);
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
  const p3 = job.phases.phase3 || {};
  const p4 = job.phases.phase4 || {};
  const patents = p3.patents ?? "-";
  const papers = p3.papers ?? "-";
  const evaluated = p4.evaluated ?? "-";
  const topScore = p4.top_score != null ? `${(p4.top_score * 100).toFixed(1)}%` : "-";

  statsArea.innerHTML = `
  <div class="stats">
    <div class="stat"><div class="val">${patents}</div><div class="lbl">Patents Found</div></div>
    <div class="stat"><div class="val">${papers}</div><div class="lbl">Papers Found</div></div>
    <div class="stat"><div class="val">${evaluated}</div><div class="lbl">Evaluated</div></div>
    <div class="stat"><div class="val">${topScore}</div><div class="lbl">Top Similarity</div></div>
  </div>`;
}

// ─── Job list ───

async function refreshJobs() {
  try {
    const resp = await fetch(`${API}/api/jobs`);
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
          await fetch(`${API}/api/jobs/${id}`, { method: "DELETE" });
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

refreshJobs();
