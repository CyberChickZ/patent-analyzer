"""
Patent Analyzer — Standalone FastAPI Web Service.

Standalone web service. Only needs:
  - ANTHROPIC_API_KEY (for Claude API)
  - SERPAPI_KEY (for patent/paper search)

Free (no key): OpenAlex (abstracts), sentence-transformers (embeddings)

Deploy: docker compose up  →  http://localhost:8000
GCP:    gcloud run deploy   (see deploy/cloudrun.yaml)
"""

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

app = FastAPI(title="Patent Analyzer", version="0.2.0")

# Serve static files (uploaded reports, etc.)
OUTPUT_BASE = Path(os.getenv("OUTPUT_DIR", "./outputs"))
OUTPUT_BASE.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# In-memory job store (replace with Redis/Firestore for production)
jobs: dict[str, dict] = {}


@app.get("/")
async def index():
    """Serve the upload page."""
    return HTMLResponse(UPLOAD_HTML)


@app.post("/analyze")
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a paper and start the full analysis pipeline."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_BASE / job_id
    job_dir.mkdir(parents=True)

    # Save uploaded file
    input_path = job_dir / file.filename
    content = await file.read()
    input_path.write_bytes(content)

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "filename": file.filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "starting",
        "phases": {},
        "output_dir": str(job_dir),
        "input_path": str(input_path),
    }

    # Run pipeline in background
    background_tasks.add_task(run_pipeline, job_id)

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/report/{job_id}")
async def get_report(job_id: str):
    """Serve the generated HTML report."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    report_path = Path(jobs[job_id]["output_dir"]) / "report.html"
    if not report_path.exists():
        raise HTTPException(404, "Report not generated yet")
    return FileResponse(str(report_path), media_type="text/html")


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Return raw results.json."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    results_path = Path(jobs[job_id]["output_dir"]) / "results.json"
    if not results_path.exists():
        raise HTTPException(404, "Results not ready")
    return FileResponse(str(results_path), media_type="application/json")


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return [{"id": j["id"], "status": j["status"], "phase": j["phase"], "filename": j["filename"]}
            for j in jobs.values()]


# ─── Pipeline ───────────────────────────────────────────────────────

async def run_pipeline(job_id: str):
    """Full async pipeline — all phases."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from app.llm import (
        detect_invention, classify_document, summarize_invention,
        classify_category, decompose_invention, generate_checklist,
        plan_delegation, evaluate_batch, generate_overall_summary,
    )
    from patent_analyzer.query_builder import build_all_queries
    from patent_analyzer.scorer import compute_total_score, classify_risk

    job = jobs[job_id]
    job_dir = Path(job["output_dir"])
    input_path = job["input_path"]

    def update(phase: str, status: str, data: dict | None = None):
        job["phase"] = phase
        job["status"] = status
        if data:
            job["phases"][phase] = data

    try:
        # ── Phase 1: IDCA ──
        update("phase1", "running")

        # Read paper text (for PDF, use first pages)
        if input_path.endswith(".pdf"):
            import fitz
            doc = fitz.open(input_path)
            paper_text = "\n".join(page.get_text() for page in doc[:10])
            doc.close()
        else:
            paper_text = Path(input_path).read_text()

        detection = await detect_invention(paper_text)
        if detection["status"] == "absent":
            update("phase1", "completed", {"status": "absent", "raw": detection["raw"]})
            job["status"] = "completed"
            return

        doc_type = await classify_document(paper_text)
        summary = await summarize_invention(paper_text)
        category = await classify_category(summary)

        phase1 = {
            "status": detection["status"],
            "doc_mode": doc_type,
            "summary": summary,
            "invention_type": category["invention_type"],
            "reasoning": category["reasoning"],
        }
        save_json(phase1, job_dir / "phase1.json")
        update("phase1", "completed", phase1)

        # ── Phase 2: Decomposition ──
        update("phase2", "running")

        ucd = await decompose_invention(paper_text)
        checklist = await generate_checklist(summary, ucd)
        delegation = await plan_delegation(summary, ucd, category["invention_type"])

        phase2 = {"ucd": ucd, "checklist": checklist, "delegation": delegation}
        save_json(phase2, job_dir / "phase2.json")
        update("phase2", "completed", {"checklist_count": len(checklist), "atoms": len(delegation.get("atoms", []))})

        # ── Phase 3: Search ──
        update("phase3", "running")

        queries = build_all_queries(delegation)
        save_json(queries, job_dir / "queries.json")

        # Run SerpAPI search
        from patent_analyzer.searcher import serpapi_search, download_pdf, save_incremental
        api_key = os.environ.get("SERPAPI_KEY", "")
        if not api_key:
            update("phase3", "error", {"error": "SERPAPI_KEY not set"})
            job["status"] = "error"
            return

        all_patents, all_papers = [], []
        seen_p, seen_s = set(), set()
        search_groups = []

        for group in queries.get("groups", []):
            gid = group.get("group_id", "")
            gr = {"group_id": gid, "label": group.get("label", ""), "patent_matches_found": 0, "paper_matches_found": 0}

            for q in group.get("patent_queries", []):
                matches = serpapi_search("google_patents", q, api_key, max_pages=1, num_per_page=100)
                for m in matches:
                    if m["pub_num"] and m["pub_num"] not in seen_p:
                        seen_p.add(m["pub_num"])
                        all_patents.append(m)
                        gr["patent_matches_found"] += 1
                gr["patent_query"] = q
                await asyncio.sleep(0.5)

            for q in group.get("paper_queries", []):
                matches = serpapi_search("google_scholar", q, api_key, max_pages=3, num_per_page=20)
                for m in matches:
                    key = m.get("pub_num") or m.get("title", "")
                    if key and key not in seen_s:
                        seen_s.add(key)
                        all_papers.append(m)
                        gr["paper_matches_found"] += 1
                gr["paper_query"] = q
                await asyncio.sleep(0.5)

            search_groups.append(gr)

        # Download PDFs
        papers_dir = job_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        for doc in all_patents + all_papers:
            links = doc.get("pdf_link", "")
            if isinstance(links, str) and links:
                links = [links]
            elif not isinstance(links, list):
                continue
            pub = doc.get("pub_num", "unknown").replace("/", "_").replace(" ", "_")
            for i, link in enumerate(links[:1]):
                local = download_pdf(link, papers_dir, f"{pub}.pdf")
                if local:
                    doc["local_pdf"] = local
                    break

        search_data = {
            "groups": search_groups,
            "all_patents": all_patents,
            "all_papers": all_papers,
            "summary": {"total_patents": len(all_patents), "total_papers": len(all_papers)},
        }
        save_json(search_data, job_dir / "phase3_search.json")
        update("phase3", "completed", {"patents": len(all_patents), "papers": len(all_papers)})

        # ── Phase 3b: Semantic ranking ──
        update("phase3b", "running")
        try:
            from patent_analyzer.semantic_search import rerank_by_embedding
            all_docs = all_patents + all_papers
            ranked = rerank_by_embedding(summary, all_docs, limit=30)
            update("phase3b", "completed", {"ranked": len(ranked)})
        except Exception as e:
            # Fallback to keyword ranking if embedding fails
            ranked = sorted(all_docs, key=lambda x: x.get("_relevance", 0), reverse=True)[:30]
            update("phase3b", "completed", {"ranked": len(ranked), "fallback": str(e)})

        # Filter to those with PDFs
        eval_candidates = [d for d in ranked if d.get("local_pdf") and Path(d["local_pdf"]).exists()][:20]

        # ── Phase 4: Deep evaluation ──
        update("phase4", "running")

        eval_results = await evaluate_batch(summary, checklist, eval_candidates, max_concurrent=5)

        # Score and sort
        scoring_report = []
        for er in eval_results:
            cr = er.get("checklist_results", {})
            score = compute_total_score(cr)
            scoring_report.append({
                "title": er.get("title", ""),
                "id": er.get("pub_num", ""),
                "manuscript_type": er.get("match_type", ""),
                "similarity_score": score,
                "similarity_categories": cr,
                "anticipation_assessment": er.get("anticipation_assessment", ""),
                "key_teachings": er.get("key_teachings", ""),
                "snippet": er.get("snippet", ""),
                "abstract": er.get("abstract", ""),
            })

        scoring_report.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_score = max((s["similarity_score"] for s in scoring_report), default=0)

        overall = await generate_overall_summary(summary, scoring_report[:10])

        update("phase4", "completed", {"evaluated": len(scoring_report), "top_score": round(top_score, 4)})

        # ── Phase 5: Report ──
        update("phase5", "running")

        results = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phase1": phase1,
            "phase2": phase2,
            "search": {"groups": search_groups, "summary": search_data["summary"]},
            "evaluation": {
                "scoring_report": scoring_report,
                "summary": overall,
                "stats": {
                    "total_evaluated": len(scoring_report),
                    "top_score": round(top_score, 4),
                    "risk_level": classify_risk(top_score),
                },
            },
        }
        save_json(results, job_dir / "results.json")

        # Generate HTML report
        from patent_analyzer.report_generator import generate_html
        html = generate_html(results)
        (job_dir / "report.html").write_text(html, encoding="utf-8")

        update("phase5", "completed")
        job["status"] = "completed"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        import traceback
        job["traceback"] = traceback.format_exc()


def save_json(data: Any, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ─── Upload Page HTML ───────────────────────────────────────────────

UPLOAD_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Patent Analyzer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--accent:#2563eb}
body{background:#f7f8fa;display:flex;justify-content:center;padding:3rem 1rem}
.wrap{max-width:640px;width:100%}
h1{font-size:1.6rem;text-align:center;margin-bottom:0.3rem}
.sub{text-align:center;color:#6b7280;font-size:0.88rem;margin-bottom:2rem}
.card{background:#fff;border:1px solid #e2e6ec;border-radius:12px;padding:2rem}
.drop{border:2px dashed #d1d5db;border-radius:10px;padding:2.5rem;text-align:center;cursor:pointer;transition:border-color .2s}
.drop:hover,.drop.over{border-color:var(--accent);background:#eff6ff}
.drop input{display:none}
.drop p{color:#6b7280;font-size:0.9rem;margin-top:0.5rem}
.drop .icon{font-size:2rem;color:#9ca3af}
.fname{margin:1rem 0;font-size:0.88rem;color:#374151;font-family:monospace}
.btn{display:block;width:100%;padding:0.7rem;border:none;border-radius:8px;background:var(--accent);color:#fff;font-size:0.95rem;font-weight:600;cursor:pointer;margin-top:1rem}
.btn:hover{background:#1d4ed8}
.btn:disabled{background:#9ca3af;cursor:not-allowed}
.status{margin-top:1.5rem;padding:1rem;background:#f0f3f7;border-radius:8px;font-size:0.85rem;display:none}
.status.show{display:block}
.phase{padding:0.3rem 0;display:flex;justify-content:space-between}
.phase .label{color:#374151}
.phase .st{font-weight:600}
.st-completed{color:#16a34a}
.st-running{color:var(--accent)}
.st-error{color:#dc2626}
.report-link{display:block;text-align:center;margin-top:1rem;font-size:0.95rem;color:var(--accent);font-weight:600}
.keys{margin-top:2rem;font-size:0.78rem;color:#9ca3af;text-align:center}
</style>
</head><body>
<div class="wrap">
<h1>Patent Analyzer</h1>
<div class="sub">Upload a paper to analyze patentability and search prior art</div>
<div class="card">
  <div class="drop" id="drop" onclick="document.getElementById('fileIn').click()">
    <div class="icon">&#x1F4C4;</div>
    <p>Drop PDF here or click to upload</p>
    <input type="file" id="fileIn" accept=".pdf,.txt,.md" onchange="fileSelected(this)">
  </div>
  <div class="fname" id="fname"></div>
  <button class="btn" id="startBtn" onclick="startAnalysis()" disabled>Analyze</button>
  <div class="status" id="status"></div>
  <a class="report-link" id="reportLink" style="display:none" target="_blank">View Report →</a>
</div>
<div class="keys">Requires: ANTHROPIC_API_KEY + SERPAPI_KEY</div>
</div>
<script>
let selectedFile=null;
const drop=document.getElementById('drop');
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over')});
drop.addEventListener('dragleave',()=>drop.classList.remove('over'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');if(e.dataTransfer.files.length){selectedFile=e.dataTransfer.files[0];showFile()}});
function fileSelected(input){if(input.files.length){selectedFile=input.files[0];showFile()}}
function showFile(){document.getElementById('fname').textContent=selectedFile.name;document.getElementById('startBtn').disabled=false}
async function startAnalysis(){
  if(!selectedFile)return;
  const fd=new FormData();fd.append('file',selectedFile);
  document.getElementById('startBtn').disabled=true;
  const resp=await fetch('/analyze',{method:'POST',body:fd});
  const data=await resp.json();
  if(data.job_id)pollStatus(data.job_id);
}
async function pollStatus(jobId){
  const el=document.getElementById('status');
  el.classList.add('show');
  const poll=async()=>{
    const resp=await fetch('/status/'+jobId);
    const job=await resp.json();
    let html='';
    const phases=['phase1','phase2','phase3','phase3b','phase4','phase5'];
    const labels={'phase1':'IDCA','phase2':'Decomposition','phase3':'Search','phase3b':'Semantic Ranking','phase4':'Deep Evaluation','phase5':'Report'};
    for(const p of phases){
      const ph=job.phases[p];
      const st=job.phase===p?job.status:(ph?'completed':'pending');
      html+=`<div class="phase"><span class="label">${labels[p]}</span><span class="st st-${st}">${st}</span></div>`;
    }
    el.innerHTML=html;
    if(job.status==='completed'){
      document.getElementById('reportLink').href='/report/'+jobId;
      document.getElementById('reportLink').style.display='block';
    }else if(job.status==='error'){
      el.innerHTML+=`<div class="phase"><span class="label">Error</span><span class="st st-error">${job.error||'unknown'}</span></div>`;
    }else{
      setTimeout(poll,2000);
    }
  };
  poll();
}
</script>
</body></html>
"""
