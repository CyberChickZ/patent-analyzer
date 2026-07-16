"""Full end-to-end test: start FastAPI server, upload PDF, poll until done, verify report.

This tests the REAL deployed code path: FastAPI → _pipeline_worker → LangGraph graph.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force LangGraph pipeline + skip auth for testing
os.environ["USE_LANGGRAPH"] = "true"
os.environ["AUTH_DISABLED"] = "true"

import httpx
import uvicorn


async def run_test():
    from app.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=18765, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    await asyncio.sleep(2)  # wait for server startup

    base = "http://127.0.0.1:18765"
    test_pdf = str(Path(__file__).parent.parent / "test_pdfs" / "fake_invention.pdf")

    async with httpx.AsyncClient(timeout=600) as client:
        # Step 1: Upload + start
        print("[E2E] Step 1: Uploading fake_invention.pdf...")
        with open(test_pdf, "rb") as f:
            resp = await client.post(
                f"{base}/analyze?hitl_enabled=false",
                files={"file": ("fake_invention.pdf", f, "application/pdf")},
            )
        assert resp.status_code == 200, f"Upload failed: {resp.status_code} {resp.text}"
        data = resp.json()
        job_id = data["job_id"]
        print(f"[E2E] Job started: {job_id}")

        # Step 2: Poll until completion
        print("[E2E] Step 2: Polling status...")
        max_wait = 600  # 10 min
        start = time.time()
        last_phase = ""
        while time.time() - start < max_wait:
            status_resp = await client.get(f"{base}/status/{job_id}")
            job = status_resp.json()
            phase = job.get("phase", "?")
            status = job.get("status", "?")

            if phase != last_phase:
                elapsed = int(time.time() - start)
                print(f"[E2E]   {elapsed:3d}s — phase={phase}, status={status}")
                last_phase = phase

            if status == "completed":
                print(f"[E2E] Pipeline completed in {int(time.time()-start)}s")
                break
            elif status == "error":
                print(f"[E2E] FAILED: {job.get('error', 'unknown error')}")
                server.should_exit = True
                await task
                return False

            await asyncio.sleep(3)
        else:
            print(f"[E2E] TIMEOUT after {max_wait}s")
            server.should_exit = True
            await task
            return False

        # Step 3: Check events
        events_resp = await client.get(f"{base}/events/{job_id}")
        events = events_resp.json()
        n_events = events.get("total", 0)
        print(f"[E2E] Step 3: {n_events} events recorded")

        # Step 4: Fetch report
        report_resp = await client.get(f"{base}/report/{job_id}")
        if report_resp.status_code == 200:
            report_html = report_resp.text
            print(f"[E2E] Step 4: Report HTML: {len(report_html)} bytes")
            assert "<html" in report_html.lower() or "<!doctype" in report_html.lower(), \
                "Report doesn't look like HTML"
            # Save locally for inspection
            out_path = Path("/tmp/e2e_test_report.html")
            out_path.write_text(report_html)
            print(f"[E2E] Report saved to {out_path}")
        else:
            print(f"[E2E] Report fetch: {report_resp.status_code}")

        # Step 5: Fetch results JSON
        results_resp = await client.get(f"{base}/results/{job_id}")
        if results_resp.status_code == 200:
            results = results_resp.json()
            eval_data = results.get("evaluation", {})
            stats = eval_data.get("stats", {})
            print(f"[E2E] Step 5: Results JSON")
            print(f"[E2E]   evaluated: {stats.get('total_evaluated', '?')}")
            print(f"[E2E]   top_score: {stats.get('top_score', '?')}")
            print(f"[E2E]   risk_level: {stats.get('risk_level', '?')}")
            print(f"[E2E]   summary length: {len(eval_data.get('summary', ''))}")
        else:
            print(f"[E2E] Results fetch: {results_resp.status_code}")

        # Step 6: Verify jobs list
        jobs_resp = await client.get(f"{base}/jobs")
        jobs_list = jobs_resp.json()
        our_job = [j for j in jobs_list if j["id"] == job_id]
        assert our_job, "Job not in jobs list"
        assert our_job[0]["status"] == "completed", f"Job status in list: {our_job[0]['status']}"
        print(f"[E2E] Step 6: Job list verified — status=completed")

    print("\n[E2E] ═══ ALL E2E TESTS PASSED ═══")

    server.should_exit = True
    await task
    return True


if __name__ == "__main__":
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)
