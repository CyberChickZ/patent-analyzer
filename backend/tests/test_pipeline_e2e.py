"""End-to-end pipeline test via LangGraph.

Runs the full graph with a test markdown file (no PDF needed, fast).
Requires Vertex AI credentials (gcloud auth application-default login).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_FILE = str(Path(__file__).parent.parent / "test_pdfs" / "fake_invention.md")


async def run_test():
    from graph.main_graph import build_graph

    graph = build_graph()

    initial_state = {
        "job_id": "test-001",
        "input_local_path": TEST_FILE,
        "output_dir": "/tmp/amie-test-001",
        "hitl_enabled": False,
        "evolve": False,
        "status": "running",
        "phase": "phase1",
        "events": [],
        "phase_results": {},
    }

    print(f"[TEST] Starting pipeline with {TEST_FILE}")
    print(f"[TEST] File exists: {Path(TEST_FILE).exists()}")

    result = await graph.ainvoke(initial_state)

    print(f"\n[RESULT] status: {result.get('status')}")
    print(f"[RESULT] phase: {result.get('phase')}")
    print(f"[RESULT] status_determination: {result.get('status_determination')}")
    print(f"[RESULT] doc_type: {result.get('doc_type')}")
    print(f"[RESULT] summary length: {len(result.get('summary', ''))}")
    print(f"[RESULT] checklist items: {len(result.get('checklist', []))}")
    print(f"[RESULT] search results: {len(result.get('search_results', []))}")
    print(f"[RESULT] scoring_report: {len(result.get('scoring_report', []))}")
    print(f"[RESULT] novelty_score: {result.get('novelty_score')}")
    print(f"[RESULT] risk_level: {result.get('risk_level')}")
    print(f"[RESULT] overall_summary length: {len(result.get('overall_summary', ''))}")
    print(f"[RESULT] report_html_gcs: {result.get('report_html_gcs', '')}")
    print(f"[RESULT] events count: {len(result.get('events', []))}")

    # Assertions
    assert result.get("status_determination") in ("Present", "Implied", "Absent"), \
        f"Unexpected status: {result.get('status_determination')}"

    if result.get("status_determination") == "Present":
        assert len(result.get("summary", "")) > 50, "Summary too short"
        assert len(result.get("checklist", [])) > 0, "No checklist items"
        print("\n[TEST] PASSED — Full pipeline completed with checklist")
    elif result.get("status_determination") == "Absent":
        assert result.get("status") == "completed"
        print("\n[TEST] PASSED — Pipeline correctly identified no invention")
    else:
        print(f"\n[TEST] PASSED — Status: {result.get('status_determination')}")

    return result


if __name__ == "__main__":
    result = asyncio.run(run_test())
