"""Test LangGraph pipeline structure and routing logic.

These tests verify the graph compiles correctly and routing decisions
work without making any LLM calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.main_graph import build_graph, route_after_idca, route_after_search


def test_graph_compiles():
    g = build_graph()
    assert g is not None
    assert type(g).__name__ == "CompiledStateGraph"


def test_route_absent():
    state = {"status_determination": "Absent", "doc_type": "invention"}
    assert route_after_idca(state) == "report"


def test_route_no_invention_doc():
    state = {"status_determination": "Present",
             "doc_type": "talks_about_invention_but_no_invention"}
    assert route_after_idca(state) == "report"


def test_route_present():
    state = {"status_determination": "Present", "doc_type": "invention"}
    assert route_after_idca(state) == "ssr"


def test_route_implied():
    state = {"status_determination": "Implied", "doc_type": "invention"}
    assert route_after_idca(state) == "ssr"


def test_route_search_failed():
    state = {"status": "failed_recall", "ranked_candidates": []}
    assert route_after_search(state) == "report"


def test_route_search_empty():
    state = {"status": "running", "ranked_candidates": []}
    assert route_after_search(state) == "report"


def test_route_search_has_results():
    state = {"status": "running", "ranked_candidates": [{"title": "x"}]}
    assert route_after_search(state) == "evaluate"


if __name__ == "__main__":
    test_graph_compiles()
    test_route_absent()
    test_route_no_invention_doc()
    test_route_present()
    test_route_implied()
    test_route_search_failed()
    test_route_search_empty()
    test_route_search_has_results()
    print("All graph structure tests passed!")
