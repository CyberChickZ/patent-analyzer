"""Main LangGraph pipeline: IDCA → Extract(subgraph) → Search → Eval(subgraph) → Report,
with a gate after each phase.

A gate (graph/gates.py) is a no-op unless its phase is listed in
state["pause_after"]; then it interrupt()s with the phase's editable
output and applies the reviewer's response on resume. With the default
pause_after=[] the graph runs exactly as before. Pausing needs a
checkpointer and a thread_id in the config.
"""

from langgraph.graph import END, StateGraph

from graph.eval_subgraph import build_eval_subgraph
from graph.extraction_subgraph import build_extraction_subgraph
from graph.gates import make_gate
from graph.ssr_subgraph import build_ssr_subgraph  # legacy; kept for EXTRACTOR=ssr
from nodes.idca import idca_node
from nodes.report import report_node
from nodes.search import search_node
from state import GraphState


def route_after_idca(state: GraphState) -> str:
    status = state.get("status_determination", "Present")
    doc_type = state.get("doc_type", "invention")
    if status == "Absent" or doc_type == "talks_about_invention_but_no_invention":
        return "report"
    return "ssr"


def route_after_search(state: GraphState) -> str:
    if state.get("status") == "failed_recall":
        return "report"
    ranked = state.get("ranked_candidates", [])
    if not ranked:
        return "report"
    return "evaluate"


def build_graph(checkpointer=None, phase2=None, nodes: dict | None = None, entry: str = "idca"):
    """Compile the pipeline. `nodes` lets tests swap phase nodes for fakes;
    `phase2` overrides the Phase 2 subgraph builder (default: extraction,
    EXTRACTOR=ssr → legacy SSR); `entry` starts mid-pipeline from a saved
    state (fallback resume when the checkpoint is gone)."""
    import os
    if phase2 is None:
        phase2 = build_ssr_subgraph if os.environ.get("EXTRACTOR", "extraction") == "ssr" else build_extraction_subgraph
    n = {"idca": idca_node, "ssr": phase2(), "search": search_node, "evaluate": build_eval_subgraph(),
         "report": report_node, **(nodes or {})}
    g = StateGraph(GraphState)
    for name in ("idca", "ssr", "search", "evaluate", "report"):
        g.add_node(name, n[name])
    for phase in ("idca", "extract", "search", "evaluate"):
        g.add_node(f"gate_{phase}", make_gate(phase))
    g.set_entry_point(entry)
    g.add_edge("idca", "gate_idca")
    g.add_conditional_edges("gate_idca", route_after_idca, {"ssr": "ssr", "report": "report"})
    g.add_edge("ssr", "gate_extract")
    g.add_edge("gate_extract", "search")
    g.add_edge("search", "gate_search")
    g.add_conditional_edges("gate_search", route_after_search, {"evaluate": "evaluate", "report": "report"})
    g.add_edge("evaluate", "gate_evaluate")
    g.add_edge("gate_evaluate", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
