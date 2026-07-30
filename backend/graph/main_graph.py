"""Main LangGraph pipeline: IDCA → SSR(subgraph) → Search → Eval(subgraph) → Report.

P1: Subgraphs for SSR (self-loop) and Eval (Map-Reduce via Send API).
P2: HITL via manual two-phase split (interrupt_after unreliable on Cloud Run astream).
"""

from langgraph.graph import END, StateGraph

from graph.eval_subgraph import build_eval_subgraph
from graph.extraction_subgraph import build_extraction_subgraph
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


def build_graph(checkpointer=None, hitl_enabled=False, phase="all"):
    """Build the pipeline graph.

    phase="all": full pipeline (default)
    phase="first_half": IDCA → SSR → END (for HITL pause after checklist)
    phase="second_half": Search → Eval → Report → END (resume after HITL)
    """
    import os
    # Phase 2 implementation: the extraction subgraph (candidate inventions +
    # claim-language elements with verbatim evidence). EXTRACTOR=ssr restores
    # the legacy SSR checklist path.
    phase2 = build_ssr_subgraph if os.environ.get("EXTRACTOR", "extraction") == "ssr" else build_extraction_subgraph
    g = StateGraph(GraphState)

    if phase == "first_half":
        g.add_node("idca", idca_node)
        g.add_node("ssr", phase2())
        g.add_node("report", report_node)
        g.set_entry_point("idca")
        g.add_conditional_edges("idca", route_after_idca, {
            "ssr": "ssr",
            "report": "report",
        })
        g.add_edge("ssr", END)
        g.add_edge("report", END)

    elif phase == "second_half":
        g.add_node("search", search_node)
        g.add_node("evaluate", build_eval_subgraph())
        g.add_node("report", report_node)
        g.set_entry_point("search")
        g.add_conditional_edges("search", route_after_search, {
            "evaluate": "evaluate",
            "report": "report",
        })
        g.add_edge("evaluate", "report")
        g.add_edge("report", END)

    else:
        g.add_node("idca", idca_node)
        g.add_node("ssr", phase2())
        g.add_node("search", search_node)
        g.add_node("evaluate", build_eval_subgraph())
        g.add_node("report", report_node)
        g.set_entry_point("idca")
        g.add_conditional_edges("idca", route_after_idca, {
            "ssr": "ssr",
            "report": "report",
        })
        g.add_edge("ssr", "search")
        g.add_conditional_edges("search", route_after_search, {
            "evaluate": "evaluate",
            "report": "report",
        })
        g.add_edge("evaluate", "report")
        g.add_edge("report", END)

    return g.compile(checkpointer=checkpointer)
