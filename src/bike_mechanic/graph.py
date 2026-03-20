"""LangGraph workflow definition."""

from langgraph.graph import END, StateGraph

from bike_mechanic.nodes.conflict import resolve_conflicts
from bike_mechanic.nodes.enrich_specs import enrich_specs
from bike_mechanic.nodes.generate import generate
from bike_mechanic.nodes.grade import grade
from bike_mechanic.nodes.retrieve import retrieve
from bike_mechanic.nodes.router import router
from bike_mechanic.nodes.web_search import web_search
from bike_mechanic.state import GraphState


def _after_grade(state: GraphState) -> str:
    """Route after grading: enrich specs for procedural, else web search decision."""
    if state.get("query_type") == "procedural":
        return "enrich_specs"
    if (
        state.get("retrieval_grade") == "sufficient"
        and state.get("retrieval_confidence") == "high"
    ):
        return "generate"
    return "web_search"


def _after_enrich(state: GraphState) -> str:
    """After spec enrichment, decide whether web search is still needed."""
    if (
        state.get("retrieval_grade") == "sufficient"
        and state.get("retrieval_confidence") == "high"
    ):
        return "generate"
    return "web_search"


def _should_resolve_conflicts(state: GraphState) -> str:
    """Decide whether conflict resolution is needed."""
    if state.get("retrieved_docs") and state.get("web_results"):
        return "resolve_conflicts"
    return "generate"


# Human-readable labels for each node, used by the CLI progress display
NODE_LABELS = {
    "router": "Classifying query",
    "retrieve": "Searching manuals",
    "grade": "Grading relevance",
    "enrich_specs": "Fetching specs for procedure",
    "web_search": "Searching web",
    "resolve_conflicts": "Checking conflicts",
    "generate": "Generating answer",
}


def build_graph():
    """Build and compile the agentic RAG graph.

    Flow:
        router -> retrieve -> grade --(procedural)--> enrich_specs -> [web_search | generate]
                                     --(sufficient+high)--> generate
                                     --(else)--> web_search -> resolve_conflicts -> generate
    """
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("router", router)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("enrich_specs", enrich_specs)
    workflow.add_node("web_search", web_search)
    workflow.add_node("resolve_conflicts", resolve_conflicts)
    workflow.add_node("generate", generate)

    # Edges
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        _after_grade,
        {"enrich_specs": "enrich_specs", "web_search": "web_search", "generate": "generate"},
    )
    workflow.add_conditional_edges(
        "enrich_specs",
        _after_enrich,
        {"web_search": "web_search", "generate": "generate"},
    )
    workflow.add_conditional_edges(
        "web_search",
        _should_resolve_conflicts,
        {"resolve_conflicts": "resolve_conflicts", "generate": "generate"},
    )

    workflow.add_edge("resolve_conflicts", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
