"""LangGraph workflow definition."""

from langgraph.graph import END, StateGraph

from adv_mechanic.nodes.conflict import resolve_conflicts
from adv_mechanic.nodes.generate import generate
from adv_mechanic.nodes.grade import grade
from adv_mechanic.nodes.retrieve import retrieve
from adv_mechanic.nodes.router import router
from adv_mechanic.nodes.web_search import web_search
from adv_mechanic.state import GraphState


def _should_web_search(state: GraphState) -> str:
    """Decide whether to do web search based on grading."""
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


def build_graph():
    """Build and compile the agentic RAG graph.

    Flow:
        router -> retrieve -> grade --(sufficient+high)--> generate
                                     --(else)--> web_search -> resolve_conflicts -> generate
    """
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("router", router)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("web_search", web_search)
    workflow.add_node("resolve_conflicts", resolve_conflicts)
    workflow.add_node("generate", generate)

    # Edges
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        _should_web_search,
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
