"""LangGraph state definition."""

from typing import TypedDict

from adv_mechanic.search import SearchResult


class WebResult(TypedDict):
    """A result from web search."""

    text: str
    source: str
    url: str


class GraphState(TypedDict, total=False):
    """State that flows through the LangGraph agent."""

    # Input
    query: str
    bike_model: str

    # Router
    query_type: str  # "lookup", "procedural", "general"

    # Retrieval
    retrieved_docs: list[SearchResult]

    # Grading
    retrieval_grade: str  # "sufficient", "partial", "insufficient"
    retrieval_confidence: str  # "high", "medium", "low"

    # Web search
    web_results: list[WebResult]
    web_search_query: str

    # Conflict resolution
    has_conflict: bool
    conflict_details: str

    # Output
    answer: str
    sources: list[str]
    safety_disclaimer: str
    confidence_score: float  # 0.0 to 1.0
