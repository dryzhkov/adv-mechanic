"""RAG retrieval node."""

from adv_mechanic.search import search_manuals, search_manuals_hybrid
from adv_mechanic.state import GraphState


def retrieve(state: GraphState) -> dict:
    """Retrieve relevant chunks from the vector store.

    Uses hybrid search (vector + FTS) for lookup queries to catch
    exact spec values that embeddings may miss.
    """
    query = state["query"]
    bike_model = state.get("bike_model", "")
    query_type = state.get("query_type", "general")

    if query_type == "lookup":
        docs = search_manuals_hybrid(query=query, bike_model=bike_model, top_k=10)
    else:
        docs = search_manuals(query=query, bike_model=bike_model, top_k=10)

    return {"retrieved_docs": docs}
