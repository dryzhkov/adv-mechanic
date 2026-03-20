"""RAG retrieval node."""

from adv_mechanic.search import search_manuals, search_manuals_hybrid
from adv_mechanic.state import GraphState


def retrieve(state: GraphState) -> dict:
    """Retrieve relevant chunks from the vector store.

    Uses hybrid search (vector + FTS) for lookup queries to catch
    exact spec values that embeddings may miss.

    For procedural queries, supplements the main results with a
    hybrid search for related specs and table content so that
    torque values, clearances, and other details are available
    for each step.
    """
    query = state["query"]
    bike_model = state.get("bike_model", "")
    query_type = state.get("query_type", "general")

    if query_type == "lookup":
        docs = search_manuals_hybrid(query=query, bike_model=bike_model, top_k=10)
    elif query_type == "procedural":
        docs = search_manuals(query=query, bike_model=bike_model, top_k=8)
        # Supplement with spec/table chunks that the procedural search may miss
        spec_docs = search_manuals_hybrid(
            query=f"{query} torque specs clearance specifications",
            bike_model=bike_model,
            top_k=5,
        )
        seen = {(d.manual_title, d.page_number, d.text[:100]) for d in docs}
        for sd in spec_docs:
            key = (sd.manual_title, sd.page_number, sd.text[:100])
            if key not in seen:
                seen.add(key)
                docs.append(sd)
    else:
        docs = search_manuals(query=query, bike_model=bike_model, top_k=10)

    return {"retrieved_docs": docs}
