"""Second-pass retrieval to fetch specs for components found in procedural docs."""

from langchain_openai import ChatOpenAI

from bike_mechanic.config import LLM_MODEL
from bike_mechanic.search import search_manuals_hybrid
from bike_mechanic.state import GraphState

EXTRACT_PROMPT = """Below is retrieved service manual content for a procedural motorcycle question.

Question: {query}

Retrieved content:
{context}

List every fastener, fluid fill, gasket, sealant, adjustment, or clearance check mentioned or implied in this procedure that would need a specific numeric spec (torque value, capacity, clearance, part number) to complete the job correctly.

Respond with one item per line, no bullets or numbering. Use the exact component name as it appears in the text. If the spec already appears inline, skip it. Only list items where the spec is MISSING from the text above.

If nothing is missing, respond with exactly: NONE"""


def enrich_specs(state: GraphState) -> dict:
    """Extract component names from procedural docs and fetch their specs."""
    query_type = state.get("query_type", "general")
    if query_type != "procedural":
        return {}

    docs = state.get("retrieved_docs", [])
    if not docs:
        return {}

    bike_model = state.get("bike_model", "")

    # Build context from retrieved docs
    doc_texts = []
    for d in docs[:8]:
        doc_texts.append(f"[p.{d.page_number} | {d.section}]\n{d.text}")
    context = "\n\n".join(doc_texts)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    query = state["query"]
    response = llm.invoke(
        EXTRACT_PROMPT.format(query=query, context=context)
    )

    lines = [
        ln.strip()
        for ln in response.content.strip().split("\n")
        if ln.strip() and ln.strip().upper() != "NONE"
    ]

    if not lines:
        return {}

    # Deduplicate key for existing docs
    seen = {(d.manual_title, d.page_number, d.text[:100]) for d in docs}
    new_docs = []

    for component in lines[:6]:  # cap at 6 targeted lookups
        spec_results = search_manuals_hybrid(
            query=f"{component} torque spec clearance capacity",
            bike_model=bike_model,
            top_k=3,
        )
        for sr in spec_results:
            key = (sr.manual_title, sr.page_number, sr.text[:100])
            if key not in seen:
                seen.add(key)
                new_docs.append(sr)

    if not new_docs:
        return {}

    return {"retrieved_docs": docs + new_docs}
