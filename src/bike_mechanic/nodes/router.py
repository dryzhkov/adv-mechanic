"""Query classification node."""

from langchain_openai import ChatOpenAI

from bike_mechanic.config import LLM_MODEL
from bike_mechanic.search import resolve_bike_model
from bike_mechanic.state import GraphState

ROUTER_PROMPT = """Classify this motorcycle service question and extract the bike model.

Question: {query}

Respond in exactly this format:
QUERY_TYPE: <lookup|procedural|general>
BIKE_MODEL: <extracted bike model or empty>

Rules:
- "lookup" = specific spec, torque value, clearance, fluid capacity
- "procedural" = how-to, step-by-step procedure
- "general" = recommendations, comparisons, general info"""


def router(state: GraphState) -> dict:
    """Classify the query type and extract bike model if not provided."""
    query = state["query"]

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = llm.invoke(ROUTER_PROMPT.format(query=query))

    query_type = "general"
    bike_model = state.get("bike_model", "")

    for line in response.content.strip().split("\n"):
        if line.startswith("QUERY_TYPE:"):
            qt = line.split(":", 1)[1].strip().lower()
            if qt in ("lookup", "procedural", "general"):
                query_type = qt
        elif line.startswith("BIKE_MODEL:") and not bike_model:
            bm = line.split(":", 1)[1].strip()
            if bm:
                bike_model = bm

    # Resolve shorthand to canonical model name from the vector store
    bike_model = resolve_bike_model(bike_model)

    return {"query_type": query_type, "bike_model": bike_model}
