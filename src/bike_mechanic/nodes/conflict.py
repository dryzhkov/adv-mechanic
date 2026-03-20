"""Conflict resolution between manual and web sources."""

from langchain_openai import ChatOpenAI

from bike_mechanic.config import LLM_MODEL
from bike_mechanic.state import GraphState

CONFLICT_PROMPT = """Compare the service manual data with community/web sources for this motorcycle question.

Question: {query}

SERVICE MANUAL DATA:
{manual_context}

WEB/COMMUNITY DATA:
{web_context}

Analyze:
1. Do the sources agree on specifications (torque values, clearances, etc.)?
2. If they differ, is this likely a manufacturer update, regional difference, or error?

Respond in this format:
HAS_CONFLICT: <yes|no>
DETAILS: <explanation of agreement or conflict, with specific values if applicable>"""


def resolve_conflicts(state: GraphState) -> dict:
    """Compare manual data with web results and flag discrepancies."""
    docs = state.get("retrieved_docs", [])
    web_results = state.get("web_results", [])
    query = state["query"]

    if not docs or not web_results:
        return {"has_conflict": False, "conflict_details": ""}

    manual_context = "\n".join(
        f"[Manual p.{d.page_number}] {d.text[:300]}" for d in docs[:3]
    )
    web_context = "\n".join(
        f"[{r['source']}] {r['text'][:300]}" for r in web_results[:3]
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = llm.invoke(
        CONFLICT_PROMPT.format(
            query=query,
            manual_context=manual_context,
            web_context=web_context,
        )
    )

    text = response.content
    has_conflict = False
    details = ""

    for line in text.strip().split("\n"):
        if line.startswith("HAS_CONFLICT:"):
            has_conflict = "yes" in line.lower()
        elif line.startswith("DETAILS:"):
            details = line.split(":", 1)[1].strip()

    if not details:
        details = text

    return {"has_conflict": has_conflict, "conflict_details": details}
