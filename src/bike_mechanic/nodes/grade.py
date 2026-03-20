"""Retrieval grading node."""

import re

from langchain_openai import ChatOpenAI

from bike_mechanic.config import LLM_MODEL
from bike_mechanic.state import GraphState

# Pattern to detect numeric specs in retrieved docs
_SPEC_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(Nm|mm|cc|lbf|ft\s*lb|in|psi|bar|L|ml)",
    re.IGNORECASE,
)

GRADE_PROMPT = """You are grading retrieved service manual content for a motorcycle Q&A system.

Question: {query}
Question type: {query_type}

Retrieved content:
{context}

Grade the retrieval:
1. Does the content address the question?
2. Does it fully answer it?
3. For lookup questions (torque specs, clearances, capacities): does the content contain the SPECIFIC numeric value requested? If no numeric specification is present, grade as INSUFFICIENT regardless of topic relevance.
4. How confident are you that the answer is in this content?

Respond in exactly this format:
GRADE: <sufficient|partial|insufficient>
CONFIDENCE: <high|medium|low>
REASONING: <one line>"""


def grade(state: GraphState) -> dict:
    """Grade the relevance and completeness of retrieved documents."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    query_type = state.get("query_type", "general")

    if not docs:
        return {
            "retrieval_grade": "insufficient",
            "retrieval_confidence": "low",
            "web_search_query": _build_web_query(query, state.get("bike_model", "")),
        }

    # For lookup queries: quick check if ANY doc contains a numeric spec
    if query_type == "lookup":
        has_spec = any(_SPEC_PATTERN.search(doc.text) for doc in docs[:5])
        if not has_spec:
            return {
                "retrieval_grade": "insufficient",
                "retrieval_confidence": "low",
                "web_search_query": _build_web_query(
                    query, state.get("bike_model", "")
                ),
            }

    # Format top docs for the LLM — full text, no truncation
    doc_texts = []
    for i, doc in enumerate(docs[:5]):
        doc_texts.append(
            f"[Doc {i + 1}] p.{doc.page_number} | {doc.content_type} | {doc.section}\n"
            f"{doc.text}"
        )
    context = "\n\n".join(doc_texts)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = llm.invoke(
        GRADE_PROMPT.format(query=query, query_type=query_type, context=context)
    )

    retrieval_grade = "insufficient"
    confidence = "low"

    for line in response.content.strip().split("\n"):
        if line.startswith("GRADE:"):
            g = line.split(":", 1)[1].strip().lower()
            if g in ("sufficient", "partial", "insufficient"):
                retrieval_grade = g
        elif line.startswith("CONFIDENCE:"):
            c = line.split(":", 1)[1].strip().lower()
            if c in ("high", "medium", "low"):
                confidence = c

    result = {
        "retrieval_grade": retrieval_grade,
        "retrieval_confidence": confidence,
    }

    # Prepare web search query if retrieval wasn't fully sufficient
    if retrieval_grade != "sufficient" or confidence != "high":
        result["web_search_query"] = _build_web_query(
            query, state.get("bike_model", "")
        )

    return result


def _build_web_query(query: str, bike_model: str) -> str:
    """Build a web search query from the original question."""
    parts = []
    if bike_model:
        parts.append(bike_model)
    parts.append(query)
    return " ".join(parts)
