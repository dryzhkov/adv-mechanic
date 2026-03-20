"""Final answer generation node."""

import re

from langchain_openai import ChatOpenAI

from bike_mechanic.config import LLM_MODEL, SAFETY_CRITICAL_KEYWORDS
from bike_mechanic.state import GraphState

SAFETY_DISCLAIMER = (
    "!! SAFETY-CRITICAL FASTENER: This specification affects rider safety. "
    "Always use a calibrated torque wrench. Apply thread-locking compound "
    "where specified. If uncertain, consult a qualified mechanic or your "
    "dealer. Values shown are for stock components -- aftermarket parts "
    "may require different specifications."
)

GENERATE_PROMPT = """You are an expert motorcycle mechanic assistant. Answer the question using ONLY the provided sources.

Question: {query}
Query type: {query_type}

SERVICE MANUAL DATA:
{manual_context}

WEB/COMMUNITY DATA:
{web_context}

{conflict_note}
{confidence_note}
{safety_note}

CRITICAL RULES:
- NEVER fabricate, estimate, or guess torque values, clearances, fluid capacities, or any numeric specification
- Every numeric spec you state MUST come from the source text above. You may match common mechanic terminology to service manual terms (e.g., "axle bolt" = "wheel spindle screw", "pinch bolt" = "triple clamp screw") but the NUMERIC VALUE must appear verbatim in a source
- If no matching specification exists in the sources, respond: "I could not find the exact specification for [X] in the currently indexed manual data. This may require ingesting additional manual sections — or contact your dealer for verification."
- Do NOT infer specifications from similar components, other model years, or general mechanical knowledge

Guidelines:
- Always cite your sources (e.g., "per service manual p.47" or "per ADVRider forum")
- For lookup queries: lead with the specific value, then context
- For procedural queries: clear step-by-step instructions. EVERY step that involves a fastener, fluid fill, gasket, sealant, or adjustment MUST include the exact spec INLINE — torque value, fluid type and capacity, clearance, or sealant part number (e.g., "Tighten the axle nut to 110 Nm (81 ft-lb)", "Fill with 1.7 L of Motorex Power Synt 4T 10W-50"). Pull these values from the SERVICE MANUAL DATA above. Do not leave any step vague — never write "tighten to spec" or "fill to the correct level" without stating the actual number. If a spec is available in the sources, it MUST appear in the step that needs it
- Use metric units primarily, include imperial in parentheses if available from sources
- When web/community data is available, include a brief "Community tips" section at the end listing practical insights, common mistakes, tool recommendations, or real-world experience shared by other riders. Keep each tip to one or two sentences
- Be direct and practical -- the user is an experienced home mechanic"""

# Pattern to extract numeric specs from generated answers
_SPEC_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(Nm|mm|cc|lbf|ft|in|psi|bar|L|ml)", re.IGNORECASE)


def _is_safety_critical(query: str) -> bool:
    lower = query.lower()
    return any(kw in lower for kw in SAFETY_CRITICAL_KEYWORDS)


def _verify_specs_in_sources(answer: str, docs: list, web_results: list) -> bool:
    """Check if numeric specs in the answer actually appear in source documents."""
    specs_in_answer = _SPEC_PATTERN.findall(answer)
    if not specs_in_answer:
        return True  # No specs to verify

    # Collect all source text
    source_text = ""
    for d in docs:
        source_text += " " + d.text
    for r in web_results:
        source_text += " " + r.get("text", "")

    # Check each spec value appears in sources
    for value, unit in specs_in_answer:
        # Look for the numeric value near the unit in source text
        pattern = re.compile(rf"{re.escape(value)}\s*{re.escape(unit)}", re.IGNORECASE)
        if not pattern.search(source_text):
            return False

    return True


def _compute_confidence_score(
    retrieval_grade: str,
    retrieval_confidence: str,
    has_docs: bool,
    has_web: bool,
    has_conflict: bool,
    specs_verified: bool,
) -> float:
    """Compute a 0.0–1.0 confidence score from graph signals.

    Scoring breakdown:
      - Retrieval grade:      up to 0.35 (sufficient=0.35, partial=0.15, insufficient=0.0)
      - Retrieval confidence:  up to 0.25 (high=0.25, medium=0.15, low=0.0)
      - Source availability:   up to 0.15 (manual=0.15, web-only=0.05, none=0.0)
      - Spec verification:    up to 0.15 (verified=0.15, not=0.0)
      - No conflict:          up to 0.10 (no conflict=0.10, conflict=-0.05)
    """
    score = 0.0

    grade_scores = {"sufficient": 0.35, "partial": 0.15, "insufficient": 0.0}
    score += grade_scores.get(retrieval_grade, 0.0)

    conf_scores = {"high": 0.25, "medium": 0.15, "low": 0.0}
    score += conf_scores.get(retrieval_confidence, 0.0)

    if has_docs:
        score += 0.15
    elif has_web:
        score += 0.05

    if specs_verified:
        score += 0.15

    if has_conflict:
        score -= 0.05
    else:
        score += 0.10

    return round(max(0.0, min(1.0, score)), 2)


def generate(state: GraphState) -> dict:
    """Generate the final answer with citations and disclaimers."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    web_results = state.get("web_results", [])
    has_conflict = state.get("has_conflict", False)
    conflict_details = state.get("conflict_details", "")
    confidence = state.get("retrieval_confidence", "low")
    query_type = state.get("query_type", "general")

    # Build context strings
    manual_context = "No manual data found."
    if docs:
        parts = []
        for d in docs[:5]:
            parts.append(
                f"[Service Manual - {d.bike_model}, p.{d.page_number}, {d.section}]\n{d.text}"
            )
        manual_context = "\n\n".join(parts)

    web_context = "No web data available."
    if web_results:
        parts = []
        for r in web_results[:5]:
            parts.append(f"[{r['source']}]\n{r['text']}")
        web_context = "\n\n".join(parts)

    conflict_note = ""
    if has_conflict:
        conflict_note = (
            f"IMPORTANT - CONFLICTING DATA DETECTED:\n{conflict_details}\n"
            "Clearly note the discrepancy and recommend verifying with the dealer "
            "or checking for Technical Service Bulletins (TSBs)."
        )

    confidence_note = ""
    if confidence == "low":
        confidence_note = "Note: Confidence is LOW. Clearly indicate uncertainty."
    elif confidence == "medium":
        confidence_note = "Note: Confidence is MEDIUM. Suggest verification."

    is_safety = _is_safety_critical(query)
    safety_note = ""
    if is_safety:
        safety_note = f"This involves a SAFETY-CRITICAL component. Include this disclaimer:\n{SAFETY_DISCLAIMER}"

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = llm.invoke(
        GENERATE_PROMPT.format(
            query=query,
            query_type=query_type,
            manual_context=manual_context,
            web_context=web_context,
            conflict_note=conflict_note,
            confidence_note=confidence_note,
            safety_note=safety_note,
        )
    )

    answer = response.content

    # Post-check: verify specs in answer appear in sources
    specs_verified = _verify_specs_in_sources(answer, docs, web_results)
    if not specs_verified:
        answer += (
            "\n\n!! NOTE: Some specifications in this answer could not be verified "
            "against the retrieved source documents. Double-check these values "
            "before use — contact your dealer if in doubt."
        )

    # Compute confidence score
    confidence_score = _compute_confidence_score(
        retrieval_grade=state.get("retrieval_grade", "insufficient"),
        retrieval_confidence=confidence,
        has_docs=bool(docs),
        has_web=bool(web_results),
        has_conflict=has_conflict,
        specs_verified=specs_verified,
    )

    # Collect sources
    sources = []
    for d in docs[:5]:
        sources.append(f"{d.manual_title} p.{d.page_number}")
    for r in web_results[:5]:
        sources.append(r["url"])

    return {
        "answer": answer,
        "sources": sources,
        "safety_disclaimer": SAFETY_DISCLAIMER if is_safety else "",
        "confidence_score": confidence_score,
    }
