# intent_router.py - DIAGRAM BOX: "intent routing".
# Decides WHICH search to run, because different questions need different tools:
#   "Which HR policies do we have from 2025?"          -> METADATA (SQL filter on department/year, no meaning needed)
#   "How many days of paid leave do I get?"            -> SEMANTIC (meaning-based vector search)
#   "What does the IT policy say about VPN on phones?"  -> HYBRID (filter department=it AND search by meaning)
# The same LLM call also REWRITES follow-up questions ("and for interns?") into a standalone search text,
# because "and for interns?" alone finds nothing - "paid leave days for interns" does.
# With LangChain the output is a typed RouteDecision object (structured output), so there's no JSON parsing in our code.

import json  # To show the model the known filter values.

from langchain_core.exceptions import OutputParserException  # Raised if the output can't be parsed.
from pydantic import ValidationError  # Raised if it has the wrong shape.

from backend.chat.llm import structured_chain  # LangChain helper: prompt | model.with_structured_output(...).
from backend.chat.metadata_search import get_known_values  # Real values that exist in Cosmos (departments, doc types...).
from backend.models import RouteDecision  # Output model.

ROUTER_PROMPT = """You route questions for a company document assistant.
Rules:
- metadata: the user asks ABOUT documents (list/find/which documents, by department, type, year, file name). Fill the filters.
- semantic: the user asks about CONTENT and names no filter (e.g. "how do I reset my password?"). Leave the filters empty.
- hybrid: the user asks about content AND names a filter (e.g. "what does the HR policy say about maternity leave?").
- Only use filter values from the KNOWN VALUES list (lowercase). If nothing fits, leave the filter empty.
- keyword: only for an exact code/term that must appear literally (e.g. "VPN", "ISO-27001").
- search_text: replace words like "it", "that", "they" with what they refer to in the conversation."""

router = structured_chain(ROUTER_PROMPT, RouteDecision)  # Chain: prompt | Azure model -> RouteDecision. Built once.


def route_question(question: str, history: list[dict]) -> RouteDecision:  # Question (+ chat so far) -> decision.
    conversation = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history) or "(no previous messages)"  # Chat as plain text.
    router_input = (  # Everything the router needs to decide (braces inside VALUES are fine - only the template is parsed):
        f"KNOWN VALUES: {json.dumps(get_known_values())}\n\n"  # e.g. {"department": ["hr","it"], "docType": ["policy"], ...}
        f"CONVERSATION:\n{conversation}\n\n"  # Previous turns (for rewriting follow-ups).
        f"QUESTION: {question}"  # The new question.
    )
    try:
        decision = router.invoke({"input": router_input})  # Run the chain -> RouteDecision object.
    except (OutputParserException, ValidationError):  # Unusable output?
        decision = None  # Handled below.
    if decision is None:  # No valid decision?
        return RouteDecision(intent="semantic", search_text=question, reason="router fallback")  # Safe default: plain semantic search.
    if decision.intent == "metadata" and decision.filters.is_empty():  # "metadata" but no filter found?
        decision.intent = "semantic"  # A SQL query without a WHERE would return random chunks - use meaning instead.
    return decision  # The route.
