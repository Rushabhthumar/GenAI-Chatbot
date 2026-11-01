# pipeline.py - the WHOLE bottom half of the diagram, in order. This is the "fixed workflow" style of GenAI app:
# OUR code decides every step; LangChain chains are used inside specific boxes (guardrails, routing, retrieval, answering).
#
#   user question
#     -> guardrails ---------------------------(blocked)--> "query fails" -> response
#     -> intent routing
#          -> metadata:  search with metadata -> sql query -> cosmos -> all matching chunks  --\
#          -> semantic:  query -> embedding -> semantic search (cosmos vectors) -> results   ---> merge
#     -> LLM chat completion (context + user query)  -> response (streamed)
#
# The function is a GENERATOR: it `yield`s small event dicts, and main.py forwards each one over the WebSocket.
# So Angular can show every step live ("guardrails... done", "routing: hybrid", "3 chunks found"...).

from collections.abc import Iterator  # Return type: a stream of event dicts.

from backend import config  # MIN_SIMILARITY for display.
from backend.chat.generator import NOT_FOUND, build_sources, stream_answer  # Generation (LangChain answer chain).
from backend.chat.guardrails import BLOCKED_MESSAGES, run_guardrails  # Guardrails (LangChain classifier chain).
from backend.chat.intent_router import route_question  # Intent routing (LangChain router chain).
from backend.chat.merge import merge_results  # Merge both branches.
from backend.chat.metadata_search import metadata_search  # SQL branch.
from backend.chat.semantic_search import retriever  # Vector branch (LangChain retriever).


def step(name: str, status: str, detail: str = "") -> dict:  # Helper: one "progress" event for the UI trace.
    return {"type": "step", "name": name, "status": status, "detail": detail}  # e.g. {"type":"step","name":"guardrails","status":"done"}


def run_pipeline(question: str, history: list[dict]) -> Iterator[dict]:  # question + history -> stream of events.
    # ---- BOX: guardrails (security) filters/LLMs ----
    yield step("guardrails", "running")  # Tell the UI we started.
    verdict = run_guardrails(question)  # Rules + LLM classifier chain + Azure content filter.
    if not verdict.allowed:  # BOX: "query fails" -> "response".
        yield step("guardrails", "done", f"blocked: {verdict.category}")  # Show why in the trace.
        yield {"type": "blocked", "category": verdict.category, "message": BLOCKED_MESSAGES.get(verdict.category, "Request blocked.")}
        yield {"type": "done"}  # End of this answer.
        return  # Stop the generator - nothing else runs (no search, no answer LLM = no cost, no leak).
    yield step("guardrails", "done", "safe")  # Passed.

    # ---- BOX: intent routing ----
    yield step("intent_routing", "running")  # Started.
    decision = route_question(question, history)  # Router chain: metadata / semantic / hybrid (+ rewrites the question).
    filters_used = {k: v for k, v in decision.filters.model_dump().items() if v is not None}  # Only filled filters, for display.
    yield step("intent_routing", "done", f"{decision.intent} | search: '{decision.search_text}' | filters: {filters_used} | {decision.reason}")

    semantic_docs = []  # Results of the vector branch.
    metadata_docs = []  # Results of the SQL branch.

    # ---- BOXES: search with metadata -> sql query -> searches into cosmos db -> all matching chunks ----
    if decision.intent in ("metadata", "hybrid") and not decision.filters.is_empty():  # Only when we have filters.
        yield step("metadata_search", "running")  # Started.
        metadata_docs = metadata_search(decision.filters)  # Parameterised SQL on metadata.
        yield step("metadata_search", "done", f"{len(metadata_docs)} chunks matched the filters")  # Count.

    # ---- BOXES: query -> embedding -> semantic search with cosmosdb -> all matching results ----
    if decision.intent in ("semantic", "hybrid"):  # Meaning-based search.
        yield step("semantic_search", "running")  # Started.
        semantic_docs = retriever.invoke(decision.search_text)  # LangChain retriever: embed + VectorDistance in Cosmos.
        yield step("semantic_search", "done", f"{len(semantic_docs)} chunks above similarity {config.MIN_SIMILARITY}")  # Count.

    # ---- BOX: all matching results <-> all matching chunks (merge) ----
    docs = merge_results(semantic_docs, metadata_docs)  # One ranked, de-duplicated list (RRF).
    yield step("merge", "done", f"{len(docs)} chunks sent to the LLM")  # How much context the model gets.

    if not docs:  # Nothing relevant found anywhere?
        yield {"type": "sources", "sources": []}  # No citations.
        yield {"type": "token", "text": NOT_FOUND}  # Fixed reply - we DON'T call the LLM, so it can't invent an answer.
        yield {"type": "done"}  # End.
        return  # Stop.

    yield {"type": "sources", "sources": [s.model_dump() for s in build_sources(docs)]}  # Citations FIRST (UI shows them as chips).

    # ---- BOX: LLMs to chat completion + user query -> response ----
    yield step("generation", "running")  # Started.
    for piece in stream_answer(question, docs, history):  # LangChain answer chain, streamed.
        yield {"type": "token", "text": piece}  # Each piece goes straight to the browser.
    yield step("generation", "done")  # Finished.
    yield {"type": "done"}  # End of the answer.
