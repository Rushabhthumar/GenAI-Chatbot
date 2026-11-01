# models.py - the DATA MODELS (shapes of our data), written with Pydantic.
# Pydantic classes = TypeScript interfaces that ALSO validate at runtime (like class-validator DTOs in NestJS).
# With LangChain they do one more job: llm.with_structured_output(GuardrailResult) turns the class (field names, types
# and the Field(description=...) texts) into a schema the LLM must fill in, and gives us back a typed object, not a string.
# Text itself travels as LangChain `Document` objects (page_content + metadata), so no text model is needed here.

from typing import Literal  # Literal["a", "b"] = the TS union type 'a' | 'b'.

from pydantic import BaseModel, Field, field_validator  # BaseModel = base class. Field = defaults + descriptions. field_validator = clean input.


# ======================= INGESTION: what one Cosmos item looks like =======================

class ChunkMetadata(BaseModel):  # The "metadata" part of "embeddings + metadata + text". camelCase because it is stored as JSON.
    fileName: str  # Original file name.
    title: str  # Document title.
    department: str  # e.g. "hr", "it", "finance" - given when you ingest. Used by the SQL/metadata search.
    docType: str  # e.g. "policy", "manual", "faq".
    year: int | None = None  # Year of the document, if known. `int | None` = `number | null`.
    page: int  # Page the chunk comes from (for citations).
    chunkIndex: int  # 0, 1, 2... order of the chunk inside its document.
    imageCaptions: list[str] = []  # Captions of images on that page. list[str] = string[] in TS.
    ingestedAt: str  # ISO date-time when we stored it.


class ChunkItem(BaseModel):  # EXACTLY what one item in Cosmos DB looks like.
    id: str  # Unique id of the item (Cosmos requires "id").
    documentId: str  # Partition key: all chunks of one document share it.
    text: str  # The chunk text (what the LLM will read).
    embedding: list[float]  # The vector (1536 floats) that represents the chunk's meaning.
    metadata: ChunkMetadata  # Filterable information.


# ======================= CHAT: structured LLM outputs =======================

class SafetyCheck(BaseModel):  # What the guardrail LLM fills in (structured output).
    reason: str = Field(description="One short sentence: what is the user trying to do?")  # FIRST: the model "thinks" before it decides.
    category: Literal["safe", "prompt_injection", "harmful", "personal_data"] = Field(
        description="safe for any normal work question; otherwise the type of attack or abuse"
    )  # No "allowed" field: OUR code derives it from the category, so the model can't contradict itself.


class GuardrailResult(BaseModel):  # Output of the whole "guardrails (security) filters/LLMs" box (rules + LLM).
    allowed: bool  # True = continue, False = go to "query fails".
    category: Literal["safe", "prompt_injection", "harmful", "personal_data", "too_long", "empty"]  # Why.
    reason: str = ""  # Short explanation (shown in the trace, useful for debugging).


class MetadataFilters(BaseModel):  # Filters the router extracts from the question (for the SQL branch).
    department: str | None = Field(default=None, description="Department, lowercase, only from the known values")
    docType: str | None = Field(default=None, description="Document type, lowercase, only from the known values")
    fileName: str | None = Field(default=None, description="Part of a file name the user mentioned")
    year: int | None = Field(default=None, description="Year of the document, e.g. 2025")
    keyword: str | None = Field(default=None, description="Exact code/term that must appear literally, e.g. 'VPN'")

    @field_validator("*", mode="before")  # Runs for EVERY field, before type checking (like a class-validator @Transform).
    @classmethod  # Pydantic validators are class methods.
    def empty_to_none(cls, value):  # Small models often send "" instead of null for "no filter"...
        return None if isinstance(value, str) and not value.strip() else value  # ...so treat "" / "  " as "not set".

    def is_empty(self) -> bool:  # A method on the model, like a method on a TS class (ignored by the LLM schema).
        return not any(self.model_dump().values())  # True if every filter is None/empty.


class RouteDecision(BaseModel):  # Output of the "intent routing" box.
    intent: Literal["metadata", "semantic", "hybrid"] = Field(
        description="metadata = question ABOUT documents (list/find by department, type, year); "
        "semantic = question about CONTENT without filters; hybrid = content question that also names a filter"
    )
    search_text: str = Field(description="The question rewritten so it is understandable WITHOUT the conversation")
    filters: MetadataFilters = Field(default_factory=MetadataFilters, description="Filters for the SQL search")  # New object each time.
    reason: str = Field(default="", description="One short sentence explaining the choice")


# ======================= CHAT: API / UI models =======================

class Source(BaseModel):  # A citation shown in the UI under the answer.
    number: int  # [1], [2]... the same number the LLM uses in its answer.
    fileName: str  # Which file.
    page: int  # Which page.
    score: float | None = None  # How well it matched (None for SQL results).


class ClientMessage(BaseModel):  # What Angular sends over the WebSocket.
    type: Literal["ask", "reset"]  # "ask" = a question, "reset" = start a new conversation.
    question: str = ""  # The question text (empty for "reset").
    mode: Literal["pipeline", "agent"] = "pipeline"  # pipeline = fixed diagram flow, agent = the LLM chooses the tools.
