# metadata_search.py - DIAGRAM BOXES: "search with metadata" -> "sql query" -> "searches into cosmos db" -> "all matching chunks".
# This is the NON-AI branch: an exact filter on the metadata fields. No embeddings involved.
# The actual query lives in the store: a parameterised SQL query in cosmos_store.py, or a Python filter in local_store.py.
# SECURITY: the LLM only gives us filter VALUES (a typed MetadataFilters object) - it never writes the query itself.

from functools import lru_cache  # Cache the "known values" lookup.

from langchain_core.documents import Document  # Output type.

from backend import store  # Cosmos DB or the local JSON store (decided by AI_PROVIDER).
from backend.chat.documents import to_document  # Store item -> Document.
from backend.models import MetadataFilters  # Input model.


def metadata_search(filters: MetadataFilters) -> list[Document]:  # Run the filter query.
    documents = [to_document(item, source="metadata") for item in store.filter_search(filters)]  # To LangChain Documents.
    return sorted(documents, key=lambda d: (d.metadata["fileName"], d.metadata["chunkIndex"]))  # By file, then position in the file.


@lru_cache  # Cached for the life of the process. Restart the API after ingesting a new department/type.
def get_known_values() -> dict[str, list]:  # Which departments/types/years/files really exist? (Given to the router and agent.)
    return {name: store.distinct_values(name) for name in ["department", "docType", "year", "fileName"]}  # Dict comprehension.
    # e.g. {"department": ["hr", "it"], "docType": ["policy"], "year": [2025], "fileName": [...]}
