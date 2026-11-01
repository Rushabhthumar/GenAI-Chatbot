# local_store.py - the store interface on a plain JSON file (used when AI_PROVIDER=local). Free, no account needed.
# It does EXACTLY what Cosmos does for us, but by hand, so you can see the mechanics:
#   - vector search  = compute cosine similarity between the question vector and EVERY chunk vector, sort, take top k
#   - metadata search = loop over the items and keep the ones whose metadata matches the filters (our "WHERE")
# Fine for hundreds or a few thousand chunks. Cosmos uses a DiskANN index instead of comparing with everything,
# which is why it stays fast with millions.

import json  # Read/write the JSON file.
import math  # Square root for cosine similarity.
import os  # File/folder checks.

from backend import config  # LOCAL_STORE_PATH, METADATA_LIMIT.
from backend.models import MetadataFilters  # Typed filters.


def _load() -> list[dict]:  # Read all items from the file. A leading _ means "private to this module".
    if not os.path.exists(config.LOCAL_STORE_PATH):  # Nothing ingested yet?
        return []  # Empty store.
    with open(config.LOCAL_STORE_PATH, encoding="utf-8") as file:  # Open for reading.
        return json.load(file)  # List of items, same shape as Cosmos items.


def _save(items: list[dict]) -> None:  # Write all items back to the file.
    os.makedirs(os.path.dirname(config.LOCAL_STORE_PATH) or ".", exist_ok=True)  # Create the data/ folder if needed.
    with open(config.LOCAL_STORE_PATH, "w", encoding="utf-8") as file:  # Open for writing (replaces the file).
        json.dump(items, file)  # Save.


def _without_embedding(item: dict) -> dict:  # Search results don't need the big vector (same as Cosmos SELECT c.id, c.text, c.metadata).
    return {"id": item["id"], "text": item["text"], "metadata": item["metadata"]}


def cosine_similarity(a: list[float], b: list[float]) -> float:  # 1.0 = same direction (same meaning), ~0 = unrelated.
    dot = sum(x * y for x, y in zip(a, b))  # Multiply matching numbers and add them up (the "dot product").
    length_a = math.sqrt(sum(x * x for x in a))  # Length of vector a (Pythagoras in 768 dimensions).
    length_b = math.sqrt(sum(y * y for y in b))  # Length of vector b.
    return dot / (length_a * length_b) if length_a and length_b else 0.0  # Divide by the lengths -> only the DIRECTION counts.


def delete_document(document_id: str) -> int:  # Remove all chunks of one document.
    items = _load()  # Everything.
    kept = [item for item in items if item["documentId"] != document_id]  # Keep the other documents.
    _save(kept)  # Write back.
    return len(items) - len(kept)  # How many were removed.


def upsert_chunks(new_items: list[dict]) -> None:  # Insert or replace items (by id).
    new_ids = {item["id"] for item in new_items}  # Set of ids being written. {.. for ..} = a Set.
    items = [item for item in _load() if item["id"] not in new_ids]  # Drop old versions of these ids...
    _save(items + new_items)  # ...and add the new ones. One file write for the whole batch.


def vector_search(vector: list[float], top_k: int) -> list[dict]:  # Same result shape as Cosmos VectorDistance.
    scored = [  # Score EVERY chunk against the question.
        {**_without_embedding(item), "score": cosine_similarity(vector, item["embedding"])}  # ** = spread, like {...obj, score}.
        for item in _load()
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)  # Most similar first.
    return scored[:top_k]  # Best top_k.


def _matches(item: dict, filters: MetadataFilters) -> bool:  # Our hand-written "WHERE ... AND ...".
    metadata = item["metadata"]  # The item's metadata.
    if filters.department and metadata["department"] != filters.department.lower():  # department = @department
        return False
    if filters.docType and metadata["docType"] != filters.docType.lower():  # docType = @docType
        return False
    if filters.year and metadata.get("year") != filters.year:  # year = @year
        return False
    if filters.fileName and filters.fileName.lower() not in metadata["fileName"].lower():  # CONTAINS(fileName, @fileName, true)
        return False
    if filters.keyword and filters.keyword.lower() not in item["text"].lower():  # CONTAINS(text, @keyword, true)
        return False
    return True  # Every filled filter matched.


def filter_search(filters: MetadataFilters, limit: int = config.METADATA_LIMIT) -> list[dict]:  # Same as the Cosmos SQL branch.
    return [_without_embedding(item) for item in _load() if _matches(item, filters)][:limit]  # Matching items, max `limit`.


def distinct_values(field: str) -> list:  # All distinct values of one metadata field.
    values = {item["metadata"].get(field) for item in _load()}  # A set removes duplicates.
    return sorted(value for value in values if value is not None)  # Without empty values, sorted.
