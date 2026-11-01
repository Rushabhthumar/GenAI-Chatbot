# store package - the "cosmosdb (vector storage)" box, behind ONE small interface with two implementations:
#   cosmos_store.py  -> Azure Cosmos DB for NoSQL (real SQL + VectorDistance + DiskANN index)
#   local_store.py   -> a JSON file + cosine similarity in plain Python (free, for learning on your PC)
# Both expose the same 5 functions, so ingestion and search code never care which one is used
# (the "repository pattern" - like an Angular service with two implementations behind one injection token).
#
#   delete_document(document_id) -> int                 remove all chunks of one document
#   upsert_chunks(items)                                insert or replace chunk items
#   vector_search(vector, top_k) -> list[dict]          most similar chunks, each with a "score"
#   filter_search(filters, limit) -> list[dict]         chunks matching metadata filters (the "sql query" box)
#   distinct_values(field) -> list                      all distinct values of one metadata field

from backend import config  # USE_LOCAL switch.

if config.USE_LOCAL:  # Decided once, at import time.
    from backend.store.local_store import delete_document, distinct_values, filter_search, upsert_chunks, vector_search  # noqa: F401
else:
    from backend.store.cosmos_store import delete_document, distinct_values, filter_search, upsert_chunks, vector_search  # noqa: F401
