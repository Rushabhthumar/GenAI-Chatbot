# cosmos_store.py - the store interface on Azure Cosmos DB for NoSQL (used when AI_PROVIDER=azure).
# SECURITY: values ALWAYS go in as @parameters, never glued into the SQL text (no SQL injection).

from backend import config  # METADATA_LIMIT.
from backend.clients import get_container  # Cosmos container (created with the vector policy on first use).
from backend.models import MetadataFilters  # Typed filters.

EXACT_FIELDS = {  # Filter name -> Cosmos path. A whitelist: only these fields can ever be filtered.
    "department": "c.metadata.department",
    "docType": "c.metadata.docType",
    "year": "c.metadata.year",
}

VECTOR_SQL = (  # Cosmos NoSQL query with a vector function:
    "SELECT TOP @topK c.id, c.text, c.metadata, VectorDistance(c.embedding, @vector) AS score "  # Score = cosine similarity (1 = same meaning).
    "FROM c "  # From our chunks container.
    "ORDER BY VectorDistance(c.embedding, @vector)"  # Cosmos orders vector distances most-similar-first automatically.
)


def delete_document(document_id: str) -> int:  # Remove all chunks of one document (before re-ingesting it).
    container = get_container()  # Cosmos container.
    old_ids = container.query_items(  # SQL query, scoped to ONE partition (cheap, because partition key = documentId).
        query="SELECT VALUE c.id FROM c WHERE c.documentId = @documentId",  # VALUE = return plain ids, not objects.
        parameters=[{"name": "@documentId", "value": document_id}],  # @parameter.
        partition_key=document_id,  # Only look inside this document's partition.
    )
    count = 0  # How many we deleted.
    for item_id in old_ids:  # Every old chunk id...
        container.delete_item(item=item_id, partition_key=document_id)  # ...delete it.
        count += 1  # Count it.
    return count  # Number deleted.


def upsert_chunks(items: list[dict]) -> None:  # Insert or replace chunk items.
    container = get_container()  # Cosmos container.
    for item in items:  # One request per item (fine for ingestion; use bulk/batch for very large documents).
        container.upsert_item(item)  # upsert = insert, or replace if the id already exists.


def vector_search(vector: list[float], top_k: int) -> list[dict]:  # BOX "semantic search with cosmosdb".
    return list(get_container().query_items(  # list() = run the query and collect all results.
        query=VECTOR_SQL,  # The query above.
        parameters=[{"name": "@topK", "value": top_k}, {"name": "@vector", "value": vector}],  # Values.
        enable_cross_partition_query=True,  # Search across all documents.
    ))


def build_sql(filters: MetadataFilters, limit: int) -> tuple[str, list[dict]]:  # BOX "sql query": filters -> (SQL, parameters).
    clauses: list[str] = []  # WHERE conditions, joined with AND.
    parameters: list[dict] = [{"name": "@limit", "value": limit}]  # Values, sent separately from the SQL.
    for name, path in EXACT_FIELDS.items():  # .items() = Object.entries().
        value = getattr(filters, name)  # filters.department, filters.docType... getattr = obj[name].
        if value is not None:  # Only filters the router actually filled.
            clauses.append(f"{path} = @{name}")  # e.g. "c.metadata.department = @department"
            parameters.append({"name": f"@{name}", "value": value.lower() if isinstance(value, str) else value})  # Lowercase strings.
    if filters.fileName:  # Partial file-name match.
        clauses.append("CONTAINS(c.metadata.fileName, @fileName, true)")  # true = ignore upper/lower case.
        parameters.append({"name": "@fileName", "value": filters.fileName})
    if filters.keyword:  # Exact word that must appear in the chunk text.
        clauses.append("CONTAINS(c.text, @keyword, true)")  # Case-insensitive "text includes keyword".
        parameters.append({"name": "@keyword", "value": filters.keyword})
    where = " AND ".join(clauses) or "true"  # No clauses -> "WHERE true" (callers avoid this case).
    sql = f"SELECT TOP @limit c.id, c.text, c.metadata FROM c WHERE {where}"  # Never select the embedding - wasteful.
    return sql, parameters  # Return both.


def filter_search(filters: MetadataFilters, limit: int = config.METADATA_LIMIT) -> list[dict]:  # BOX "searches into cosmos db".
    sql, parameters = build_sql(filters, limit)  # Build it safely.
    return list(get_container().query_items(query=sql, parameters=parameters, enable_cross_partition_query=True))  # Run it.


def distinct_values(field: str) -> list:  # All distinct values of one metadata field (field comes from OUR code, never the user).
    query = f"SELECT DISTINCT VALUE c.metadata.{field} FROM c"  # e.g. all departments.
    return list(get_container().query_items(query=query, enable_cross_partition_query=True))  # Run it.
