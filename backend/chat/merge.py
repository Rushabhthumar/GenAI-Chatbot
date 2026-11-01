# merge.py - DIAGRAM BOX: "all matching results" <-> "all matching chunks" (the double arrow = combine both branches).
# The two lists can't be compared by score (SQL hits have no score), so we merge by RANK using
# Reciprocal Rank Fusion (RRF): every list gives each chunk 1 / (60 + its position). Chunks found by BOTH branches add up and rise.
# (LangChain's EnsembleRetriever does the same RRF for several retrievers; we need it for a retriever + a SQL function, so it's 10 lines here.)

from langchain_core.documents import Document  # Chunk type.

from backend import config  # FINAL_CONTEXT_CHUNKS.

RRF_K = 60  # The standard constant from the RRF paper. It stops the #1 result from completely dominating.


def merge_results(semantic_docs: list[Document], metadata_docs: list[Document]) -> list[Document]:  # Two lists -> one ranked list.
    scores: dict[str, float] = {}  # chunk id -> RRF score. dict = a Map/object in TS.
    docs_by_id: dict[str, Document] = {}  # chunk id -> the Document itself (no duplicates).
    for result_list in (semantic_docs, metadata_docs):  # Each branch's results...
        for rank, doc in enumerate(result_list, start=1):  # ...with their position 1, 2, 3...
            chunk_id = doc.metadata["chunkId"]  # Same chunk = same Cosmos id in both lists.
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)  # Add this list's vote. .get(k, 0) = map.get(k) ?? 0.
            docs_by_id.setdefault(chunk_id, doc)  # Keep the first copy (the semantic one, which has a similarity score).
    ranked_ids = sorted(scores, key=scores.get, reverse=True)  # Ids, highest RRF score first.
    return [docs_by_id[i] for i in ranked_ids[:config.FINAL_CONTEXT_CHUNKS]]  # Only the best few go to the LLM (tokens cost money).
