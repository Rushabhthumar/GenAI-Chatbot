# semantic_search.py - DIAGRAM BOXES: "query -> embedding" -> "semantic search with cosmosdb" <-> "cosmosdb (vector storage)"
#                      -> "all matching results".
# Finds chunks by MEANING: the question's vector is compared with the chunk vectors using cosine similarity.
# Azure: Cosmos does the maths with VectorDistance() + a DiskANN index. Local: local_store.py does it in plain Python.
#
# LANGCHAIN CONCEPT: a RETRIEVER = anything that takes a text and returns relevant Documents: retriever.invoke("question").
# We write our own by extending BaseRetriever and implementing ONE method: _get_relevant_documents().
# Because it is a Runnable, it can be used in chains (retriever | ...), traced in LangSmith, and swapped for another retriever.
# (LangChain also ships a ready-made Cosmos vector store, AzureCosmosDBNoSqlVectorSearch; our own retriever keeps
#  full control of the item shape, the score threshold and the documentId partition key.)

from langchain_core.callbacks import CallbackManagerForRetrieverRun  # Passed in by LangChain (for tracing/callbacks).
from langchain_core.documents import Document  # Output type.
from langchain_core.retrievers import BaseRetriever  # The base class every LangChain retriever extends.

from backend import config, store  # TOP_K, MIN_SIMILARITY + the vector store (Cosmos or local).
from backend.chat.documents import to_document  # Store item -> Document.
from backend.ingestion.embedder import embed_query  # Same embedding model as ingestion (must be!).


class VectorStoreRetriever(BaseRetriever):  # A LangChain retriever backed by our store's vector search.
    k: int = config.TOP_K  # How many chunks to fetch. BaseRetriever is a Pydantic model, so settings are typed fields.
    min_score: float = config.MIN_SIMILARITY  # Drop chunks below this similarity.

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        vector = embed_query(query)  # BOX "query -> embedding": the question as a list of numbers.
        items = store.vector_search(vector, self.k)  # BOX "semantic search with cosmosdb": the k most similar chunks.
        return [  # Convert to Documents, keeping only strong matches:
            to_document(item, source="semantic", score=item["score"])
            for item in items
            if item["score"] >= self.min_score  # Vector search ALWAYS returns something, even for "how do I bake a cake?".
        ]  # The threshold turns "closest but unrelated" into "nothing found".


retriever = VectorStoreRetriever()  # One shared instance. Usage: retriever.invoke("how many leave days?")
