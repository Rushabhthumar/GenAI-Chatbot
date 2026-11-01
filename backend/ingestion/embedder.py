# embedder.py - DIAGRAM BOXES: "embeddings + metadata + text" and "query -> embedding".
# An EMBEDDING is a list of numbers (1536 for text-embedding-3-small, 768 for nomic-embed-text) that represents MEANING.
# Texts with similar meaning get similar vectors, even with different words:
#   "How many holidays do I get?"  is close to  "Employees receive 24 days of paid leave"  (no shared words!)
# IMPORTANT: documents and questions MUST be embedded with the SAME model, or the numbers are not comparable.
#
# LangChain's Embeddings interface has exactly two methods, the same for every provider:
#   embed_documents(list_of_texts) -> list of vectors    (batches the API calls for us)
#   embed_query(text)              -> one vector

from backend import config  # USE_LOCAL.
from backend.clients import get_embeddings  # Shared embeddings client (Azure or Ollama).

# nomic-embed-text was TRAINED with these prefixes: stored texts and questions get different labels -> better matches.
# Azure's text-embedding-3 models need no prefix, so they are empty strings there.
DOCUMENT_PREFIX = "search_document: " if config.USE_LOCAL else ""
QUERY_PREFIX = "search_query: " if config.USE_LOCAL else ""


def embed_texts(texts: list[str]) -> list[list[float]]:  # Many texts -> many vectors (same order).
    return get_embeddings().embed_documents([DOCUMENT_PREFIX + text for text in texts])  # vectors[i] belongs to texts[i].


def embed_query(text: str) -> list[float]:  # The question at chat time -> one vector.
    return get_embeddings().embed_query(QUERY_PREFIX + text)  # Same model as the documents.
