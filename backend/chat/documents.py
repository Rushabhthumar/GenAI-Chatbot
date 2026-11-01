# documents.py - converts a Cosmos item into a LangChain Document, so BOTH search branches return the same type.
# Everything downstream (merge, prompt, citations, agent tools) then only deals with Document(page_content, metadata).

from langchain_core.documents import Document  # LangChain's text container.


def to_document(item: dict, source: str, score: float | None = None) -> Document:  # Cosmos dict -> Document.
    return Document(
        page_content=item["text"],  # The chunk text.
        metadata={  # Stored metadata + a few extra fields we need later:
            **item["metadata"],  # fileName, title, department, docType, year, page, chunkIndex... (** = spread, like {...obj}).
            "chunkId": item["id"],  # Cosmos id - used to remove duplicates when merging.
            "score": score,  # Cosine similarity (semantic search) or None (SQL search).
            "source": source,  # "semantic" or "metadata": which branch found it.
        },
    )
