# ingest.py - runs the WHOLE top half of the diagram for one file:
#   pdfs/docs -> extract (meta, paras, images) -> meta + text -> split 1800/200 -> embeddings + metadata + text -> Cosmos DB
# Usage (from the project folder):
#   python -m backend.ingestion.ingest "C:\docs\leave-policy.pdf" --department hr --doc-type policy --year 2025
# Run this ONCE per document (or again when the document changes). Chat never re-reads the PDF - it only reads Cosmos.

import argparse  # Reads command-line arguments (like yargs/commander in Node).
import re  # Regular expressions.
from datetime import datetime, timezone  # For the "ingestedAt" timestamp.

from backend import config, store  # AI_PROVIDER + the vector store (Cosmos DB or the local JSON file).
from backend.ingestion.chunker import split_into_chunks  # Step 2 (LangChain splitter).
from backend.ingestion.embedder import embed_texts  # Step 3 (LangChain embeddings).
from backend.ingestion.extract import extract_document  # Step 1 (Document Intelligence / pypdf -> LangChain Documents).
from backend.models import ChunkItem, ChunkMetadata  # Shape of a stored item.


def make_document_id(file_name: str) -> str:  # Cosmos ids must not contain / \ ? #, so make a safe id.
    return re.sub(r"[^a-zA-Z0-9_-]", "-", file_name.lower())  # "Leave Policy.pdf" -> "leave-policy-pdf".


def ingest_file(path: str, department: str, doc_type: str, year: int | None) -> int:  # The full ingestion pipeline.
    extractor = "pypdf (local)" if config.USE_LOCAL else "Document Intelligence"  # Which extractor runs.
    print(f"1) Extracting {path} with {extractor}...")  # f"..." = template string `...${x}...`.
    pages = extract_document(path)  # BOXES "extracts like meta, paras, images" + "meta + text": one Document per page.
    if not pages:  # Nothing readable?
        print("   No text found - nothing to ingest.")  # e.g. an empty or broken file.
        return 0  # Stop.
    title = pages[0].metadata["title"]  # Same title on every page.
    print(f"   {len(pages)} pages with text. Title: {title}")  # Progress.

    print("2) Splitting into chunks of max 1800 chars with 200 overlap...")  # BOX: "text -- split them".
    chunks = split_into_chunks(pages)  # LangChain splitter; each chunk keeps its page's metadata.
    print(f"   {len(chunks)} chunks.")  # Count.

    print("3) Creating embeddings...")  # BOX: "embeddings".
    vectors = embed_texts([f"{title}\n{chunk.page_content}" for chunk in chunks])  # Title prefix gives each chunk context.

    target = config.LOCAL_STORE_PATH if config.USE_LOCAL else "Cosmos DB"  # Where it goes.
    print(f"4) Saving embeddings + metadata + text to {target}...")  # BOX: "cosmosdb (vector storage)".
    file_name = chunks[0].metadata["fileName"]  # File name (from the metadata).
    document_id = make_document_id(file_name)  # Partition key value.
    removed = store.delete_document(document_id)  # Re-ingesting? Remove the old chunks first (no duplicates).
    if removed:  # 0 counts as False.
        print(f"   Removed {removed} old chunks of this document.")  # Only printed when re-ingesting.

    now = datetime.now(timezone.utc).isoformat()  # e.g. "2026-10-25T10:00:00+00:00".
    items = []  # All items for this document.
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):  # zip pairs chunk[i] with vector[i]; enumerate adds i.
        item = ChunkItem(  # BOX: "embeddings + metadata + text" - one stored item.
            id=f"{document_id}-{index:04d}",  # e.g. "leave-policy-pdf-0003". :04d = pad with zeros.
            documentId=document_id,  # Partition key.
            text=chunk.page_content,  # TEXT.
            embedding=vector,  # EMBEDDING.
            metadata=ChunkMetadata(  # METADATA: what LangChain carried along + what you passed on the command line.
                fileName=file_name,  # File.
                title=title,  # Title.
                department=department.lower(),  # Store lowercase so filters match regardless of casing.
                docType=doc_type.lower(),  # Same.
                year=year,  # Optional year.
                page=chunk.metadata["page"],  # Page for citations.
                chunkIndex=index,  # Order in the document.
                imageCaptions=chunk.metadata["imageCaptions"],  # Images on that page.
                ingestedAt=now,  # When.
            ),
        )
        items.append(item.model_dump())  # model_dump() = to a plain dict (JSON-ready).
    store.upsert_chunks(items)  # upsert = insert, or replace if the id already exists.

    print(f"Done: {len(chunks)} chunks stored for '{file_name}'.")  # Finished.
    return len(chunks)  # Count for callers.


if __name__ == "__main__":  # Only when run directly (python -m backend.ingestion.ingest ...), not when imported.
    parser = argparse.ArgumentParser(description="Ingest a document into Cosmos DB for the RAG chatbot.")  # CLI definition.
    parser.add_argument("path", help="Path to a PDF / DOCX / image file")  # Required positional argument.
    parser.add_argument("--department", required=True, help="e.g. hr, it, finance")  # Metadata for the SQL branch.
    parser.add_argument("--doc-type", required=True, help="e.g. policy, manual, faq")  # Becomes args.doc_type.
    parser.add_argument("--year", type=int, default=None, help="Year of the document, e.g. 2025")  # Optional.
    args = parser.parse_args()  # Read sys.argv.
    ingest_file(args.path, args.department, args.doc_type, args.year)  # Go.
