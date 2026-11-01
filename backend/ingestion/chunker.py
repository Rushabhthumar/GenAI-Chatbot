# chunker.py - DIAGRAM BOX: "text -- split them 1800 + 200 overlap".
# Why split? We can't send a whole 200-page document with every question (token limits, cost, slower, worse focus).
# Why 1800 chars? Roughly 400-450 tokens: one complete idea, still precise when searching.
# Why 200 overlap? If a sentence is cut at a chunk border, the next chunk repeats it, so the meaning is not lost.
#
# LangChain's RecursiveCharacterTextSplitter tries the separators IN ORDER: first it cuts at blank lines (paragraphs),
# only if a piece is still too long at single newlines, then at sentences, then at spaces, and only as a last resort mid-word.
# split_documents() copies each page's metadata (fileName, page...) onto every chunk made from it.

from langchain_core.documents import Document  # LangChain's text container.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # The most used LangChain splitter.

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE  # 1800 and 200.

splitter = RecursiveCharacterTextSplitter(  # Created once, reused for every document.
    chunk_size=CHUNK_SIZE,  # Max 1800 characters per chunk.
    chunk_overlap=CHUNK_OVERLAP,  # Neighbouring chunks share up to 200 characters.
    separators=["\n\n", "\n", ". ", " ", ""],  # Preferred cut points: paragraph > line > sentence > word > character.
)


def split_into_chunks(pages: list[Document]) -> list[Document]:  # Page Documents -> chunk Documents (metadata kept).
    return splitter.split_documents(pages)  # A chunk never spans two pages, so its page citation is always exact.
