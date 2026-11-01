# extract.py - DIAGRAM BOXES: "pdfs/docs" -> "extracts like meta, paras... images" -> "meta + text".
# Output: LangChain `Document`s, ONE per page = page text (page_content) + metadata (file, title, page, image captions).
#
# Azure mode: Azure AI Document Intelligence ("prebuilt-layout") reads PDF, DOCX, images, scanned pages (built-in OCR)
#             and returns paragraphs with page + role, and figures (images) with captions.
# Local mode: pypdf reads the text layer of a PDF (no OCR, no image captions). .md/.txt files are read as one page.
#
# Why not LangChain's ready-made AzureAIDocumentIntelligenceLoader? Its page mode doesn't give us figure captions,
# and its markdown mode loses page numbers. Writing it ourselves keeps BOTH, and it is only a few lines.

import os  # For os.path.basename / splitext.

from langchain_core.documents import Document  # LangChain's text container: Document(page_content="...", metadata={...}).

from backend import config  # USE_LOCAL switch.

SKIP_ROLES = {"pageHeader", "pageFooter", "pageNumber"}  # Repeated noise on every page - it would pollute every chunk. {..} = a Set.


def make_documents(file_name: str, title: str, page_texts: dict[int, list[str]], page_images: dict[int, list[str]]) -> list[Document]:
    return [  # "meta + text": one Document per page (shared by both extractors).
        Document(
            page_content="\n\n".join(texts),  # Paragraphs separated by a blank line (the splitter prefers to cut there).
            metadata={  # Metadata travels WITH the text through the splitter into every chunk.
                "fileName": file_name,  # File.
                "title": title,  # Title.
                "page": page,  # Page number for citations.
                "imageCaptions": page_images.get(page, []),  # Images on this page. .get(k, default) = map.get(k) ?? default.
            },
        )
        for page, texts in sorted(page_texts.items())  # Pages in order. .items() = Object.entries().
        if any(text.strip() for text in texts)  # Skip pages without any text.
    ]


def extract_with_docintel(path: str) -> list[Document]:  # AZURE: Document Intelligence.
    from backend.clients import get_docintel  # Imported here so local mode never needs Azure settings.

    with open(path, "rb") as file:  # "rb" = read bytes. `with` closes the file automatically (like try/finally).
        poller = get_docintel().begin_analyze_document("prebuilt-layout", body=file)  # Start the analysis (runs in Azure)...
    result = poller.result()  # ...and wait until it is finished (a "long-running operation").

    def page_of(element) -> int:  # Page number of a paragraph/figure.
        return element.bounding_regions[0].page_number if element.bounding_regions else 1  # Ternary: a if cond else b.

    page_texts: dict[int, list[str]] = {}  # page number -> its paragraphs.
    title = None  # Filled from the first paragraph marked as "title".
    for paragraph in result.paragraphs or []:  # `or []` = if Azure returned None, loop over an empty list.
        if paragraph.role in SKIP_ROLES:  # Header/footer/page number?
            continue  # Skip it.
        if paragraph.role == "title" and title is None:  # First title in the document?
            title = paragraph.content  # Remember it.
        page_texts.setdefault(page_of(paragraph), []).append(paragraph.content)  # setdefault = "create the list if missing", then push.

    page_images: dict[int, list[str]] = {}  # page number -> image captions.
    for figure in result.figures or []:  # Every figure/image Document Intelligence detected.
        caption = figure.caption.content if figure.caption else "Image without caption"  # Its caption, if the PDF has one.
        page = page_of(figure)  # Its page.
        page_images.setdefault(page, []).append(caption)  # Keep the caption as metadata...
        page_texts.setdefault(page, []).append(f"[Image on page {page}: {caption}]")  # ...AND as text, so search can find it.

    file_name = os.path.basename(path)  # "C:/docs/leave-policy.pdf" -> "leave-policy.pdf".
    return make_documents(file_name, title or file_name, page_texts, page_images)  # Build the Documents.


def extract_locally(path: str) -> list[Document]:  # LOCAL: pypdf for PDFs, plain reading for .md/.txt.
    file_name = os.path.basename(path)  # File name.
    if os.path.splitext(path)[1].lower() in (".md", ".txt"):  # A text file? splitext("a.md") = ("a", ".md").
        with open(path, encoding="utf-8") as file:  # Read it as text.
            text = file.read()  # Whole file.
        first_line = text.strip().splitlines()[0] if text.strip() else file_name  # First line, e.g. "# Leave Policy".
        title = first_line.lstrip("# ").strip() or file_name  # Remove markdown "# " -> "Leave Policy".
        return make_documents(file_name, title, {1: [text]}, {})  # One "page".

    from pypdf import PdfReader  # Imported here: only needed for PDFs in local mode.

    reader = PdfReader(path)  # Open the PDF.
    page_texts = {number: [page.extract_text() or ""] for number, page in enumerate(reader.pages, start=1)}  # page -> text.
    title = (reader.metadata.title if reader.metadata else None) or file_name  # PDF's own title, else the file name.
    return make_documents(file_name, title, page_texts, {})  # pypdf can't caption images, so none.


def extract_document(path: str) -> list[Document]:  # The function ingest.py calls.
    return extract_locally(path) if config.USE_LOCAL else extract_with_docintel(path)  # Pick by AI_PROVIDER.
