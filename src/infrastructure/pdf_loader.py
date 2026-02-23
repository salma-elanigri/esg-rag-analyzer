from typing import List

import pymupdf

from src.core.interfaces import IDocumentLoader
from src.core.models import DocumentChunk


class PdfLoader(IDocumentLoader):
    def load(self, file_path: str) -> List[DocumentChunk]:
        document_chunks = []
        document = pymupdf.open(file_path)
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_text = page.get_text()
            document_chunk = DocumentChunk(content=page_text, page_number=page_index + 1)
            document_chunks.append(document_chunk)
        return document_chunks
