import io
from typing import List, Union

import pymupdf

from src.core.interfaces import IDocumentLoader
from src.core.models import DocumentChunk


class PdfLoader(IDocumentLoader):
    def load(self, file_source: Union[str, io.BytesIO]) -> List[DocumentChunk]:
        document_chunks = []
        # Check if source is a string (path) or bytes (stream)
        if isinstance(file_source, str):
            # a string file path
            document = pymupdf.open(file_source)
        else:
            # a BytesIO
            document = pymupdf.open(stream=file_source, filetype="pdf")

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_text = page.get_text()
            document_chunk = DocumentChunk(content=page_text, page_number=page_index + 1)
            document_chunks.append(document_chunk)
        return document_chunks
