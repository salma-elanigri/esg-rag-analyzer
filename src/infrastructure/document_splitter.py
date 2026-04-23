from src.core.interfaces import IDocumentSplitter
from typing import List

from src.core.models import DocumentChunk

CHUNK_SIZE = 1000
NEXT_CHUNK_BUFFER_size = 200


class DocumentSplitter(IDocumentSplitter):
    def split(self, page_chunk: DocumentChunk) -> List[DocumentChunk]:
        text_chunks = []
        # Define the sliding window, which will be of 800 characters
        step = CHUNK_SIZE - NEXT_CHUNK_BUFFER_size
        # Sliding window loop
        for start in range(0, len(page_chunk.content), step):
            # extract chunks of 1000 pieces with overlap
            end = start + CHUNK_SIZE
            text_chunk = DocumentChunk(
                content=page_chunk.content[start:end],
                page_number=page_chunk.page_number,
            )
            text_chunks.append(text_chunk)
        return text_chunks
