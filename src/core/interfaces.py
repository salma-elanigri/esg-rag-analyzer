from abc import ABC, abstractmethod
from typing import List
from src.core.models import DocumentChunk


class IDocumentLoader(ABC):
    """
    This is the PORT.
    Any class that wants to be a 'loader' MUST implement these methods.
    """

    @abstractmethod
    def load(self, file_path: str) -> List[DocumentChunk]:
        """
        The rule: You must accept a file path and return a list of chunks.
        """
        pass


class IDocumentSplitter(ABC):
    @abstractmethod
    def split(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """
        Accepts a single DocumentChunk (e.g., one page).
        Returns a list of smaller DocumentChunks.
        """
        pass


class IEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Embedding a single text
        """
        pass

    @abstractmethod
    def embed_documents(self, text: List[str]) -> List[List[float]]:
        """
        Embedding multiple texts
        """
        pass
class IVectorStore(ABC):
    @abstractmethod
    def add_document(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        pass
    @abstractmethod
    def similarity_search(self, query: str,  k:int=4) -> List[DocumentChunk]:
        pass