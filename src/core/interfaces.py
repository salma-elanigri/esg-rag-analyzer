import io
from abc import ABC, abstractmethod
from typing import List, Union, Generator
from src.core.models import DocumentChunk


class IDocumentLoader(ABC):

    @abstractmethod
    def load(self, file_source: Union[str, io.BytesIO]) -> List[DocumentChunk]:
        """
        Accepts a file path either a string or bytes and return a list of chunks.
        Returns a list of smaller DocumentChunks.
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
    def embed_query(self, text: str) -> List[float]:
        """
        Embedding a single text
        Returns a list of float values.
        """
        pass

    @abstractmethod
    def embed_documents(self, text: List[str]) -> List[List[float]]:
        """
        Embedding multiple texts
        Returns a list of float values.
        """
        pass


class IVectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[DocumentChunk]:
        pass

class ILLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> Generator[str, None, None]:
        pass
