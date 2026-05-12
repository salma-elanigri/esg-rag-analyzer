from typing import List
from src.core.interfaces import IVectorStore, IEmbedder
from src.core.models import DocumentChunk
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever


def _generate_ids(chunks: List[DocumentChunk]):
    """This function prepares data for chromadb add function"""
    ids = []
    documents = []
    metadatas = []
    for idx, chunk in enumerate(chunks):
        documents.append(chunk.content)
        metadatas.append({"page_number": chunk.page_number})
        ids.append(f"doc_{idx}")
    return ids, documents, metadatas


class ChromaStore(IVectorStore):
    def __init__(
        self,
        embedder: IEmbedder,
    ) -> None:
        self.embedder = embedder
        self.vectorstore = Chroma(
            persist_directory="./chroma_db", embedding_function=self.embedder
        )
        self.bm25_retriever = self.build_bm25()
        self.model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.reranker = CrossEncoderReranker(model=self.model, top_n=5)

    def build_bm25(self) -> BM25Retriever | None:
        """Builds BM25 retriever from existing docs in chroma store"""

        docs = self.vectorstore.get()
        if docs["documents"]:
            return BM25Retriever.from_texts(
                texts=docs["documents"], metadatas=docs["metadatas"]
            )
        return None

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        # generate ids, extract texts and metadata in separate lists
        ids, documents, metadatas = _generate_ids(chunks)
        # store texts and embeddings in the collection
        self.vectorstore.add_texts(texts=documents, metadatas=metadatas, ids=ids)

    def as_retriever(self, **kwargs):
        if self.bm25_retriever is None:
            # no docs ingested yet, fallback to dense only
            dense_retriever = self.vectorstore.as_retriever(**kwargs)
            return dense_retriever
        else:
            self.bm25_retriever.k = kwargs.get("search_kwargs", {}).get("k", 5)
            # Dense retriever
            dense_retriever = self.vectorstore.as_retriever(**kwargs)

            # Combine — weights must sum to 1.0
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, dense_retriever], weights=[0.5, 0.5]
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker, base_retriever=ensemble_retriever
            )
            return compression_retriever

    def similarity_search(
        self, query: str, search_type: str = "similarity", k: int = 5
    ) -> List[DocumentChunk]:
        retriever = self.as_retriever(search_type=search_type, search_kwargs={"k": k})
        results = retriever.invoke(query)

        return [
            DocumentChunk(
                content=doc.page_content, page_number=doc.metadata["page_number"]
            )
            for doc in results
        ]
