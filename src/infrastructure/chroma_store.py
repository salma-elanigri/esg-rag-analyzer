from typing import List
from src.core.interfaces import IVectorStore, IEmbedder
from src.core.models import DocumentChunk
from langchain_chroma import Chroma


def generate_ids(chunks: List[DocumentChunk]):
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

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        # generate ids, extract texts and metadata in separate lists
        ids, documents, metadatas = generate_ids(chunks)
        # store texts and embeddings in the collection
        self.vectorstore.add_texts(texts=documents, metadatas=metadatas, ids=ids)

    def as_retriever(self, **kwargs):
        return self.vectorstore.as_retriever(**kwargs)

    def similarity_search(
        self, query: str, search_type: str = "mmr", k: int = 5
    ) -> List[DocumentChunk]:
        # apply a search using a query
        retriever = self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )
        results = retriever.invoke(query)
        # reconstruct results to DocumnetChunk
        query_result_chunks = []
        for doc in results:
            document_chunk = DocumentChunk(
                content=doc.page_content, page_number=doc.metadata["page_number"]
            )
            query_result_chunks.append(document_chunk)
        return query_result_chunks
