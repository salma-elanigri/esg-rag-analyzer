from typing import List
from datetime import datetime
import chromadb
from src.core.interfaces import IVectorStore, IEmbedder
from src.core.models import DocumentChunk


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
    def __init__(self, embedder: IEmbedder, ) -> None:
        self.embedder = embedder
        client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = client.get_or_create_collection(name="esg_reports", embedding_function=None,
                                                          metadata={"description": "ESG reports chunks collection",
                                                                    "created": str(datetime.now())})

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        # generate ids, extract texts and metadata in separate lists
        ids, documents, metadatas = generate_ids(chunks)
        # embed ALL texts at once (Fast)
        embeddings = self.embedder.embed_documents(documents)
        # store texts and embeddings in the collection
        self.collection.add(documents=documents,
                            ids=ids,
                            embeddings=embeddings,
                            metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5) -> List[DocumentChunk]:
        # embedd query as embedding function is None
        query_embeddings = self.embedder.embed_text(query)
        # apply a search using a query
        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        # reconstruct results to DocumnetChunk
        query_result_chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        for doc, meta in zip(documents, metadatas):
            document_chunk = DocumentChunk(content=doc, page_number=meta["page_number"])
            query_result_chunks.append(document_chunk)
        return query_result_chunks
