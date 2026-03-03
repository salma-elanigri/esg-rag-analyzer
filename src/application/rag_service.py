import logging
from typing import List, Generator

from src.core.interfaces import IDocumentSplitter, IVectorStore, ILLMService, IDocumentLoader
from src.core.models import DocumentChunk


class RagService:
    def __init__(self, loader: IDocumentLoader, splitter: IDocumentSplitter, vector_store: IVectorStore,
                 llm_service: ILLMService):
        self.loader = loader
        self.splitter = splitter
        self.vector_store = vector_store
        self.llm_service = llm_service

    def ingest(self, file_path: str):
        # 1. load ESG report
        logging.info(f"Loading PDF: {file_path}...")
        pages = self.loader.load(file_path)
        # 2. split document to chunks
        logging.info(f"Splitting {len(pages)} pages...")
        all_chunks = []
        for page in pages:
            splits = self.splitter.split(page)
            all_chunks.extend(splits)
        # 3. store the chunks in vector store
        self.vector_store.add_documents(all_chunks)

    def _build_prompt(self, question: str, chunks: List[DocumentChunk]) -> str:
        """This function aims to create prompt from document chunks retirved by similarity and the question qsked by user"""
        context_text = "\n\n---\n\n".join([chunk.content for chunk in chunks])
        prompt = f"""You are an intelligent assistant specialized in ESG (Environmental, Social, and Governance) analysis.
            Use the following pieces of context to answer the question at the end.
            If the answer is not in the context, just say that you don't know, don't try to make up an answer.

            Context:
            {context_text}

            Question: {question}

            Answer:
            """""
        return prompt

    def ask(self, question: str) -> Generator[str, None, None]:
        # get relevant text chunks obtained by semantic similarity from chromadb
        relevant_chunks = self.vector_store.similarity_search(question, k=10)
        # convert qst to prompt
        prompt = self._build_prompt(question, relevant_chunks)
        # Call LLM service over the reconstructed prompt
        result = self.llm_service.generate(prompt)
        return result
