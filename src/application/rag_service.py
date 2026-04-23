import logging
from typing import List, Generator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.core.interfaces import IDocumentSplitter, IVectorStore, ILLMService, IDocumentLoader
from src.core.models import DocumentChunk


def _format_docs(documents: List[DocumentChunk]):
    """This function aim to format a list of documents into a string"""
    return "\n\n---\n\n".join([chunk.page_content for chunk in documents])


class RagService:
    def __init__(self, loader: IDocumentLoader, splitter: IDocumentSplitter, vectorstore: IVectorStore,
                 llm_service: ILLMService):
        self.loader = loader
        self.splitter = splitter
        self.vectorstore = vectorstore
        self.llm_service = llm_service
        self.prompt = ChatPromptTemplate.from_template("""
        You are an intelligent assistant specialized in ESG analysis.
        Use the following pieces of context to answer the question at the end.
        If the answer is not in the context, just say that you don't know.

        Context:
        {context}

        Question: {question}

        Answer:
        """)

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
        self.vectorstore.add_documents(all_chunks)

    def ask(self, question: str) -> Generator[str, None, None]:
        # Build the chain
        chain = (
                {"context": self.vectorstore.as_retriever(search_kwargs={"k": 10}) | RunnableLambda(_format_docs) , "question": RunnablePassthrough()}
                | self.prompt
                | self.llm_service.get_llm()
                | StrOutputParser()
        )
        return chain.stream(question)
