import logging
import sys

from src.application.rag_service import RagService
from src.infrastructure.ollama_service import OllamaService

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Imports
from src.infrastructure.pdf_loader import PdfLoader
from src.infrastructure.document_splitter import DocumentSplitter
from src.infrastructure.embedder import Embedder
from src.infrastructure.chroma_store import ChromaStore
from src.core.models import DocumentChunk


def main():
    logging.info("=== Starting E2E Test ===")

    # 1. Initialize Adapters
    logging.info("Initializing adapters...")
    loader = PdfLoader()
    splitter = DocumentSplitter()
    embedder = Embedder()  # Downloads model if needed
    vector_store = ChromaStore(embedder=embedder)
    llm_service = OllamaService()

    # 2. Initialize the RAG orchestrator
    logging.info("Initializing RAG orchestrator...")
    rag = RagService(loader, splitter, vector_store, llm_service)
    # 3. Ingestion Pipeline
    logging.info("Ingesting report...")
    pdf_path = "2025-pwc-network-sustainability-report.pdf"
    rag.ingest(pdf_path)
    # 4. Ask
    question = "What are the main environmental risks?"
    answer = rag.ask(question)

    logging.info(f"Question: {question}")
    logging.info(f"Answer: {answer}")


if __name__ == "__main__":
    main()
