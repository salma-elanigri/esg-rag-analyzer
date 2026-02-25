import logging
import sys

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

    # 2. Ingestion Pipeline
    pdf_path = "2025-pwc-network-sustainability-report.pdf"

    logging.info(f"Loading PDF: {pdf_path}...")
    pages = loader.load(pdf_path)

    logging.info(f"Splitting {len(pages)} pages...")
    all_chunks = []
    for page in pages:
        splits = splitter.split(page)
        all_chunks.extend(splits)

    logging.info(f"Total chunks generated: {len(all_chunks)}")

    # 3. Store in Vector DB
    logging.info("Saving chunks to ChromaDB (this might take a moment)...")
    vector_store.add_documents(all_chunks)
    logging.info("Data saved successfully!")

    # 4. Retrieval Test (The "Query")
    query = "What are the main environmental risks?"
    logging.info(f"\n=== Querying: '{query}' ===")

    results = vector_store.similarity_search(query, k=3)

    if results:
        logging.info(f"Found {len(results)} relevant chunks:")
        for i, chunk in enumerate(results):
            logging.info(f"\n--- Result {i + 1} ---")
            logging.info(f"Source Page: {chunk.page_number}")
            logging.info(f"Content Preview: {chunk.content[:150]}...")
    else:
        logging.warning("No results found.")


if __name__ == "__main__":
    main()