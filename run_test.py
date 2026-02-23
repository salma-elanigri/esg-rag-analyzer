import logging
from src.infrastructure.document_splitter import DocumentSplitter
from src.infrastructure.embedder import Embedder
from src.infrastructure.pdf_loader import PdfLoader

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Initialize Adapters
loader = PdfLoader()
splitter = DocumentSplitter()
embedder = Embedder()

# 2. Load
pages = loader.load("2025-pwc-network-sustainability-report.pdf")
logging.info(f"Total pages loaded: {len(pages)}")

# 3. Split
all_splits = []
for page in pages:
    splits = splitter.split(page)
    all_splits.extend(splits)

# 4. Embed
# Take the first 3 chunks
sample_texts = [chunk.content for chunk in all_splits[:3]]
vectors = embedder.embed_document(sample_texts)

logging.info("--- Embedding Test ---")
logging.info(f"Generated {len(vectors)} vectors.")
logging.info(f"Dimension of each vector: {len(vectors[0])}") 
# Should be 384 for MiniLM
logging.info(f"First 5 numbers of vector 1: {vectors[0][:5]}")

# 5. Verify
logging.info(f"Total chunks AFTER split: {len(all_splits)}")

if all_splits:
    logging.info("--- First Split Chunk Preview ---")
    logging.info(f"Content: {all_splits[0].content[:200]}...")
    logging.info(f"Source Page: {all_splits[0].page_number}")

    # Check the overlap: Let's look at the second chunk
    logging.info("--- Second Split Chunk Preview ---")
    logging.info(f"Content: {all_splits[1].content[:200]}...")
    logging.info(f"Source Page: {all_splits[1].page_number}")
else:
    logging.warning("No chunks found.")
